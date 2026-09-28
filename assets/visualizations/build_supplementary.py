#!/usr/bin/env python3
"""Build the interactive supplementary explorer (one self-contained HTML file).

Read-only with respect to the code base: loads existing checkpoints through
eval_patch_size.build_model and existing result files, and writes

  experiments/SSOG/supplementary/ssog_explorer.html

Contents embedded as JSON:
  * validation images (JPEG, 224px) with class names;
  * for every SSoG model: the learned Gaussian atoms (mu, sigma, lambda,
    temperature) of every layer, from which the page computes fixed kernels
    for any query patch in the browser;
  * for steered models: the per-query steered atoms of every layer, recomputed
    from each attention layer's input and checked against the module's own
    attention output before being quantized to 8 bits;
  * class probabilities through depth (logit lens: the model's own final
    norm and classifier applied to every intermediate state);
  * the efficiency, FLOPs, robustness, width, ablation and sink results.

python -m assets.visualizations.build_supplementary [--cifar-images 6] [--imagenet-images 4]
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from assets.analysis import analyze_attention_sinks as S
import eval_patch_size as E
from models.DiscreteSSOGViT import DiscreteSSOG_ViT
from models.MatchedViT import MatchedBViT, MatchedViT
from models.SSOGViT import SSOG_ViT, SSOGAttention, _positive

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "supplementary/explorer_template.html"
KERNEL_JS = HERE / "supplementary/kernel.js"
OUTPUT = ROOT / "experiments/SSOG/supplementary/ssog_explorer.html"

# key: (dataset, label, config, evaluation.model, checkpoint or None, kind)
MODELS = {
    "cifar_continuous": ("cifar100", "Continuous · fixed", "evaluation/patch_size/cifar100/continuous.yaml", "continuous", None, "ssog"),
    "cifar_discrete": ("cifar100", "Discrete · fixed", "evaluation/patch_size/cifar100/discrete.yaml", "discrete", None, "ssog"),
    "cifar_looped": ("cifar100", "Looped · fixed", "evaluation/patch_size/cifar100/looped.yaml", "looped", None, "ssog"),
    "cifar_discrete_steered": ("cifar100", "Discrete · steered", "ssog/discrete/cifar100/steered_dim768_h12_r4_depth12.yaml", "discrete", None, "ssog"),
    "cifar_looped_steered": ("cifar100", "Looped · steered", "ssog/looped_ssog/cifar100/steered_dim768_h12_r4_depth12.yaml", "looped", None, "ssog"),
    "cifar_mha": ("cifar100", "Discrete · MHA (ViT)", "evaluation/patch_size/cifar100/dense_discrete.yaml", "dense_discrete", None, "mha"),
    "imagenet_discrete": ("imagenet", "Discrete · fixed", "evaluation/patch_size/imagenet/discrete.yaml", "discrete", None, "ssog"),
    "imagenet_mha": ("imagenet", "Discrete · MHA (ViT-B/16)", "ssog/baselines/imagenet/vit_b16_resume_raw_lr2e4.yaml", "dense_discrete",
                     S.CKPT / "imagenet1k-matched-baselines/imagenet-vit-b16-depth12-raw-resume-lr2e4-seed42/"
                              "286-imagenet-vit-b16-depth12-raw-resume-lr2e4-seed42.pt.ckpt", "mha"),
}
SELECTION_MODELS = {"cifar100": ("cifar_discrete", "cifar_looped"), "imagenet": ("imagenet_discrete",)}
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ---------------------------------------------------------------- loading

def load_model(key):
    dataset, _, config, model_key, checkpoint, _ = MODELS[key]
    cfg = S.load_config(config, model_key, checkpoint)
    path = E.resolve_checkpoint(cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = E.build_model(cfg, ckpt["state_dict"], int(cfg.modeling.inputs.patch_size), "token").eval()
    return model, cfg, int(ckpt.get("epoch", -1))


def load_images(dataset, cfg, count):
    if dataset == "imagenet":
        loader = S.imagenet_loader(count, count)
    else:
        loader = S.cifar_loader(cfg, count, count)
    return next(iter(loader))


def class_names(dataset, cfg):
    if dataset == "imagenet":
        from datasets import Dataset
        from tests.ssog_experiments.common import find_imagenet_arrow_shards
        names = Dataset.from_file(str(find_imagenet_arrow_shards()[0])).features["label"].names
        return [n.split(",")[0] for n in names]
    from torchvision.datasets import CIFAR100
    return list(CIFAR100(root=cfg.data.dataset.dataset_path, train=False, download=False).classes)


def to_jpeg(tensor):
    image = (tensor * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    buffer = io.BytesIO()
    Image.fromarray((image * 255).round().astype(np.uint8)).save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def quantize(array):
    array = np.asarray(array, dtype=np.float32)
    lo, hi = float(array.min()), float(array.max())
    q = np.zeros_like(array, dtype=np.uint8) if hi == lo else np.round((array - lo) / (hi - lo) * 255).astype(np.uint8)
    return {"lo": lo, "hi": hi, "shape": list(array.shape), "b64": base64.b64encode(q.tobytes()).decode()}


# ------------------------------------------------------ geometry and depth

def ssog_modules(model):
    return [m for m in model.modules() if isinstance(m, SSOGAttention)]


def fixed_atoms(attn):
    with torch.no_grad():
        sigma = _positive(attn.raw_sigma, floor=attn.sigma_floor)
        lam = torch.softmax(attn.log_lambda, dim=-1)
        temperature = F.softplus(attn.raw_temperature) + 0.5
    return {"mu": attn.mu.tolist(), "sigma": sigma.tolist(), "lam": lam.tolist(),
            "temperature": float(temperature), "grid": attn.grid_h, "cls": bool(attn.has_cls_token),
            "steered": bool(attn.lookat)}


def steered_atoms(attn, x):
    """Per-query steered atoms, replicating SSOGAttention._steered_apply."""
    gh, gw, h, r = attn.grid_h, attn.grid_w, attn.num_heads, attn.num_atoms
    x_patch = x[:, 1:] if attn.has_cls_token else x
    b = x.shape[0]
    x_grid = x_patch.reshape(b, gh, gw, attn.dim)
    mu, sigma = attn.mu, _positive(attn.raw_sigma, floor=attn.sigma_floor)
    mu_scale = _positive(attn.raw_mu_delta_scale)
    dmu = attn.mu_delta(x_grid).reshape(b, gh, gw, h, r, 2)
    mu_q = mu[None, None, None] + mu_scale * attn.max_offset * torch.tanh(dmu)
    sig_scale = _positive(attn.raw_sigma_delta_scale)
    dsig = attn.sigma_delta(x_grid).reshape(b, gh, gw, h, r, 2)
    sig_q = torch.clamp_min(sigma[None, None, None] * torch.exp(
        sig_scale * attn.max_sigma_log_offset * torch.tanh(dsig)), attn.sigma_floor)
    lam_scale = _positive(attn.raw_lambda_gate_scale)
    gate = attn.lambda_gate(x_grid).reshape(b, gh, gw, h, r)
    lam_q = torch.softmax(attn.log_lambda[None, None, None] + lam_scale * torch.tanh(gate), dim=-1)
    return mu_q, sig_q, lam_q          # [B, gh, gw, H, R, 2], same, [B, gh, gw, H, R]


def kernel_from_atoms(mu, sig, lam, qi, qj, grid, temperature):
    """Dense [H, G, G] map for one query, same formula as kernel.js."""
    d = qi - np.arange(grid)[None, None, :], qj - np.arange(grid)[None, None, :]
    maps = []
    for axis in (0, 1):
        log = -0.5 * math.log(2 * math.pi) - np.log(sig[..., axis:axis + 1]) \
            - (d[axis] - mu[..., axis:axis + 1]) ** 2 / (2 * sig[..., axis:axis + 1] ** 2)
        z = log / temperature
        z = np.exp(z - z.max(-1, keepdims=True))
        maps.append(z / z.sum(-1, keepdims=True))            # [H, R, G]
    return np.einsum("hr,hri,hrj->hij", lam, maps[0], maps[1])


def steered_kernel(mu_q, sig_q, lam_q, qi, qj, grid, temperature):
    """Dense [H, G, G] map of a steered query, same formula as kernel.js.

    SSoG filters rows, then columns, so the row profile reaching query (qi, qj)
    from source (j, k) is taken at the intermediate position (qi, k):
        A[j, k] = sum_r lam(qi,qj) * a_x(qi,qj)[k] * a_y(qi,k)[j]
    (the formula in SSOGAttention._dense_steered_attention). mu_q, sig_q:
    [G, G, H, R, 2]; lam_q: [G, G, H, R].
    """
    def axis(q, mu, sig):
        s = np.arange(grid)
        log = -0.5 * math.log(2 * math.pi) - np.log(sig[..., None]) - (q - s - mu[..., None]) ** 2 / (2 * sig[..., None] ** 2)
        z = log / temperature
        z = np.exp(z - z.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)
    ax = axis(qj, mu_q[qi, qj, ..., 1], sig_q[qi, qj, ..., 1])                     # [H, R, K]
    ay = np.stack([axis(qi, mu_q[qi, k, ..., 0], sig_q[qi, k, ..., 0]) for k in range(grid)])  # [K, H, R, J]
    return np.einsum("hr,hrk,khrj->hjk", lam_q[qi, qj], ax, ay)


def class_trajectory(model, images):
    """Per-layer class probabilities via the model's own head (logit lens)."""
    with torch.no_grad():
        if isinstance(model, SSOG_ViT):
            out = model(images)
            states = out.hidden_states[1:]
            logits = [model.classifier(s[:, 0]) for s in states]
        elif isinstance(model, DiscreteSSOG_ViT):
            out = model(images, output_hidden_states=True)
            logits = [model.classifier(model.final_norm(s).mean(1)) for s in out.hidden_states[1:]]
        elif isinstance(model, (MatchedViT, MatchedBViT)):
            out = model(images, output_hidden_states=True)
            logits = [model.vit.heads(model.vit.encoder.ln(s)[:, 0]) for s in out.hidden_states[1:]]
        else:
            raise TypeError(type(model))
        assert torch.allclose(logits[-1], out.logits, atol=1e-3), "logit lens must end at the model's output"
    return torch.stack([l.softmax(-1) for l in logits], 1)      # [B, L, C]


def trajectory_record(probs, label, top=5):
    final_top = probs[-1].topk(top).indices.tolist()
    classes = list(dict.fromkeys([label] + final_top))[:top + 1]
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(probs.shape[-1])
    return {"classes": classes, "prob": probs[:, classes].T.round(decimals=4).tolist(),
            "entropy": ent.round(decimals=4).tolist(), "pred": int(probs[-1].argmax())}


# ------------------------------------------------------------ result data

def read_csv(path):
    with open(ROOT / path) as handle:
        return list(csv.DictReader(handle))


def results_data():
    from assets.plotting import plot_ablation_horizontal as ablation
    from assets.plotting import plot_efficiency_delta_by_resolution as delta
    from assets.plotting import plot_width_scaling as width
    thr, mem = delta.collapse_by_resolution(*delta.load())
    flops = delta.load_flops_saved()
    patch = {}
    for model in ("continuous", "discrete", "looped"):
        for prefix in ("", "dense_"):
            for r in read_csv(f"experiments/SSOG/patch_size_eval/cifar100/{prefix}{model}.csv"):
                patch.setdefault(model, {}).setdefault(("MHA" if prefix else "SSoG") + ("" if r["geometry"] == "token" else " rescaled"), {})[
                    int(r["patch_size"])] = float(r["acc_1"])
    return {
        "efficiency": {"resolutions": delta.RESOLUTIONS,
                       "throughput": {f: [thr[f][r] for r in delta.RESOLUTIONS] for f in delta.FAMILY_ORDER},
                       "memory": {f: [mem[f][r] for r in delta.RESOLUTIONS] for f in delta.FAMILY_ORDER},
                       "flops_model": [flops["model"][r] for r in delta.RESOLUTIONS],
                       "flops_attention": [flops["attention"][r] for r in delta.RESOLUTIONS]},
        "flops": [{k: (float(v) if k.endswith("gflops") else v) for k, v in r.items()}
                  for r in read_csv("experiments/SSOG/efficiency/flops/flops.csv")],
        "patch": patch,
        "width": {k: {"mha_params": v[0], "ssog_params": v[1], "mha_acc": v[2], "ssog_acc": v[3]}
                  for k, v in width.DATA.items()},
        "ablation": [{"title": t, "xs": xs, "series": s} for _, t, xs, s in ablation.PANELS],
        "sinks": read_csv("experiments/SSOG/attention_sinks/summary.csv"),
    }


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar-images", type=int, default=6)
    parser.add_argument("--imagenet-images", type=int, default=4)
    parser.add_argument("--pool", type=int, default=96, help="validation images scanned when picking examples")
    args = parser.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    loaded = {}
    for key in MODELS:
        loaded[key] = load_model(key)
        print(f"loaded {key} (epoch {loaded[key][2]})", flush=True)

    data = {"models": {}, "datasets": {}, "results": results_data()}
    for dataset, count in (("cifar100", args.cifar_images), ("imagenet", args.imagenet_images)):
        keys = [k for k in MODELS if MODELS[k][0] == dataset]
        cfg = loaded[keys[0]][1]
        images, labels = load_images(dataset, cfg, args.pool)
        # Pick correctly classified examples, preferring ones where the model
        # stays unsure deep into the network (the most instructive trajectories).
        probs = {k: class_trajectory(loaded[k][0], images) for k in SELECTION_MODELS[dataset]}
        correct = torch.stack([p[:, -1].argmax(-1) == labels for p in probs.values()]).all(0)
        lateness = torch.stack([(p[torch.arange(len(labels)), :, labels] < 0.5).float().sum(1) for p in probs.values()]).mean(0)
        order = [i for i in torch.argsort(-lateness).tolist() if correct[i]]
        picked, seen = [], set()
        for i in order:
            if int(labels[i]) not in seen:
                picked.append(i); seen.add(int(labels[i]))
            if len(picked) == count:
                break
        names = class_names(dataset, cfg)
        chosen = images[picked]
        entries = []
        for j, i in enumerate(picked):
            entries.append({"jpeg": to_jpeg(images[i]), "label": int(labels[i]), "name": names[int(labels[i])],
                            "models": {}})
        for key in keys:
            model = loaded[key][0]
            traj = class_trajectory(model, chosen)
            for j in range(len(picked)):
                entries[j]["models"][key] = {"trajectory": trajectory_record(traj[j], entries[j]["label"])}
            if MODELS[key][5] == "ssog":
                modules = ssog_modules(model)
                inputs = []
                handles = [m.register_forward_pre_hook(lambda mod, a, inputs=inputs: inputs.append((mod, a[0].detach())))
                           for m in modules]
                captured = []
                for m in modules:
                    m.capture_attention = True
                    handles.append(m.register_forward_hook(lambda mod, a, out, captured=captured: captured.append(out[1].detach())))
                with torch.no_grad():
                    model(chosen)
                for h in handles:
                    h.remove()
                calls = []
                for (mod, x), attention in zip(inputs, captured):
                    if not mod.lookat:
                        calls.append(None)
                        continue
                    with torch.no_grad():
                        mu_q, sig_q, lam_q = steered_atoms(mod, x)
                    # Check the recomputed atoms against the module's own attention.
                    g = mod.grid_h; off = 1 if mod.has_cls_token else 0
                    temperature = float(F.softplus(mod.raw_temperature.detach()) + 0.5)
                    for qi, qj in ((3, 4), (7, 7), (11, 2)):
                        ours = steered_kernel(mu_q[0].numpy(), sig_q[0].numpy(), lam_q[0].numpy(),
                                              qi, qj, g, temperature)
                        theirs = attention[0, :, off + qi * g + qj, off:].reshape(-1, g, g).numpy()
                        assert np.abs(ours - theirs).max() < 1e-4, f"steered atoms mismatch in {key}"
                    calls.append((mu_q, sig_q, lam_q))
                if any(c is not None for c in calls):
                    for j in range(len(picked)):
                        entries[j]["models"][key]["steered"] = {
                            "mu": quantize(np.stack([c[0][j].numpy() for c in calls])),
                            "logsigma": quantize(np.stack([c[1][j].log().numpy() for c in calls])),
                            "lam": quantize(np.stack([c[2][j].numpy() for c in calls])),
                        }
            label, config, kind = MODELS[key][1], MODELS[key][2], MODELS[key][5]
            record = {"dataset": dataset, "label": label, "kind": kind, "epoch": loaded[key][2],
                      "depth": len(traj[0])}
            if kind == "ssog":
                modules = ssog_modules(model)
                steps = record["depth"]
                # A looped/continuous model reuses one module at every step.
                record["layers"] = [fixed_atoms(modules[min(i, len(modules) - 1)] if len(modules) > 1 else modules[0])
                                    for i in range(steps)]
            data["models"][key] = record
            print(f"  {dataset}/{key}: done", flush=True)
        data["datasets"][dataset] = {"images": entries, "classes": names}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.read_text().replace("/*__KERNEL_JS__*/", KERNEL_JS.read_text()).replace(
        "\"__DATA__\"", payload)
    # A complete standalone document: reviewers open the file directly from disk.
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
    OUTPUT.write_text(html)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
