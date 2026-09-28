# assets/visualizations

Qualitative visualizations of trained SSoG models, and the interactive
supplementary page. Run from the repository root.

## build_supplementary.py (+ supplementary/)

Builds the **interactive supplementary material**: one self-contained HTML file
that opens offline in any browser, makes no network requests and contains no
identifying information.

It lets a reader pick a validation image, a model (continuous, discrete or
looped; fixed or steered), a depth and a head, then click any patch to see:

- where that patch attends, on the image;
- its Gaussian atoms (μ, σ, λ) on the 14 × 14 token grid;
- the class probability after every layer (logit lens), with the standard
  ViT on the same image for comparison;
- fixed and steered kernels side by side;
- the efficiency, FLOPs, robustness, width, ablation and attention-sink results
  as interactive charts and tables.

Fixed kernels are computed in the browser from the learned parameters (checked
against PyTorch to within 10⁻⁷). Steered kernels are recomputed from each
layer's input on the embedded images and checked against the model's own
attention output; the build stops if they disagree. They are stored at 8-bit
precision, which changes the maps by about 1% of the attention mass.

**Run** (CPU; a few minutes)

```bash
python -m assets.visualizations.build_supplementary
python -m assets.visualizations.build_supplementary --cifar-images 6 --imagenet-images 4 --pool 96
```

| Option | Default | |
|---|---|---|
| `--cifar-images`, `--imagenet-images` | 6, 4 | examples embedded per dataset |
| `--pool` | 96 | validation images scanned to pick them (correctly classified, one per class, preferring late decisions) |

**Needs** the checkpoints referenced in the `MODELS` table at the top of the
script, CIFAR-100, the local ImageNet validation shard, and the result files of
`benchmarking/`, `analysis/` and `eval_patch_size.py`.

**Writes** `experiments/SSOG/supplementary/ssog_explorer.html` (about 10 MB).
Open that file; `supplementary/explorer_template.html` is only the template
and shows a message pointing to the built file.

**Files in `supplementary/`**

- `explorer_template.html`: layout, styles and page logic; the build fills in
  the data and the kernel code.
- `kernel.js`: the browser implementation of the fixed and steered SSoG kernels,
  mirroring `SSOGAttention` in `models/SSOGViT.py`.

Rebuild after any checkpoint or result changes; the page lists the checkpoint
epoch of every model.

## visualize_ssog_nfe.py (+ ssog_gif_render.py)

Animated GIFs of one continuous SSoG model across ODE steps: the class
probabilities and the Gaussian atoms of a query patch at every step.
`ssog_gif_render.py` holds the Pillow drawing helpers it uses.

```bash
sbatch scripts/supervised/visualization/visualize_ssog_nfe.sh <model-config> <checkpoint> [output-dir]
python -m assets.visualizations.visualize_ssog_nfe \
  --model-config ssog/cifar100/fixed_r4_nfe12.yaml --checkpoint <file.ckpt> \
  --indices 0 1 2 3 --query 7 7
```

| Option | Default | |
|---|---|---|
| `--model-config`, `--checkpoint` | required | training config (under `configs/classification`) and checkpoint |
| `--indices` | `0 ...` | CIFAR-100 validation images |
| `--query` | model default | query patch (row, column) |
| `--top-k` | 5 | classes shown in the probability panel |
| `--duration` | 0.7 | seconds per frame |
| `--output-dir` | `artifacts/ssog_nfe_gifs` | |

**Writes** one probability GIF and one atom GIF per image, plus a
`metadata.yaml` describing the run.

## visualize_ssog_embeddings.py

PCA and t-SNE of the final representations of an SSoG model on CIFAR-100,
coloured by class.

```bash
sbatch scripts/supervised/visualization/visualize_ssog_embeddings.sh <model-config> <checkpoint> [output-dir]
python -m assets.visualizations.visualize_ssog_embeddings \
  --model-config <config.yaml> --checkpoint <file.ckpt> --output-dir artifacts/ssog_embeddings
```

| Option | Default | |
|---|---|---|
| `--model-config`, `--checkpoint`, `--output-dir` | required | |
| `--samples-per-class` | 25 | |
| `--batch-size` | 128 | |
| `--perplexity`, `--tsne-iterations` | 30, 1500 | t-SNE settings |
| `--legend-classes` | 20 | classes named in the legend |
| `--seed` | 42 | |

**Writes** PCA and t-SNE plots, `cluster_metrics.json` (cluster-separation
metrics) and a compressed `.npz` of the embeddings to the output directory. A
GPU is used when available; t-SNE itself runs on CPU.
