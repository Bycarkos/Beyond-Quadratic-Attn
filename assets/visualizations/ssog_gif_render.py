"""Pillow-only rendering helpers for SSOG NFE animations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def _rgb_image(rgb, size):
    array = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array).resize(size, Image.Resampling.BICUBIC)


def _overlay(rgb, field, size, alpha=0.60):
    field = field.detach().float().cpu()
    field = (field - field.min()) / (field.max() - field.min()).clamp_min(1.0e-8)
    field = F.interpolate(
        field[None, None],
        size=(size[1], size[0]),
        mode="bicubic",
        align_corners=False,
    )[0, 0].clamp(0, 1)
    red = (255 * field.sqrt()).byte()
    green = (255 * field.square()).byte()
    blue = (180 * (1 - field) * field.sqrt()).byte()
    heat = Image.fromarray(torch.stack((red, green, blue), -1).numpy(), mode="RGB")
    return Image.blend(_rgb_image(rgb, size), heat, alpha)


def render_probability_frames(rgb, probabilities, class_names, true_label, top_k):
    selected = probabilities[-1].topk(min(top_k, probabilities.shape[-1])).indices
    if true_label not in selected:
        selected = torch.cat((selected[:-1], torch.tensor([true_label])))
    frames = []
    for step in range(probabilities.shape[0]):
        canvas = Image.new("RGB", (1200, 520), "white")
        draw = ImageDraw.Draw(canvas)
        canvas.paste(_rgb_image(rgb, (440, 440)), (25, 55))
        draw.text((25, 20), f"True class: {class_names[true_label]}", fill="black")
        draw.text(
            (500, 20),
            f"ODE classification trajectory - NFE {step}/{probabilities.shape[0]-1}",
            fill="black",
        )
        for row, index in enumerate(selected):
            value = float(probabilities[step, index])
            y = 75 + row * 82
            color = "#ef476f" if int(index) == true_label else "#118ab2"
            draw.text((500, y), class_names[int(index)], fill="black")
            draw.rectangle((500, y + 24, 1150, y + 55), outline="#444444")
            draw.rectangle((500, y + 24, 500 + round(650 * value), y + 55), fill=color)
            draw.text((505 + round(650 * value), y + 31), f"{value:.4f}", fill="black")
        frames.append(canvas)
    return frames


@torch.no_grad()
def render_atom_frames(
    rgb, probabilities, inputs, attention, class_names, query, geometry_fn
):
    qy, qx = query
    heads, atoms = attention.num_heads, attention.num_atoms
    frames = []
    cell_w, cell_h, title_h = 240, 230, 70
    for step, x in enumerate(inputs, start=1):
        fields, weights, mu, sigma = geometry_fn(attention, x, qy, qx)
        canvas = Image.new(
            "RGB", ((atoms + 1) * cell_w, title_h + heads * cell_h), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for head in range(heads):
            y = title_h + head * cell_h
            mixture = (fields[head] * weights[head, :, None, None]).sum(0)
            canvas.paste(_overlay(rgb, mixture, (220, 190)), (10, y + 28))
            draw.text((10, y + 7), f"Head {head} mixture", fill="black")
            for atom in range(atoms):
                x_position = (atom + 1) * cell_w + 10
                canvas.paste(
                    _overlay(rgb, fields[head, atom], (220, 190)),
                    (x_position, y + 28),
                )
                draw.text(
                    (x_position, y + 4),
                    f"H{head} A{atom} l={weights[head, atom].item():.2f} "
                    f"mu=({mu[head, atom, 0].item():.1f},{mu[head, atom, 1].item():.1f}) "
                    f"s=({sigma[head, atom, 0].item():.1f},{sigma[head, atom, 1].item():.1f})",
                    fill="black",
                )
        prediction = int(probabilities[step].argmax())
        mode = "steered" if attention.lookat else "fixed"
        draw.text(
            (15, 18),
            f"SSOG {mode} atoms - NFE {step}/{len(inputs)}, query={query}, "
            f"prediction={class_names[prediction]} ({probabilities[step, prediction]:.3f})",
            fill="black",
        )
        frames.append(canvas)
    return frames
