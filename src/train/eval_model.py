"""Evaluation: metrics dict from a loader + confusion-matrix PNG writer.

evaluate() computes every number THROUGH the registry's metrics module so
the owner's core/metrics.py drop-in changes training and evaluation at
once. The majority-class baseline is returned alongside so every report can
print it next to every accuracy (an accuracy is only meaningful relative to
it on this long-tailed data).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data

from src import registry

#: k for the reported "top-5" number (duplicated from top-1 when C <= 5)
TOPK = 5


def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device | str,
             num_verbs: int) -> dict:
    """Eval-mode metrics for a model over a loader.

    Inputs:
        model: nn.Module mapping clips (B,T,C,H,W) float32 -> logits
            (B, num_verbs) float32 (VerbModel). Left in eval() mode on
            return (BatchNorm running stats — the batch-size-1 trap).
        loader: DataLoader yielding (clips (B,T,C,H,W), labels (B,)).
        device: torch.device or 'cpu'/'cuda' str; clips/labels moved there.
        num_verbs: int C, number of classes.

    Output: dict with keys
        'top1': float, top-1 accuracy (registry topk_accuracy, k=1)
        'top5': float, top-5 accuracy, or a DUPLICATE of top1 when
            num_verbs <= 5 (there is no distinct 2nd..5th class to miss)
        'note': str, present only in the duplicated case, explaining it
        'per_class': list[float] length C, per-class accuracy (0.0 for
            classes absent from the loader's labels)
        'mean_class': float, mean of per_class over classes PRESENT in y
        'majority_baseline': float, always-predict-most-frequent accuracy
        'confusion': list[list[int]] (C x C), rows = true, cols = predicted
        'n_samples': int, number of samples evaluated
    """
    device = torch.device(device)
    metrics = registry.get("metrics")

    model.eval()  # BN running stats; without this, batch-1 batches lie
    all_scores, all_y = [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)     # (B,T,C,H,W) float32
            labels = labels.to(device)   # (B,) int64
            logits = model(clips)        # (B, num_verbs) float32
            all_scores.append(logits.float().cpu().numpy())  # (B,C) float32
            all_y.append(labels.cpu().numpy())               # (B,) int64

    if not all_scores:  # empty loader: well-defined zeros, never a crash
        return {"top1": 0.0, "top5": 0.0,
                "note": "empty loader: 0 samples evaluated",
                "per_class": [0.0] * num_verbs, "mean_class": 0.0,
                "majority_baseline": 0.0,
                "confusion": [[0] * num_verbs for _ in range(num_verbs)],
                "n_samples": 0}

    scores = np.concatenate(all_scores)  # (N, C) float32
    y = np.concatenate(all_y)            # (N,) int64
    pred = scores.argmax(axis=1)         # (N,) int64

    top1 = metrics.topk_accuracy(scores, y, k=1)
    result: dict = {"top1": float(top1)}
    if num_verbs >= TOPK:
        result["top5"] = float(metrics.topk_accuracy(scores, y, k=TOPK))
    else:
        # With C <= 5 classes, top-5 always contains the top-1 prediction, so
        # the number is definitionally equal to top-1; report it duplicated
        # with an explicit note instead of pretending a second stat exists.
        result["top5"] = float(top1)
        result["note"] = (f"top5 == top1 because num_verbs ({num_verbs}) "
                          f"<= {TOPK}")
    result["per_class"] = metrics.per_class_accuracy(
        pred, y, num_verbs).tolist()                       # list[float] (C,)
    result["mean_class"] = float(
        metrics.mean_class_accuracy(pred, y, num_verbs))  # scalar float
    result["majority_baseline"] = float(metrics.majority_baseline(y))
    result["confusion"] = metrics.confusion_matrix(
        pred, y, num_verbs).tolist()                       # (C,C) nested list
    result["n_samples"] = int(y.shape[0])
    return result


def save_confusion_png(cm, class_names: list, path) -> None:
    """Render a confusion matrix to PNG with plain matplotlib (Agg backend).

    Inputs:
        cm: array-like (C, C) ints; rows = true class, cols = predicted
            (the registry convention).
        class_names: list[str] length C, tick labels (rows AND columns).
        path: output .png path (parents created).

    Output: None (file written; 300 dpi, cell counts annotated).
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe (WSL2 without DISPLAY)
    import matplotlib.pyplot as plt
    from pathlib import Path

    cm = np.asarray(cm)  # (C, C) numeric
    C = cm.shape[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * C + 2),
                                    max(3.5, 0.5 * C + 1.5)))
    im = ax.imshow(cm, cmap="Blues")  # rows top->bottom = true classes
    ax.set_xticks(range(C), class_names, rotation=45, ha="right")
    ax.set_yticks(range(C), class_names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("confusion matrix (rows = true)")
    vmax = cm.max() if cm.size else 1
    for i in range(C):        # one text per cell: white on dark cells
        for j in range(C):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
