"""Şablon yerleşiminde katı nedensel bilgi erişimini görünür kıl."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), layout="constrained")
    allowed = np.tri(5, dtype=int)
    for ax, labels, row, title in [
        (axes[0], ["e", "c", "MASK", "p1", "p2"], 2, "Makale sırası: e, c, MASK, prompt"),
        (axes[1], ["p1", "p2", "e", "c", "MASK"], 4, "Ayrı prefix kontrolü: prompt, e, c, MASK")]:
        ax.imshow(allowed, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5), labels)
        ax.set_yticks(range(5), labels)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(allowed[i, j]), ha="center", va="center",
                        color="white" if allowed[i, j] else "#536878")
        ax.add_patch(Rectangle((-.5, row-.5), 5, 1, fill=False, edgecolor="#c56b14", linewidth=3))
        ax.set(title=title, xlabel="Görülebilen kaynak konum", ylabel="Bilgi alan konum")
    fig.suptitle("Katı nedensel dikkatte maskeli cevabın görebildiği konumlar\n1: erişim var, 0: erişim yok; turuncu satır maskeli cevaptır", fontsize=13)
    fig.supxlabel("Okunabilirlik için iki pseudo token çizildi; deneyde sekiz kullanılır. Gözlenen geçmiş burada çizilmedi.", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(ROOT / f"outputs/figures/paper_prompt_visibility.{suffix}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
