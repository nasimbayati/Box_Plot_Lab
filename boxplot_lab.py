# boxplot_lab.py
"""
Boxplot & Histogram Lab
-----------------------
An original, GitHub‑ready visualization showcasing:
  • Custom histogram + overlaid Gaussian using sample mean/std
  • Horizontal boxplots with custom styling (fliers, caps, notches)
  • A layout and aesthetic intentionally different from typical coursework

Run:
    python boxplot_lab.py

Optionally save a PNG for your README:
    python -c "import boxplot_lab as b; b.main(save_png=True)"
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
import math

# ---- math helpers -----------------------------------------------------------
def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Simple normal PDF; avoids scipy dependency."""
    if sigma <= 0:
        return np.zeros_like(x)
    c = 1.0 / (sigma * math.sqrt(2 * math.pi))
    z = (x - mu) ** 2 / (2 * sigma ** 2)
    return c * np.exp(-z)

# ---- data synthesis ---------------------------------------------------------
def make_mixture(seed: int = 7, n: int = 8000) -> np.ndarray:
    """Create a clearly custom dataset: a trimodal mixture with light tails."""
    rng = np.random.default_rng(seed)
    a = rng.normal(35, 6, size=n // 2)
    b = rng.normal(55, 4, size=n // 3)
    c = rng.normal(75, 3, size=n - len(a) - len(b))
    data = np.concatenate([a, b, c])
    # Tame tails with a small winsorization-like clip
    lo, hi = np.percentile(data, [0.5, 99.5])
    return np.clip(data, lo, hi)

def make_shifted_variant(base: np.ndarray, shift: float = 18.0, scale: float = 0.92,
                         outliers: int = 6, seed: int = 11) -> np.ndarray:
    """Create a related but different sample with shift/scale and injected outliers."""
    rng = np.random.default_rng(seed)
    x = base * scale + shift
    # sprinkle high and low outliers
    lo = np.min(base) - 12
    hi = np.max(base) + 18
    spikes = rng.uniform(low=lo, high=hi, size=outliers)
    return np.concatenate([x, spikes])

# ---- plotting primitives ----------------------------------------------------
def hist_plus_gaussian(ax, data: np.ndarray, bins: int = 40, color: str = "#1565c0"):
    n, edges, _ = ax.hist(data, bins=bins, density=True, color=color, alpha=0.35,
                          edgecolor="#0d47a1", linewidth=0.6)
    mu, sd = float(mean(data)), float(stdev(data))
    xs = np.linspace(edges[0], edges[-1], 400)
    ax.plot(xs, normal_pdf(xs, mu, sd), linewidth=2.0, color="#1b5e20",
            label=f"Normal(mu={mu:.1f}, sd={sd:.1f})")
    ax.grid(True, alpha=0.25)
    return mu, sd


def styled_box(ax, data, *, title="", showfliers=True, notch=False, showcaps=True):
    boxprops = dict(facecolor="#ef6c00", alpha=0.35, edgecolor="#bf360c", linewidth=1.2)
    whiskerprops = dict(color="#6d4c41", linewidth=1.2)
    capprops = dict(color="#3e2723", linewidth=1.2)
    medianprops = dict(color="#c62828", linewidth=1.6)
    flierprops = dict(marker="d", markersize=4, markeredgecolor="#4a148c",
                      markerfacecolor="#ce93d8", alpha=0.8)

    ax.boxplot(
        data,
        vert=False,
        widths=0.5,
        showfliers=showfliers,
        notch=notch,
        showcaps=showcaps,
        patch_artist=True,       # <-- add this line
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        flierprops=flierprops,
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


# ---- main figure ------------------------------------------------------------
def main(save_png: bool = False) -> None:
    base = make_mixture()
    variant = make_shifted_variant(base)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    (ax_h1, ax_h2), (ax_b1, ax_b2) = axes

    # top-left: histogram + Gaussian fit for base
    mu1, sd1 = hist_plus_gaussian(ax_h1, base, bins=45, color="#1976d2")
    ax_h1.set_title("Mixture: Histogram + Gaussian fit")
    ax_h1.set_ylabel("Density")

    # top-right: histogram + Gaussian fit for variant (different palette)
    mu2, sd2 = hist_plus_gaussian(ax_h2, variant, bins=45, color="#00897b")
    ax_h2.set_title("Shifted/Scaled Variant: Histogram + Gaussian fit")

    # bottom-left: clean box without outliers (summary view)
    styled_box(ax_b1, base, title="Boxplot (no fliers)", showfliers=False, notch=False)
    ax_b1.set_xlabel("Value")

    # bottom-right: notched box with fliers hidden caps
    styled_box(ax_b2, variant, title="Notched Boxplot (caps hidden)", notch=True, showcaps=False)
    ax_b2.set_xlabel("Value")

    # unobtrusive legend for PDFs
    ax_h1.legend(loc="upper right", framealpha=0.85)
    ax_h2.legend(loc="upper right", framealpha=0.85)

    fig.suptitle("Boxplot & Histogram Lab — Nasim Bayati", fontsize=14)

    if save_png:
        fig.savefig("boxplot_lab.png", dpi=160, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
