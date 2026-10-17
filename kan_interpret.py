"""
kan_interpret.py
================
Spline-curve figure for the KAN branch of HybridKAN (see kan_hybrid.py).

Each edge i -> o of a KANLayer is a learned univariate function

    phi_{o,i}(x) = w_base[o,i] * silu(x) + sum_c w_spline[o,i,c] * B_c(x)

Plotting phi for the most relevant features shows *what shape* the model learned
for each feature (monotone, threshold, U-shaped, ...). An MLP cannot show this.

Usage in main.py (after training, model in eval mode):

    from kan_interpret import plot_kan_splines
    plot_kan_splines(
        cae.hybrid.kan_branch, X_train, X_df.columns.tolist(),
        out_path=os.path.join(OUT_DIR, "kan_splines"),   # writes .png and .pdf
        top_k=6, device=DEVICE,
    )

Requires: torch, numpy, matplotlib, and kan_hybrid.py.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless, safe on HPC nodes
import matplotlib.pyplot as plt
import numpy as np
import torch

from kan_hybrid import KAN, KANLayer


@torch.no_grad()
def edge_curve(layer: KANLayer, i: int, o: int, x_grid: torch.Tensor) -> torch.Tensor:
    """phi_{o,i} evaluated on x_grid [G] -> [G]. Same maths as KANLayer.forward."""
    x = torch.zeros(x_grid.numel(), layer.in_features, device=x_grid.device, dtype=x_grid.dtype)
    x[:, i] = x_grid
    base = layer.base_activation(x[:, i]) * layer.base_weight[o, i]       # [G]
    bases = layer.b_splines(x)[:, i, :]                                   # [G, C]
    return base + bases @ layer.spline_weight[o, i]                       # [G]


def _grid_limits(layer: KANLayer) -> tuple[float, float]:
    """Domain covered by the splines (outside it only the SiLU base term is active)."""
    k = layer.spline_order
    return layer.grid[0, k].item(), layer.grid[0, -k - 1].item()


@torch.no_grad()
def plot_kan_splines(
    kan: KAN,
    X: np.ndarray,
    feature_names: list[str],
    out_path: str,
    top_k: int = 6,
    n_cols: int = 3,
    n_score_samples: int = 512,
    n_other_edges: int = 3,
    share_y: bool = True,
    device: str | torch.device = "cpu",
    dpi: int = 300,
) -> list[int]:
    """
    Plot the learned edge function for the `top_k` most relevant input features.

    kan:      a single-layer KAN, e.g. cae.hybrid.kan_branch  (width [n_features, k]).
    X:        the *model-input* features, already scaled the way the model saw them
              (for you that is X_train, the StandardScaler output).
    out_path: path WITHOUT extension. Writes <out_path>.png and <out_path>.pdf.

    For each feature the bold curve is its strongest edge (largest std across the
    data, i.e. the edge that moves the most); thin grey curves are its next
    strongest edges. The rug at the bottom shows where the real data lies, so you
    can tell supported regions from extrapolation.

    Returns the plotted feature indices, most relevant first.
    """
    assert len(kan.layers) == 1, "plot_kan_splines expects a single-layer KAN (the branch)"
    layer = kan.layers[0]
    n_feat = layer.in_features
    assert len(feature_names) == n_feat, "feature_names must match the KAN input width"

    was_training = kan.training
    kan.eval()
    dev = torch.device(device)
    xb = torch.as_tensor(np.asarray(X[:n_score_samples]), dtype=torch.float32, device=dev)

    # feature ranking and per-edge relevance [out, in] from one attribution pass
    scores = kan.feature_score(xb).cpu().numpy()                 # [n_feat], sums to 1
    edge_rel = layer.edge_relevance.cpu().numpy()                # [out, in]
    order = np.argsort(-scores)[: min(top_k, n_feat)]

    lo, hi = _grid_limits(layer)
    X_np = np.asarray(X)

    n_cols = min(n_cols, len(order))
    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.7 * n_rows), squeeze=False)

    for ax, i in zip(axes.ravel(), order):
        col = X_np[:, i]
        p1, p99 = np.percentile(col, [0.5, 99.5])
        pad = 0.1 * (p99 - p1 + 1e-8)
        x_lo, x_hi = max(lo, p1 - pad), min(hi, p99 + pad)
        grid = torch.linspace(x_lo, x_hi, 300, device=dev)

        outs = np.argsort(-edge_rel[:, i])
        for o in outs[1 : 1 + n_other_edges]:
            ax.plot(grid.cpu(), edge_curve(layer, i, int(o), grid).cpu(),
                    color="0.75", lw=1.0, zorder=1)
        best = int(outs[0])
        ax.plot(grid.cpu(), edge_curve(layer, i, best, grid).cpu(),
                color="#1f4e9c", lw=2.2, zorder=2)

        # rug of real data, on a subsample so the PDF stays light. y is in axes
        # coordinates, so it stays at the bottom whatever the y-limits end up being.
        rng = np.random.default_rng(0)
        sample = rng.choice(col, size=min(300, len(col)), replace=False)
        sample = sample[(sample >= x_lo) & (sample <= x_hi)]
        ax.plot(sample, np.full_like(sample, 0.02), "|", color="0.35", ms=5, alpha=0.5,
                zorder=0, transform=ax.get_xaxis_transform())

        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_title(f"{feature_names[i]}  (score {scores[i]:.2f})", fontsize=9)
        ax.set_xlabel("scaled feature value", fontsize=8)
        ax.set_ylabel(rf"$\phi(x)$ to unit {best}", fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    used = axes.ravel()[: len(order)]
    if share_y:
        # one common y-range, so a flat curve reads as "unimportant" instead of
        # being stretched to fill its panel
        y_lo = min(a.get_ylim()[0] for a in used)
        y_hi = max(a.get_ylim()[1] for a in used)
        for a in used:
            a.set_ylim(y_lo, y_hi)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)

    if was_training:
        kan.train()
    return order.tolist()


# --------------------------------------------------------------------------- #
#  Self-test: can the plot recover functions we know the answer to?
#     y = 1[ x0^2 + 1.5*sin(2*x1) > 1 ]   (x2..x7 are pure noise)
#  Expect x0 -> U-shape, x1 -> sine wave, noise features ranked low.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import torch.nn as nn

    torch.manual_seed(0)
    np.random.seed(0)
    N, D = 4000, 8
    X = np.random.randn(N, D).astype(np.float32)
    y = ((X[:, 0] ** 2 + 1.5 * np.sin(2 * X[:, 1])) > 1.0).astype(np.int64)

    class Toy(nn.Module):
        """Same shape as HybridKAN's symbolic path: kan_branch -> final_kan."""
        def __init__(self):
            super().__init__()
            self.kan_branch = KAN([D, 4], grid_size=8, grid_range=(-3.5, 3.5))
            self.final_kan = KAN([4, 2], grid_size=8, grid_range=(-3.5, 3.5))
        def forward(self, x):
            return self.final_kan(self.kan_branch(x))

    model = Toy()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for step in range(600):
        idx = torch.randint(0, N, (256,))
        loss = nn.functional.cross_entropy(model(Xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    acc = (model(Xt).argmax(1) == yt).float().mean().item()
    print(f"toy accuracy: {acc:.3f}")

    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kan_splines_selftest"
    order = plot_kan_splines(
        model.kan_branch, X, [f"x{i}" for i in range(D)], out, top_k=6, n_cols=3
    )
    print("top features:", order, "(true signal is x0 and x1)")
    print("wrote", out + ".png")
