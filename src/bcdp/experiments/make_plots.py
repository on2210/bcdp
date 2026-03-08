# bcdp/experiments/make_plots.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_layer_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Missing metrics dir: {metrics_dir}")

    files = sorted(metrics_dir.glob("layer_*.json"))
    if not files:
        raise FileNotFoundError(f"No layer_*.json files under {metrics_dir}")

    rows = []
    for fp in files:
        with open(fp, "r") as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: int(r["layer"]))
    return rows


def mean_std(xs: List[float]) -> Tuple[float, float]:
    arr = np.array(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_line_with_band(
    *,
    x: np.ndarray,
    y: np.ndarray,
    band_mean: Optional[np.ndarray],
    band_std: Optional[np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
):
    plt.figure()
    plt.plot(x, y, marker="o")
    if band_mean is not None and band_std is not None:
        lower = band_mean - band_std
        upper = band_mean + band_std
        plt.fill_between(x, lower, upper, alpha=0.2)
        plt.plot(x, band_mean, linestyle="--")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_line(
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def get_nested(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def summarize_random_band(rows: List[Dict[str, Any]], key_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    key_path points to a list of dicts per layer, each dict containing delta_* fields.
    Returns mean/std arrays for that delta field across n_random.
    """
    means = []
    stds = []
    for r in rows:
        rand_list = get_nested(r, key_path, default=None)
        if not isinstance(rand_list, list) or len(rand_list) == 0:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        # The caller will specify which field to read by including it in key_path? no.
        raise ValueError("summarize_random_band expects full field extraction, use summarize_random_field_band.")
    return np.array(means), np.array(stds)


def summarize_random_field_band(rows: List[Dict[str, Any]], list_path: str, field: str) -> Tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for r in rows:
        rand_list = get_nested(r, list_path, default=[])
        vals = []
        if isinstance(rand_list, list):
            for item in rand_list:
                if isinstance(item, dict) and field in item:
                    v = item[field]
                    if v is not None:
                        vals.append(float(v))
        if not vals:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            m, s = mean_std(vals)
            means.append(m)
            stds.append(s)
    return np.array(means, dtype=np.float64), np.array(stds, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path like runs/gemma-2-2b/main_0")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_layer_metrics(run_dir)

    plots_dir = run_dir / "plots"
    ensure_dir(plots_dir)

    layers = np.array([int(r["layer"]) for r in rows], dtype=np.int32)

    # -----------------------
    # 1) k by layer
    # -----------------------
    k = np.array([int(r.get("k", 0)) for r in rows], dtype=np.float64)
    plot_line(
        x=layers,
        y=k,
        title="DBCM rank k by layer",
        xlabel="Layer",
        ylabel="k (mask sum)",
        out_path=plots_dir / "k_by_layer.png",
    )

    # -----------------------
    # 2) Subspace necessity Δmargin vs random band
    # -----------------------
    nec_delta_margin = np.array(
        [float(get_nested(r, "subspace_necessity.delta_margin", 0.0)) for r in rows],
        dtype=np.float64,
    )
    nec_rand_mean, nec_rand_std = summarize_random_field_band(
        rows,
        list_path="subspace_necessity_random",
        field="delta_margin",
    )
    plot_line_with_band(
        x=layers,
        y=nec_delta_margin,
        band_mean=nec_rand_mean,
        band_std=nec_rand_std,
        title="Subspace necessity: Δmargin (project-out) vs random subspaces",
        xlabel="Layer",
        ylabel="Δmargin",
        out_path=plots_dir / "subspace_necessity_delta_margin.png",
    )

    nec_delta_acc = np.array(
        [float(get_nested(r, "subspace_necessity.delta_acc", 0.0)) for r in rows],
        dtype=np.float64,
    )
    nec_rand_acc_mean, nec_rand_acc_std = summarize_random_field_band(
        rows,
        list_path="subspace_necessity_random",
        field="delta_acc",
    )
    plot_line_with_band(
        x=layers,
        y=nec_delta_acc,
        band_mean=nec_rand_acc_mean,
        band_std=nec_rand_acc_std,
        title="Subspace necessity: Δacc (project-out) vs random subspaces",
        xlabel="Layer",
        ylabel="Δacc",
        out_path=plots_dir / "subspace_necessity_delta_acc.png",
    )

    # -----------------------
    # 3) Subspace sufficiency (transfer) Δdonor_acc vs random band
    # -----------------------
    suf_delta_donor_acc = np.array(
        [float(get_nested(r, "subspace_sufficiency_transfer.delta_donor_acc", 0.0)) for r in rows],
        dtype=np.float64,
    )
    suf_rand_mean, suf_rand_std = summarize_random_field_band(
        rows,
        list_path="subspace_sufficiency_transfer_random",
        field="delta_donor_acc",
    )
    plot_line_with_band(
        x=layers,
        y=suf_delta_donor_acc,
        band_mean=suf_rand_mean,
        band_std=suf_rand_std,
        title="Subspace sufficiency (transfer): Δdonor_acc vs random subspaces",
        xlabel="Layer",
        ylabel="Δdonor_acc",
        out_path=plots_dir / "subspace_sufficiency_delta_donor_acc.png",
    )

    suf_delta_donor_margin = np.array(
        [float(get_nested(r, "subspace_sufficiency_transfer.delta_donor_margin", 0.0)) for r in rows],
        dtype=np.float64,
    )
    suf_rand_m_mean, suf_rand_m_std = summarize_random_field_band(
        rows,
        list_path="subspace_sufficiency_transfer_random",
        field="delta_donor_margin",
    )
    plot_line_with_band(
        x=layers,
        y=suf_delta_donor_margin,
        band_mean=suf_rand_m_mean,
        band_std=suf_rand_m_std,
        title="Subspace sufficiency (transfer): Δdonor_margin vs random subspaces",
        xlabel="Layer",
        ylabel="Δdonor_margin",
        out_path=plots_dir / "subspace_sufficiency_delta_donor_margin.png",
    )

    # -----------------------
    # 4) Head ranking summary + head necessity vs random
    # -----------------------
    # We'll summarize head ranking by max score among the top list (if present).
    max_head_score = []
    for r in rows:
        tops = r.get("head_ranking_top", [])
        if isinstance(tops, list) and len(tops) > 0 and isinstance(tops[0], dict) and "score" in tops[0]:
            max_head_score.append(float(tops[0]["score"]))
        else:
            max_head_score.append(np.nan)
    max_head_score = np.array(max_head_score, dtype=np.float64)

    plot_line(
        x=layers,
        y=max_head_score,
        title="Head writer ranking: max score per layer (o_proj slice alignment)",
        xlabel="Layer",
        ylabel="max head score",
        out_path=plots_dir / "head_ranking_max_score.png",
    )

    head_nec_delta_margin = np.array(
        [float(get_nested(r, "head_necessity_topH.delta_margin", 0.0)) for r in rows],
        dtype=np.float64,
    )
    head_rand_mean, head_rand_std = summarize_random_field_band(
        rows,
        list_path="head_necessity_random",
        field="delta_margin",
    )
    plot_line_with_band(
        x=layers,
        y=head_nec_delta_margin,
        band_mean=head_rand_mean,
        band_std=head_rand_std,
        title="Head necessity: Δmargin (ablate top-H heads) vs random heads",
        xlabel="Layer",
        ylabel="Δmargin",
        out_path=plots_dir / "head_necessity_delta_margin.png",
    )

    head_nec_delta_acc = np.array(
        [float(get_nested(r, "head_necessity_topH.delta_acc", 0.0)) for r in rows],
        dtype=np.float64,
    )
    head_rand_acc_mean, head_rand_acc_std = summarize_random_field_band(
        rows,
        list_path="head_necessity_random",
        field="delta_acc",
    )
    plot_line_with_band(
        x=layers,
        y=head_nec_delta_acc,
        band_mean=head_rand_acc_mean,
        band_std=head_rand_acc_std,
        title="Head necessity: Δacc (ablate top-H heads) vs random heads",
        xlabel="Layer",
        ylabel="Δacc",
        out_path=plots_dir / "head_necessity_delta_acc.png",
    )

    # -----------------------
    # 5) Writer necessity vs random + writer ranking summary
    # -----------------------
    max_writer_score = []
    for r in rows:
        tops = r.get("writer_ranking_top", [])
        if isinstance(tops, list) and len(tops) > 0 and isinstance(tops[0], dict) and "score" in tops[0]:
            max_writer_score.append(float(tops[0]["score"]))
        else:
            max_writer_score.append(np.nan)
    max_writer_score = np.array(max_writer_score, dtype=np.float64)

    plot_line(
        x=layers,
        y=max_writer_score,
        title="MLP writer ranking: max score per layer (down_proj column alignment)",
        xlabel="Layer",
        ylabel="max writer score",
        out_path=plots_dir / "writer_ranking_max_score.png",
    )

    writer_nec_delta_margin = np.array(
        [float(get_nested(r, "writer_necessity_topM.delta_margin", 0.0)) for r in rows],
        dtype=np.float64,
    )
    writer_rand_mean, writer_rand_std = summarize_random_field_band(
        rows,
        list_path="writer_necessity_random",
        field="delta_margin",
    )
    plot_line_with_band(
        x=layers,
        y=writer_nec_delta_margin,
        band_mean=writer_rand_mean,
        band_std=writer_rand_std,
        title="MLP writer necessity: Δmargin (ablate top-M writers) vs random writers",
        xlabel="Layer",
        ylabel="Δmargin",
        out_path=plots_dir / "writer_necessity_delta_margin.png",
    )

    writer_nec_delta_acc = np.array(
        [float(get_nested(r, "writer_necessity_topM.delta_acc", 0.0)) for r in rows],
        dtype=np.float64,
    )
    writer_rand_acc_mean, writer_rand_acc_std = summarize_random_field_band(
        rows,
        list_path="writer_necessity_random",
        field="delta_acc",
    )
    plot_line_with_band(
        x=layers,
        y=writer_nec_delta_acc,
        band_mean=writer_rand_acc_mean,
        band_std=writer_rand_acc_std,
        title="MLP writer necessity: Δacc (ablate top-M writers) vs random writers",
        xlabel="Layer",
        ylabel="Δacc",
        out_path=plots_dir / "writer_necessity_delta_acc.png",
    )

    # -----------------------
    # 6) Stability split-half
    # -----------------------
    stab = np.array(
        [float(r["stability_split_half"]) if r.get("stability_split_half") is not None else np.nan for r in rows],
        dtype=np.float64,
    )
    plot_line(
        x=layers,
        y=stab,
        title="Subspace stability: split-half similarity",
        xlabel="Layer",
        ylabel="similarity",
        out_path=plots_dir / "stability_split_half.png",
    )

    # -----------------------
    # 7) Quick summary print
    # -----------------------
    def _best_layer(arr: np.ndarray) -> int:
        if np.all(np.isnan(arr)):
            return -1
        return int(layers[int(np.nanargmax(arr))])

    print(f"[PLOTS] wrote to: {plots_dir}")
    print(f"Best layer by sufficiency Δdonor_acc: { _best_layer(suf_delta_donor_acc) }")
    print(f"Most negative necessity Δmargin (largest drop): { int(layers[int(np.nanargmin(nec_delta_margin))]) }")


if __name__ == "__main__":
    main()