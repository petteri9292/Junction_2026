"""Plot actual assistant-reviewed development cutoff tradeoffs from analyzer output.

No inference or model loading. Undefined metrics remain missing, never zero.
Example:
  .venv/bin/python scripts/plot-reranker-cutoff.py \
    --analysis reports/cutoff-development/analysis.json \
    --output reports/reranker-cutoff-development
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def metric(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Expected a finite fraction or null metric; got {value!r}")
    return value


def prepare(analysis, caps):
    if analysis.get("split") != "development":
        raise ValueError(
            "This plot is for development analysis; held-out data must not guide cutoff selection"
        )
    if analysis.get("labels_used", 0) <= 0:
        raise ValueError(
            "No reviewed labels are present; no quality chart will be generated"
        )
    series = {cap: {} for cap in caps}
    for result in analysis["results"]:
        policy = result["policy"]
        cap = int(policy["cap"])
        if policy["kind"] != "absolute" or cap not in series:
            continue
        cutoff = policy["value"]
        cutoff = None if cutoff is None else float(cutoff)
        if cutoff is not None and not math.isfinite(cutoff):
            raise ValueError("Cutoff must be finite or null for the no-cutoff baseline")
        if cutoff in series[cap]:
            raise ValueError(f"Duplicate absolute cutoff/cap pair: {cutoff}, {cap}")
        summary = result["summary"]
        series[cap][cutoff] = {
            "conditional_precision": metric(summary.get("micro_conditional_precision")),
            "reviewed_positive_coverage": metric(
                summary.get("reviewed_pool_positive_coverage")
            ),
            "selected_reviewed_count": summary.get("selected_reviewed_count"),
            "selected_unreviewed_count": summary.get("selected_unreviewed_count"),
            "selected_review_fraction": metric(summary.get("selected_review_fraction")),
            "queries": summary.get("queries"),
        }
    if not any(
        point[key] is not None
        for points in series.values()
        for point in points.values()
        for key in ["conditional_precision", "reviewed_positive_coverage"]
    ):
        raise ValueError(
            "Every requested quality metric is undefined; no chart will be generated"
        )
    finite = sorted(
        {
            cutoff
            for points in series.values()
            for cutoff in points
            if cutoff is not None
        }
    )
    if not finite:
        raise ValueError("No absolute-cutoff observations are available to plot")
    return series, finite


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/reranker-cutoff-tradeoff"),
        help="Output filename prefix; creates .png, .svg and a plotted-data .json sidecar",
    )
    parser.add_argument("--caps", type=int, nargs="+", default=[10, 25])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(set(args.caps)) != len(args.caps) or any(cap < 1 for cap in args.caps):
        raise ValueError("Caps must be distinct positive integers")
    raw = args.analysis.read_bytes()
    analysis = json.loads(raw)
    series, cutoffs = prepare(analysis, args.caps)
    prefix = (
        args.output.with_suffix("")
        if args.output.suffix.lower() in (".png", ".svg")
        else args.output
    )
    paths = {suffix: Path(str(prefix) + suffix) for suffix in [".png", ".svg", ".json"]}
    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(
            "Output already exists; choose a new prefix or pass --overwrite"
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Plotting libraries are imported only after checking that reviewed data exist.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    span = max(2.0, cutoffs[-1] - cutoffs[0])
    baseline_x = cutoffs[0] - max(1.5, span * 0.13)
    xs = [baseline_x] + cutoffs
    colors = ["#226c5c", "#bf792c", "#526ba0", "#934b6b"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharex=True, sharey=True)
    fields = [
        ("conditional_precision", "Conditional relevance precision"),
        ("reviewed_positive_coverage", "Reviewed-positive coverage"),
    ]
    plotted = []
    for axis, (field, title) in zip(axes, fields):
        for index, cap in enumerate(args.caps):
            points = series[cap]
            ys = []
            for cutoff in [None] + cutoffs:
                value = points.get(cutoff, {}).get(field)
                ys.append(float("nan") if value is None else value)
                if value is not None:
                    plotted.append(
                        {
                            "cap": cap,
                            "cutoff": cutoff,
                            "metric": field,
                            "value": value,
                            **{
                                key: points[cutoff][key]
                                for key in [
                                    "selected_reviewed_count",
                                    "selected_unreviewed_count",
                                    "selected_review_fraction",
                                    "queries",
                                ]
                            },
                        }
                    )
            if all(math.isnan(value) for value in ys):
                continue
            color = colors[index % len(colors)]
            # NaN creates a gap rather than fabricating/interpolating an undefined point.
            axis.plot(
                xs,
                ys,
                color=color,
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                label=f"At most {cap} passages",
            )
            if not math.isnan(ys[0]):
                axis.scatter(
                    [baseline_x],
                    [ys[0]],
                    color=color,
                    marker="D",
                    s=45,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5,
                )
        axis.set_title(title, fontsize=13, pad=12)
        axis.set_xlabel("Minimum raw BGE reranker logit", labelpad=11)
        axis.set_xticks(
            xs,
            ["No cutoff\n(baseline)"] + [format(x, "g") for x in cutoffs],
            fontsize=9,
        )
        axis.set_ylim(-0.035, 1.035)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", color="#dfe5df", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(
        "Passage cutoff tradeoffs · development review",
        fontsize=17,
        x=0.065,
        ha="left",
        y=0.98,
    )
    note = (
        "Assistant-reviewed conditional metrics · grades ≥2 count as useful evidence.\n"
        "Missing/uncertain labels remain unknown; undefined points are omitted. Coverage is relative to\n"
        "reviewed positives, not corpus recall. Scores are logits, not relevance probabilities."
    )
    fig.text(0.065, 0.015, note, fontsize=9, color="#4d5d53", va="bottom")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.25, wspace=0.16)
    metadata = {
        "Title": "Reranker cutoff development tradeoffs",
        "Description": "Assistant-reviewed conditional relevance metrics. Not probabilities or corpus recall. Source SHA256="
        + hashlib.sha256(raw).hexdigest(),
    }
    fig.savefig(paths[".png"], dpi=180, bbox_inches="tight", metadata=metadata)
    fig.savefig(paths[".svg"], bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    sidecar = {
        "analysis_source": str(args.analysis),
        "analysis_sha256": hashlib.sha256(raw).hexdigest(),
        "configuration_sha256": analysis.get("configuration_sha256"),
        "split": "development",
        "caps": args.caps,
        "points": plotted,
        "undefined_points": "omitted, never zero-filled",
        "scope": "Assistant-reviewed conditional precision and reviewed-positive-pool coverage. Not corpus recall or calibrated probabilities.",
    }
    paths[".json"].write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
