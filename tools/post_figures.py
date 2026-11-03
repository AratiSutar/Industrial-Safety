"""Turn the engine comparison (data/output/compare_report.json) into a results table and a cost graph.

Usage:
    python tools/compare_engines.py                              # first: writes the report
    python tools/post_figures.py --jev-usd-per-mtok 0.50         # Jev price per million input tokens
    python tools/post_figures.py --jev-usd-per-mtok 0.50 --pc-usd 600 --extra-watts 15

Cost per day, running costs only:
    Jev:   worker checks per day x input tokens per worker check (measured by compare_engines.py) x price
    local: share of the PC price (--pc-usd over --pc-years) + extra power for the PPE model (--extra-watts).
           Both default to 0: the same PC already runs pose tracking, so the PPE model adds CPU time, not a machine.

Writes data/output/post_results_table.png, post_comparison.png and post_cost.png (git-ignored) and prints the
numbers to fill in. Only measured numbers are used: no price, no cost graph.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety.evaluation import break_even_checks, usd_per_check  # noqa: E402

NAMES = {"jev_vlm": "Jev (VLM, no training)", "local_ml": "Off-the-shelf detector (YOLOv8s)"}
COLORS = {"jev_vlm": "#2a78d6", "local_ml": "#eb6834"}  # categorical slots 1 and 2
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def pct_of(value: float | None, part: int, whole: int) -> str:
    return "-" if value is None else f"{100 * value:.0f}% ({part} of {whole})"


def table_rows(report: dict, usd_per_mtok: float | None) -> tuple[list[str], dict[str, list[str]]]:
    """Row labels and one column of text per engine."""
    labels = ["Accuracy (unclear counts as wrong)", "Accuracy when it answered",
              'False "yes" (gear missing, said present)', "Unclear (gate stays shut)",
              "People let in without full PPE"]
    labels += [f"Accuracy: {item.replace('_', ' ')}" for item in report["items"]]
    labels += ["Time per item check (p50 / p95)", "Input tokens per worker check"]
    if usd_per_mtok is not None:
        labels.append("Cost per 1,000 worker checks")
    columns = {}
    for name, r in report["engines"].items():
        o, g = r["overall"], r["gate"]
        lat, tokens = r["latency"]["per_call"], r["tokens"].get("mean_input_per_worker_check")
        correct = o["confusion"]["yes"]["yes"] + o["confusion"]["no"]["no"]
        col = [
            pct_of(o["accuracy"], correct, o["labelled"]),
            pct(o["accuracy_when_answered"]),
            pct_of(o["false_positive_rate"], o["confusion"]["no"]["yes"], o["labelled_missing"]),
            pct(o["unclear_rate"]),
            f"{g['false_unlocks']} of {g['people_missing_ppe']}",
        ]
        col += [pct(r["per_item"][item]["accuracy"]) for item in report["items"]]
        col.append(f"{lat['p50_ms']:.0f} / {lat['p95_ms']:.0f} ms" if lat["count"] else "-")
        col.append(f"{tokens:.0f}" if tokens else "none (runs on this PC)")
        if usd_per_mtok is not None:
            col.append(f"${1000 * usd_per_check(tokens, usd_per_mtok):.2f}" if tokens else "$0 extra")
        columns[name] = col
    return labels, columns


def pyplot():
    import matplotlib

    matplotlib.use("Agg")  # files only: works without a screen (Docker)
    import matplotlib.pyplot as plt

    return plt


def draw_table(report: dict, usd_per_mtok: float | None, out: Path) -> None:
    plt = pyplot()
    labels, columns = table_rows(report, usd_per_mtok)
    names = list(columns)
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(labels) + 0.5), facecolor=SURFACE)
    ax.axis("off")
    table = ax.table(cellText=[[columns[n][i] for n in names] for i in range(len(labels))],
                     rowLabels=labels, colLabels=[NAMES.get(n, n) for n in names],
                     cellLoc="center", rowLoc="left", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK)
        if row == 0:
            cell.get_text().set_weight("bold")
            cell.set_edgecolor(COLORS.get(names[col], GRID) if col >= 0 else GRID)
    dataset = report["dataset"]
    ax.set_title(f"PPE checks on {dataset['evaluated']} labelled photos (one worker each)",
                 color=INK, fontsize=13, pad=4)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def comparison_rows(report: dict) -> list[tuple[str, dict[str, tuple[float, str]]]]:
    """Metric label -> {engine: (share 0-1, 'part of whole')}, for the side-by-side bar chart."""
    def share(part: int, whole: int) -> tuple[float, str]:
        return (part / whole if whole else 0.0), f"{part} of {whole}"

    rows: list[tuple[str, dict[str, tuple[float, str]]]] = []
    specs = [
        ("Right answers", lambda o, g: (o["confusion"]["yes"]["yes"] + o["confusion"]["no"]["no"], o["labelled"])),
        ("Right gate decisions", lambda o, g: (round((g["accuracy"] or 0) * g["people"]), g["people"])),
        ("Unclear answers (gate stays shut)\nlower is better",
         lambda o, g: (sum(o["confusion"][label]["unclear"] for label in ("yes", "no")), o["labelled"])),
        ('False "yes" on missing gear\nlower is better',
         lambda o, g: (o["confusion"]["no"]["yes"], o["labelled_missing"])),
        ("Workers with missing gear let in\nlower is better",
         lambda o, g: (g["false_unlocks"], g["people_missing_ppe"])),
    ]
    for label, counts in specs:
        rows.append((label, {name: share(*counts(r["overall"], r["gate"])) for name, r in report["engines"].items()}))
    return rows


def draw_comparison(report: dict, out: Path) -> None:
    plt = pyplot()
    rows = comparison_rows(report)
    names = list(report["engines"])
    height = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 1.15 * len(rows) + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (_, values) in enumerate(rows):
        for j, name in enumerate(names):
            share, text = values[name]
            y = i + (j - (len(names) - 1) / 2) * height
            ax.barh(y, share, height=height * 0.85, color=COLORS.get(name, MUTED),
                    label=NAMES.get(name, name) if i == 0 else None)
            ax.annotate(f"{100 * share:.0f}%  ({text})", (share, y), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], color=INK, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=len(names), labelcolor=INK,
              fontsize=11)
    ax.set_title(f"Same {report['dataset']['evaluated']} photos, same blurred crops", color=MUTED, fontsize=11,
                 loc="left", pad=34)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def draw_cost(jev_per_check: float, local_per_day: float, out: Path) -> float | None:
    plt = pyplot()
    even = break_even_checks(local_per_day, jev_per_check)
    top = max(1000.0, 2 * even) if even else 1000.0
    x = [0.0, top]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(x, [0, jev_per_check * top], color=COLORS["jev_vlm"], linewidth=2, label=NAMES["jev_vlm"])
    ax.plot(x, [local_per_day] * 2, color=COLORS["local_ml"], linewidth=2, label=NAMES["local_ml"])
    # Direct labels at the right end, in text ink; the line beside them carries the colour
    ax.annotate(f"Jev ${jev_per_check * top:.2f}/day", (top, jev_per_check * top), xytext=(-6, 6),
                textcoords="offset points", ha="right", color=INK, fontsize=10)
    ax.annotate(f"Local ${local_per_day:.2f}/day", (top, local_per_day), xytext=(-6, 6),
                textcoords="offset points", ha="right", color=INK, fontsize=10)
    if even:
        ax.axvline(even, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"break-even ~{even:,.0f} checks/day", (even, 0), xytext=(6, 10),
                    textcoords="offset points", color=MUTED, fontsize=10)
    ax.set_xlim(0, top)
    ax.set_ylim(0, None)
    ax.set_xlabel("Worker checks per day (hat + vest + boots = 1 check)", color=MUTED)
    ax.set_ylabel("Running cost per day (USD)", color=MUTED)
    ax.set_title("PPE check running cost per day", color=INK, fontsize=13, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return even


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default="data/output/compare_report.json", help="from compare_engines.py")
    parser.add_argument("--out-dir", default="data/output", help="where to write the PNGs (git-ignored)")
    parser.add_argument("--jev-usd-per-mtok", type=float, help="Jev price, USD per million input tokens")
    parser.add_argument("--pc-usd", type=float, default=0.0, help="PC price to charge to the PPE model (default 0)")
    parser.add_argument("--pc-years", type=float, default=3.0, help="years the PC is used")
    parser.add_argument("--extra-watts", type=float, default=0.0, help="extra power the PPE model draws, all day")
    parser.add_argument("--kwh-usd", type=float, default=0.30, help="electricity price per kWh")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        sys.exit(f"{path} not found; run tools/compare_engines.py first")
    report = json.loads(path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = out_dir / "post_results_table.png"
    draw_table(report, args.jev_usd_per_mtok, table)
    print(f"Results table: {table}")
    comparison = out_dir / "post_comparison.png"
    draw_comparison(report, comparison)
    print(f"Comparison chart: {comparison}")

    labels, columns = table_rows(report, args.jev_usd_per_mtok)
    for name, col in columns.items():
        print(f"\n{NAMES.get(name, name)}")
        for label, value in zip(labels, col, strict=True):
            print(f"  {label:42} {value}")

    tokens = report["engines"].get("jev_vlm", {}).get("tokens", {}).get("mean_input_per_worker_check")
    if args.jev_usd_per_mtok is None or not tokens:
        print("\nNo cost graph: needs --jev-usd-per-mtok and a report with Jev's token counts.")
        return
    jev_per_check = usd_per_check(tokens, args.jev_usd_per_mtok)
    local_per_day = args.pc_usd / (args.pc_years * 365) + args.extra_watts * 24 / 1000 * args.kwh_usd
    cost = out_dir / "post_cost.png"
    even = draw_cost(jev_per_check, local_per_day, cost)
    print(f"\nCost graph: {cost}")
    print(f"Jev: ${jev_per_check:.5f} per worker check ({tokens:.0f} tokens) | local: ${local_per_day:.2f} per day")
    if even == 0:
        print("The local model costs nothing extra per day, so it is cheaper at any volume.")
    elif even is not None:
        print(f"Break-even: about {even:,.0f} worker checks per day (Jev is cheaper below that).")


if __name__ == "__main__":
    main()
