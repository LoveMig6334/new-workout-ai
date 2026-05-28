"""Gather {name -> (AP, size_mb, latency_ms)} into a results table + Pareto plots.

Baseline numbers (RTMPose-t/s/m, MoveNet, MediaPipe) are entered here from their
own benchmark runs / published values; this module only tabulates and plots.
Fairness caveat: MoveNet is COCO-17; MediaPipe BlazePose is 33-kpt — compare on
overlapping joints and annotate cross-topology entries.
"""
import argparse
import json
import matplotlib.pyplot as plt


def plot_frontier(results: dict, out_png: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, r in results.items():
        ax1.scatter(r["latency_ms"], r["AP"]); ax1.annotate(name, (r["latency_ms"], r["AP"]))
        ax2.scatter(r["size_mb"], r["AP"]); ax2.annotate(name, (r["size_mb"], r["AP"]))
    ax1.set_xlabel("latency (ms)"); ax1.set_ylabel("COCO AP"); ax1.set_title("Accuracy vs Latency")
    ax2.set_xlabel("model size (MB)"); ax2.set_ylabel("COCO AP"); ax2.set_title("Accuracy vs Size")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="JSON: {name: {AP, size_mb, latency_ms}}")
    ap.add_argument("--out", default="research/compression/frontier.png")
    a = ap.parse_args()
    plot_frontier(json.loads(open(a.results).read()), a.out)
