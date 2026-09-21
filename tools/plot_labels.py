"""Plot auto-labeled second-tap events on a recorded IMU CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tap_recognition.dataset import (
    load_full_recording,
    load_segment_row_bounds,
    load_trigger_events,
)
from tap_recognition.labels import acc_energy_delta, detect_second_tap_frame
from tap_recognition.physics import IMUSimulator


def plot_labels(csv_path: Path, out_path: Path) -> dict[str, int | float]:
    imu, fs = load_full_recording(csv_path)
    events = load_trigger_events(csv_path.with_suffix(".txt"))
    bounds = load_segment_row_bounds(csv_path)
    t = np.arange(len(imu)) / fs
    sim = IMUSimulator(sample_rate=fs, window_samples=300)
    hp = sim.highpass(imu)
    acc_d = acc_energy_delta(hp[:, :3])
    gyro = np.linalg.norm(hp[:, 3:], axis=1)

    t1s: list[int] = []
    dts: list[float] = []
    for (start, end), (n2, _cls) in zip(bounds, events):
        pair = detect_second_tap_frame(
            imu[start:end, :3], imu[start:end, 3:], sample_rate=fs
        )
        if pair is None:
            continue
        n1 = start + int(pair[0])
        t1s.append(n1)
        dts.append((n2 - n1) / fs)

    dt_txt = (
        f"dt median={np.median(dts):.3f}s [{min(dts):.3f}, {max(dts):.3f}]"
        if dts
        else "no pairs"
    )
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 0.9, 0.9, 1.2], hspace=0.38)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    fig.suptitle(f"{csv_path.name}  |  n2={len(events)}/{len(bounds)}  {dt_txt}", fontsize=12)

    ax = axes[0]
    ax.plot(t, acc_d, color="#1f77b4", lw=0.7, label="acc energy Δ")
    for (n2, cls), t1 in zip(events, t1s):
        color = "#2ca02c" if cls == 1 else "#d62728"
        if t1 is not None:
            ax.axvline(t[t1], color="#ff7f0e", ls=":", lw=0.8, alpha=0.8)
        ax.axvline(t[n2], color=color, ls="--", lw=0.9, alpha=0.85)
    ax.set_ylabel("acc Δ")
    ax.set_title("A. High-pass acc energy  (orange=n1, green=left n2, red=right n2)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(t, gyro, color="#9467bd", lw=0.7, label="|ω| HP")
    for n2, cls in events:
        color = "#2ca02c" if cls == 1 else "#d62728"
        ax.axvline(t[n2], color=color, ls="--", lw=0.9, alpha=0.85)
    ax.set_ylabel("|ω|")
    ax.set_title("B. High-pass gyro magnitude")

    ax = axes[2]
    labels_path = csv_path.with_name(csv_path.stem + ".labels.txt")
    if labels_path.exists() and labels_path.stat().st_size > 0:
        dense = np.loadtxt(labels_path, delimiter=",", skiprows=1)
        ax.plot(t, dense[:, 0], color="#7f7f7f", lw=0.6, label="P(none)")
        ax.plot(t, dense[:, 1], color="#2ca02c", lw=1.0, label="P(left)")
        ax.plot(t, dense[:, 2], color="#d62728", lw=1.0, label="P(right)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("soft label")
    ax.set_title("C. Three-class Gaussian labels around n2")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    if bounds:
        z0, z1 = bounds[1] if len(bounds) > 1 else bounds[0]
        sl = slice(z0, z1)
        tz = t[sl]
        ax.plot(tz, acc_d[sl], color="#1f77b4", lw=1.0, label="acc energy")
        for n1 in t1s:
            if z0 <= n1 < z1:
                ax.axvline(t[n1], color="#ff7f0e", ls=":", lw=1.4, label="n1")
        for n2, cls in events:
            if z0 <= n2 < z1:
                color = "#2ca02c" if cls == 1 else "#d62728"
                ax.axvline(t[n2], color=color, ls="--", lw=1.4, label="n2")
        ax.set_title("D. Zoom of one 3s segment (n1 orange, n2 green/red)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acc Δ")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

    fig.subplots_adjust(top=0.92, hspace=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    n_seg = len(load_segment_row_bounds(csv_path))
    return {
        "file": csv_path.name,
        "events": len(events),
        "segments": n_seg,
        "dt_median": float(np.median(dts)) if dts else 0.0,
        "out": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--out-dir", default="demo_outputs")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    for raw in args.csvs:
        csv_path = Path(raw)
        stats = plot_labels(csv_path, out_dir / f"labels_{csv_path.stem}.png")
        print(
            f"{stats['file']}: events={stats['events']}/{stats['segments']} "
            f"Δt={stats['dt_median']:.3f}s -> {stats['out']}"
        )


if __name__ == "__main__":
    main()
