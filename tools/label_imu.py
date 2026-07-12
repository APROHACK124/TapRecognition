import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import SpanSelector

DEFAULT_CSV_FILE = r'data\train_data\imu_ZR_knock_twice_20260712_103508.csv'
ACTION_LABEL = 1  # 1: the second tap


def calculate_advanced_energy(df):
    df['frame'] = range(len(df))
    df['gyro_energy'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    acc_mag = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['acc_energy_delta'] = np.abs(np.diff(acc_mag, prepend=acc_mag[0]))

    return df


def label_imu(csv_file: str, action_label: int = ACTION_LABEL) -> Optional[str]:
    output_label_file = os.path.splitext(csv_file)[0] + ".txt"
    df = pd.read_csv(csv_file)
    df = calculate_advanced_energy(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8))

    ax1.plot(df['frame'], df['acc_energy_delta'], label='Acc Energy Delta')
    ax1.set_title('ACC Energy Delta')
    ax1.set_ylabel('m/s^2')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(df['frame'], df['gyro_energy'], label='Gyro Energy')
    ax2.set_title('GYRO Energy')
    ax2.set_ylabel('rad/s')
    ax2.set_xlabel('Frame Index')
    ax2.grid(True, linestyle='--', alpha=0.6)

    trigger_frame = []

    def onselect(xmin, xmax):
        start_frame = max(int(xmin), 0)
        end_frame = min(int(xmax), len(df) - 1)

        if end_frame - start_frame < 3:
            return

        sub_df = df.iloc[start_frame:end_frame + 1].copy()
        gyro_min, gyro_max = sub_df['gyro_energy'].min(), sub_df['gyro_energy'].max()
        norm_gyro_energy = (sub_df['gyro_energy'] - gyro_min) / (gyro_max - gyro_min + 1e-6)

        trigger_idx = int(norm_gyro_energy.idxmin())

        ax1.axvline(x=trigger_idx, color='red', linestyle='--', linewidth=1)
        ax2.axvline(x=trigger_idx, color='red', linestyle='--', linewidth=1)

        y_limit_acc = ax1.get_ylim()
        y_pos_acc = y_limit_acc[0] + (y_limit_acc[1] - y_limit_acc[0]) * 0.85

        ax1.text(
            x=trigger_idx,
            y=y_pos_acc,
            s=f'Trigger Frame: {trigger_idx}',
            color='red',
            fontsize=10,
            ha='right',
            va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

        fig.canvas.draw_idle()
        print(f'Trigger Frame: {trigger_idx}')
        trigger_frame.append(trigger_idx)

    plt.tight_layout()

    # Keep hard references; otherwise SpanSelector is GC'd and drag stops working.
    span_selectors = [
        SpanSelector(
            ax1,
            onselect,
            direction='horizontal',
            useblit=False,
            interactive=True,
            drag_from_anywhere=True,
            props=dict(alpha=0.3, facecolor='tab:blue'),
        ),
        SpanSelector(
            ax2,
            onselect,
            direction='horizontal',
            useblit=False,
            interactive=True,
            drag_from_anywhere=True,
            props=dict(alpha=0.3, facecolor='tab:blue'),
        ),
    ]

    fig.suptitle('Drag on either plot to select a tap region', fontsize=11, y=0.98)
    fig.canvas.manager.set_window_title(f'IMU Data Trigger Labeling - {csv_file}')
    plt.show()

    if trigger_frame:
        trigger_frame.sort()

        with open(output_label_file, 'w', encoding='utf-8') as f:
            for frame in trigger_frame:
                f.write(f'{frame}, {action_label}\n')

        print(f'Labeled {len(trigger_frame)} frames saved to {output_label_file}')
        return output_label_file

    print('No frames labeled')
    return None


def main():
    parser = argparse.ArgumentParser(description='Interactive IMU trigger labeling tool')
    parser.add_argument('csv_file', nargs='?', default=None, help='CSV file to label')
    parser.add_argument(
        '--skip_exist', '--skip-existing',
        action='store_true',
        help='Label all CSVs under data/; skip knock_twice files with existing labels',
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Data root when using --skip_exist (default: data)',
    )
    args = parser.parse_args()

    if args.skip_exist:
        import subprocess
        import sys
        from pathlib import Path

        batch_script = Path(__file__).with_name('label_all_imu.py')
        cmd = [sys.executable, str(batch_script), '--skip-existing', '--data-dir', args.data_dir]
        subprocess.run(cmd, check=True)
        return

    csv_file = args.csv_file or DEFAULT_CSV_FILE
    label_imu(csv_file)


if __name__ == '__main__':
    main()
