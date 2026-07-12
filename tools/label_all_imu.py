import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from label_imu import label_imu

DATA_DIR = Path('data')


def is_empty_label_action(filename: str) -> bool:
    name = filename.lower()
    return 'arbitrary' in name or 'knock_once' in name


def create_empty_label_file(csv_path: Path) -> Path:
    label_path = csv_path.with_suffix('.txt')
    label_path.write_text('', encoding='utf-8')
    return label_path


def iter_csv_files(data_dir: Path):
    return sorted(data_dir.rglob('*.csv'))


def main():
    parser = argparse.ArgumentParser(description='Label all IMU CSV files under data/')
    parser.add_argument(
        '--data-dir',
        default=str(DATA_DIR),
        help='Root data directory to scan (default: data)',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip knock_twice files that already have a non-empty label file',
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_files = iter_csv_files(data_dir)

    if not csv_files:
        print(f'No CSV files found under {data_dir}')
        return

    print(f'Found {len(csv_files)} CSV file(s) under {data_dir}')

    for csv_path in csv_files:
        label_path = csv_path.with_suffix('.txt')

        if is_empty_label_action(csv_path.name):
            create_empty_label_file(csv_path)
            print(f'Created empty label file: {label_path}')
            continue

        if 'knock_twice' not in csv_path.name.lower():
            print(f'Skipping unrecognized action type: {csv_path}')
            continue

        if args.skip_existing and label_path.exists() and label_path.read_text(encoding='utf-8').strip():
            print(f'Skipping already labeled file: {csv_path}')
            continue

        print(f'\nLabeling: {csv_path}')
        label_imu(str(csv_path))


if __name__ == '__main__':
    main()
