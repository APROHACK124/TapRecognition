"""Create a separate fast comparison notebook; never overwrite any notebook."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_notebook(original):
    notebook = deepcopy(original)

    def cell(needle):
        matches = [c for c in notebook['cells'] if needle in ''.join(c.get('source', []))]
        if len(matches) != 1:
            raise ValueError(f'Expected one cell with {needle!r}, got {len(matches)}')
        return matches[0]

    def replace(target, before, after):
        source = ''.join(target['source'])
        if source.count(before) != 1:
            raise ValueError(f'Expected one occurrence of {before!r}')
        target['source'] = source.replace(before, after).splitlines(keepends=True)

    notebook['cells'][0]['source'] += ['\n\n## Fast comparison\n',
        'Same frozen operating points, CPU inference calls, reset boundaries, matching rules and table schemas as '
        '`compare.ipynb`. Threshold-independent probability traces and independent fitted-window scores are cached '
        'in memory/on disk. Alarms use exact candidate-based replay. No batching, GPU, reduced precision or retuning.\n',
        '\nSet `COMPARISON_CACHE_DIR=None` for memory-only caching. Cached predictions are invalidated by '
        'input/checkpoint content, inference code, preprocessing, boundaries and backend settings. Current file '
        'hashes and provenance are checked before cached inputs are reused. Timing/cache statistics appear after '
        'each replay. The original notebook and exports are not modified.\n']
    setup = cell('from notebooks_analysis.normal_usage_evaluation import replay_normal_usage')
    replace(setup, 'from notebooks_analysis.normal_usage_evaluation import replay_normal_usage\n',
            'from notebooks_analysis.normal_usage_evaluation import replay_normal_usage\n'
            'from notebooks_analysis.fast_comparison import ComparisonRuntime\n'
            'from functools import partial\n'
            "COMPARISON_CACHE_DIR = ROOT / '.cache/fast_comparison'  # None for memory-only caching.\n"
            'COMPARISON_CPU_THREADS = 2\n'
            'torch.set_num_threads(COMPARISON_CPU_THREADS)\n'
            'comparison_runtime = ComparisonRuntime(COMPARISON_CACHE_DIR)\n'
            'durations_for_export = partial(durations_for_export, cache=comparison_runtime)\n'
            'replay_normal_usage = partial(replay_normal_usage, runtime=comparison_runtime)\n')
    normal_cell = cell('normal_usage_scores, normal_usage_file_scores, normal_usage_alarm_details,')
    replace(normal_cell, 'replay_normal_usage(ROOT, normal_usage_points, NORMAL_USAGE_SOURCE_DIR,\n',
            "comparison_runtime.timed('Full normal-use replay', replay_normal_usage, ROOT, normal_usage_points, NORMAL_USAGE_SOURCE_DIR,\n")
    valid_cell = cell('replay_examples = model_comparison_examples.replay_examples')
    replace(valid_cell, 'replay_examples = model_comparison_examples.replay_examples\n',
            'replay_examples = partial(model_comparison_examples.replay_examples, runtime=comparison_runtime)\n')
    replace(valid_cell, 'torch.set_num_threads(2)\n', 'torch.set_num_threads(COMPARISON_CPU_THREADS)\n')
    replace(valid_cell, 'replay_examples(ROOT, detail_points, source_dir=MINING_SOURCE_DIR)',
            "comparison_runtime.timed('Validation replay', replay_examples, ROOT, detail_points, source_dir=MINING_SOURCE_DIR)")
    train_cell = cell('train_example_recordings, train_example_traces, train_segment_scores,')
    replace(train_cell, "replay_examples(ROOT, detail_points, split='train', source_dir=MINING_SOURCE_DIR)",
            "comparison_runtime.timed('Training replay', replay_examples, ROOT, detail_points, split='train', source_dir=MINING_SOURCE_DIR)")
    for index, target in enumerate(notebook['cells']):
        if target['cell_type'] == 'code':
            target['outputs'] = []
            target['execution_count'] = None
            target.get('metadata', {}).pop('execution', None)
            compile(''.join(target['source']), f'fast-comparison:cell{index}', 'exec')
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'main_notebooks/compare.ipynb')
    parser.add_argument('--output', type=Path, default=ROOT / 'main_notebooks/fast_comparison.ipynb')
    args = parser.parse_args()
    result = build_notebook(json.loads(args.source.read_text()))
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print(f'Created {args.output}; source unchanged.')


if __name__ == '__main__':
    main()
