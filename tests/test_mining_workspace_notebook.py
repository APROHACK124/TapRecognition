"""New-workspace isolation and batch source-type filtering in the mining notebook."""

import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class MiningWorkspaceNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / 'main_notebooks/mine_hard_negatives.ipynb').read_text())
        cls.code = [''.join(cell['source']) for cell in cls.notebook['cells']
                    if cell['cell_type'] == 'code']
        batch = next(source for source in cls.code if 'BATCH_SOURCE_TYPES = ' in source)
        cls.selection_code = batch.split('source_type_aliases = ', 1)[1].split(
            'batch_candidates, batch_file_stats, batch_stats = ', 1)[0]
        cls.selection_code = 'source_type_aliases = ' + cls.selection_code

    def select(self, types, files=None):
        inventory = pd.DataFrame([
            dict(file='new_negative.csv', recording_type='hard_negative', annotated_events=0),
            dict(file='new_normal.csv', recording_type='normal_usage', annotated_events=0),
            dict(file='positive.csv', recording_type='knock_twice', annotated_events=20),
            dict(file='annotated_normal.csv', recording_type='normal_usage', annotated_events=1),
            dict(file='old__crop_f000000_f000300.csv', recording_type='hard_negative', annotated_events=0),
        ])
        env = dict(inventory=inventory, BATCH_SOURCE_TYPES=types,
                   BATCH_SELECT_ALL=files is None, BATCH_FILES=files or [],
                   MODELS={'fixture': dict(left_threshold=.5, right_threshold=.5)},
                   BATCH_MODELS=['fixture'])
        exec(compile(self.selection_code, 'batch-selection', 'exec'), env)
        return env['batch_files']

    def test_types_can_be_selected_independently(self):
        self.assertEqual(self.select(['negative_mining']), ['new_negative.csv'])
        self.assertEqual(self.select(['normal_usage']), ['new_normal.csv'])
        self.assertEqual(self.select(['negative_mining', 'normal_usage']),
                         ['new_negative.csv', 'new_normal.csv'])

    def test_explicit_files_obey_type_filter(self):
        self.assertEqual(self.select(['normal_usage'], ['new_normal.csv']), ['new_normal.csv'])
        with self.assertRaises(AssertionError):
            self.select(['normal_usage'], ['new_negative.csv'])

    def test_invalid_types_are_rejected(self):
        for types in ([], ['knock_twice']):
            with self.assertRaises(AssertionError):
                self.select(types)

    def test_workspace_and_model_defaults(self):
        setup = next(source for source in self.code if 'MINING_WORKSPACE = ' in source)
        settings = 'MINING_WORKSPACE = ' + setup.split('MINING_WORKSPACE = ', 1)[1]
        env = dict(ROOT=ROOT)
        exec(compile(settings, 'workspace-settings', 'exec'), env)
        workspace = ROOT / 'hard_negative_mining_22_10_26'
        self.assertEqual(env['INPUT_DIR'], workspace / 'source_recordings')
        self.assertEqual(env['OUTPUT_DIR'], workspace / 'crops')
        self.assertEqual(env['MODELS'], {})

    def test_code_compiles(self):
        for index, source in enumerate(self.code):
            compile(source, f'mining-cell-{index}', 'exec')


if __name__ == '__main__':
    unittest.main()
