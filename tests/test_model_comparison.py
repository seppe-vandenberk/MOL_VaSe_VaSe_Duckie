"""Check log weighting, completed evaluations, and the notebook's data loader."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / 'scripts/model_comparison.ipynb'


def notebook_functions():
    book=json.loads(NOTEBOOK.read_text())
    namespace={}
    for cell in book['cells']:
        source=''.join(cell['source'])
        if cell['cell_type']=='code' and ('from pathlib import Path' in source or 'def model_name' in source):
            exec(compile(source,str(NOTEBOOK),'exec'),namespace)
    return namespace


class ModelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions=notebook_functions()

    def evaluation(self, *, steps=(100,300), complete=True, passed=False):
        return dict(requested_runs=2,passed=passed,runs=[
            dict(steps=count,reward_per_step=value,reward_total=count*value,
                 both_lines_fraction=fraction,completed=complete,failures=int(not passed))
            for count,value,fraction in zip(steps,(1.,3.),(.5,1.))])

    def write_training(self, root, name='model', stop='wall-clock limit', evaluation=None):
        folder=root/name;folder.mkdir()
        (folder/'progress.csv').write_text('steps,reward_per_step,both_lines_fraction\n256,1,0.5\n320,3,1\n')
        (folder/'summary.json').write_text(json.dumps(dict(model=f'/models/{name}.zip',settings={},steps=320,
                                                         stop_reason=stop,evaluation=evaluation)))
        return folder

    def test_partial_final_batch_and_evaluations_are_weighted_by_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.write_training(root,evaluation=self.evaluation())
            table,curves,evaluations,_=self.functions['load_logged_models'](root)
            row=table.iloc[0]
            self.assertAlmostEqual(row.training_reward_per_step,1.4)
            self.assertAlmostEqual(row.training_both_lines_fraction,.6)
            self.assertAlmostEqual(row.evaluation_reward_per_step,2.5)
            self.assertAlmostEqual(row.evaluation_both_lines_fraction,.875)
            self.assertEqual(row.evaluation_steps,400)
            self.assertEqual(curves['model'].weight.tolist(),[256,64])
            self.assertIn('model',self.functions['complete_evaluations'](table,evaluations))
            self.assertFalse(row.evaluation_passed)

    def test_batch_fallback_uses_batch_sizes_and_cumulative_x_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'batch_rewards.csv').write_text('batch,steps,reward_total,both_lines_fraction\n1,256,256,.5\n2,64,192,1\n')
            curve=self.functions['read_training_curve'](root)
            self.assertEqual(curve.steps.tolist(),[256,320])
            self.assertEqual(curve.weight.tolist(),[256,64])
            self.assertEqual(curve.reward_per_step.tolist(),[1.,3.])

    def test_incomplete_evaluation_and_interrupted_training_are_excluded(self):
        for kind in ('missing_run','shortened_run','interrupted','error'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);evaluation=self.evaluation()
                if kind=='missing_run':evaluation['runs'].pop()
                if kind=='shortened_run':evaluation['runs'][1]['completed']=False
                if kind=='error':evaluation['error']='camera timeout'
                self.write_training(root,stop='interrupted' if kind=='interrupted' else 'wall-clock limit',evaluation=evaluation)
                table,_,evaluations,_=self.functions['load_logged_models'](root)
                self.assertEqual(self.functions['complete_evaluations'](table,evaluations),{})

    def test_standalone_report_is_deduplicated_and_finished_report_is_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.write_training(root,evaluation=self.evaluation())
            (root/'later_incomplete.json').write_text(json.dumps(dict(model='models/model.zip',**self.evaluation(complete=False))))
            (root/'standalone.json').write_text(json.dumps(dict(model='models/checkpoint.zip',**self.evaluation(passed=True))))
            table,_,evaluations,_=self.functions['load_logged_models'](root)
            self.assertEqual(set(table.model),{'model','checkpoint'})
            self.assertEqual(set(self.functions['complete_evaluations'](table,evaluations)),{'model','checkpoint'})
            self.assertTrue(evaluations['model']['runs'][0]['completed'])

    def test_empty_logs_and_physical_reports_do_not_invent_training_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'physical.json').write_text(json.dumps(dict(hostname='ducksolo.local',steps=50,completed=True,samples=[])))
            (root/'broken.json').write_text('{')
            table,curves,evaluations,diagnostics=self.functions['load_logged_models'](root)
            self.assertTrue(table.empty)
            self.assertFalse(curves)
            self.assertFalse(evaluations)
            self.assertEqual(len(diagnostics),1)


if __name__=='__main__':
    unittest.main()
