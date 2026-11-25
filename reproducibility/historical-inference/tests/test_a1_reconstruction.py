"""Audit produced reconstruction artifacts against the pre-inference freeze."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from reconstruct_a1 import metrics, OUT, chip, rows

class ReconstructionTests(unittest.TestCase):
    def test_coverage_and_frozen_selection(self):
        plan=json.loads((OUT/'selection_frozen.json').read_text())
        result=json.loads((OUT/'results.json').read_text())
        self.assertEqual(len(result['coverage']),10)
        self.assertEqual({(x['event'],x['seed'],x['sha256']) for x in plan['selected']},{(x['event'],x['seed'],x['checkpoint_sha256']) for x in result['individual']})
        usa=next(x for x in plan['selected'] if x['event']=='USA')
        self.assertTrue(usa['path'].endswith('/best-v1.ckpt'))
        self.assertEqual(usa['run_status'],'FINISHED')
        self.assertTrue(usa['initialization_evidence'].startswith('U:'))
    def test_counts_membership_masks_and_ignored_chips(self):
        plan=json.loads((OUT/'selection_frozen.json').read_text())
        for event,manifest in plan['manifests'].items():
            with (OUT/'counts'/('generalist_'+event+'.csv')).open() as f:g=list(csv.DictReader(f))
            self.assertEqual({x['chip_id'] for x in g},{chip(x[0]) for x in rows(ROOT/manifest['path'])})
            for x in g:
                self.assertEqual(sum(int(x[k]) for k in ('tp','fp','fn','tn')),int(x['valid_pixels']))
                if not int(x['valid_pixels']):
                    self.assertEqual(x['iou'],'');self.assertEqual(x['f1'],'');self.assertEqual(x['status'],'all_ignored')
            for entry in [x for x in plan['selected'] if x['event']==event]:
                with (OUT/'counts'/(event+'_s'+str(entry['seed'])+'.csv')).open() as f:loco=list(csv.DictReader(f))
                self.assertEqual([(x['chip_id'],x['mask_sha256']) for x in g],[(x['chip_id'],x['mask_sha256']) for x in loco])
    def test_no_scores_for_all_ignored_and_pooled_counts(self):
        empty=dict(tp=0,fp=0,fn=0,tn=0,valid_pixels=0)
        self.assertIsNone(metrics([empty])['iou'])
        ordinary=dict(tp=2,fp=1,fn=1,tn=5,valid_pixels=9)
        self.assertEqual(metrics([empty,ordinary])['iou'],.5)
        self.assertAlmostEqual(metrics([ordinary])['f1'],2/3)

if __name__=='__main__':unittest.main()
