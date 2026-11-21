"""Gerçek zamanlı kayıt sınırları, kalıcılık ve hedef cevabın gizlenmesi."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0,str(ROOT/'src'))
from kt_project.live_session import LiveResources, LiveSession
from kt_project.patterns import fit_observation_limits
from kt_project.recommendations import training_question_catalog
from kt_project.tracking import ReferenceBank
from kt_project.tracking import snapshot_for
from test_student_tracking import toy_store, toy_tracer


class FixtureClock:
    def __init__(self):
        self.value = datetime(2026,1,1,tzinfo=timezone.utc)
        self.tick = 0.
    def now(self):
        return self.value
    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)
        self.tick += seconds


class LiveSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(2)
        torch.manual_seed(17)
        store = toy_store(self.root/'events.npz')
        tracer = toy_tracer(store)
        tracer.selection = {'method':'kmpt','checkpoint_sha256':'toy','inference_policy':'toy'}
        references = self.root/'refs.json'
        references.write_text(json.dumps({str(s):[{'question':q,'skills':[2,3],'difficulty':.4}
            for q in (3,5,7)] for s in (2,3)}))
        bank = ReferenceBank(references)
        patterns = SimpleNamespace(minimum_history=5,limits=fit_observation_limits(store),
            clusterer=SimpleNamespace(cluster_centers_=np.zeros((1,4))),
            scaler=SimpleNamespace(mean_=np.zeros(4),scale_=np.ones(4)),
            imputer=SimpleNamespace(statistics_=np.zeros(4)),
            predict=lambda feature:{'cluster':0,'description':'test fixture'})
        config=json.loads((ROOT/'configs/tracking.json').read_text())
        self.resources=LiveResources(self.root,tracer,bank,patterns,training_question_catalog(store),
            config,{2:'A',3:'B'},{q:q+100 for q in range(3,11)},[2,3])
        self.clock=FixtureClock()
        self.session=LiveSession(self.resources,_clock=self.clock.now,_monotonic=lambda:self.clock.tick)

    def tearDown(self):
        self.temp.cleanup()

    def respond(self, correct=1):
        pending=self.session.prepare(103)
        self.clock.advance(4)
        return self.session.answer(pending['question_token'],correct)

    def test_empty_start_has_no_fabricated_history_or_observed_percent(self):
        initial=self.session.snapshot()['initial_state']
        self.assertEqual(initial['history_count'],0)
        self.assertIsNone(initial['observed_recent_correct_ratio'])
        self.assertIsNone(initial['behavior'])
        self.assertTrue(all(row['recent_skill_answers']==0 and row['observed_correct_percent'] is None
            for row in initial['skill_scores']))
        self.assertNotIn('label',self.session.tracer.query_batch([
            (snapshot_for(self.session.store,0,0),
             self.resources.catalog[3])]))

    def test_prediction_precedes_answer_and_pending_is_not_history(self):
        pending=self.session.prepare(103)
        self.assertEqual(self.session.store.spans[0],(0,0))
        self.clock.advance(4)
        event=self.session.answer(pending['question_token'],0,2)
        self.assertLess(event['predicted_at'],event['answer_received_at'])
        self.assertEqual(event['history_count_before'],0)
        self.assertEqual(event['after']['history_count'],1)
        self.assertEqual(event['response_duration_seconds'],4)
        self.assertEqual(event['hint_count'],2)
        self.assertEqual(event['after']['skill_scores'][0]['observed_correct_percent'],0)

    def test_duplicate_submission_is_idempotent_and_conflicting_result_rejected(self):
        event=self.respond()
        self.assertEqual(self.session.answer(event['question_token'],1),event)
        self.assertEqual(len(self.session.snapshot()['events']),1)
        with self.assertRaises(ValueError):
            self.session.answer(event['question_token'],0)

    def test_unsupported_ids_outcomes_and_tokens_do_not_change_history(self):
        for value in (99999,'103',True):
            with self.assertRaises(ValueError):
                self.session.prepare(value)
        pending=self.session.prepare(103)
        for value in (2,'1',True):
            with self.assertRaises(ValueError):
                self.session.answer(pending['question_token'],value)
        with self.assertRaises(ValueError):
            self.session.answer('wrong-token',1)
        self.assertEqual(self.session.snapshot()['events'],[])

    def test_resume_preserves_pending_and_uses_real_wall_time_for_interruption(self):
        pending=self.session.prepare(103)
        self.clock.advance(11)
        resumed=LiveSession.resume(self.resources,self.session.snapshot()['session_id'],_clock=self.clock.now)
        event=resumed.answer(pending['question_token'],1)
        self.assertEqual(event['duration_clock'],'UTC_after_resume')
        self.assertEqual(event['response_duration_seconds'],11)
        closed=resumed.close()
        again=LiveSession.resume(self.resources,closed['session_id'],_clock=self.clock.now)
        self.assertEqual(again.snapshot(),closed)
        with self.assertRaises(ValueError):
            again.prepare(103)

    def test_clock_regression_and_second_pending_question_are_rejected(self):
        pending=self.session.prepare(103)
        with self.assertRaises(ValueError):
            self.session.prepare(104)
        self.clock.advance(-1)
        with self.assertRaises(ValueError):
            self.session.answer(pending['question_token'],1)
        self.assertEqual(self.session.snapshot()['events'],[])

    def test_disk_failure_rolls_back_answer_and_stale_view_cannot_overwrite(self):
        pending=self.session.prepare(103)
        self.clock.advance(1)
        with patch('kt_project.live_session.os.replace',side_effect=OSError('disk fixture')):
            with self.assertRaises(OSError):
                self.session.answer(pending['question_token'],1)
        self.assertEqual(self.session.store.spans[0],(0,0))
        resumed=LiveSession.resume(self.resources,self.session.snapshot()['session_id'],_clock=self.clock.now)
        resumed.answer(pending['question_token'],1)
        with self.assertRaises(ValueError):
            self.session.answer(pending['question_token'],1)

    def test_persisted_tampering_is_detected_and_external_snapshot_cannot_mutate(self):
        self.respond()
        view=self.session.snapshot()
        view['events'][0]['correct']=0
        self.assertEqual(self.session.snapshot()['events'][0]['correct'],1)
        saved=json.loads(self.session.path.read_text(encoding='utf-8'))
        saved['events'][0]['correct']=0
        self.session.path.write_text(json.dumps(saved),encoding='utf-8')
        with self.assertRaises(ValueError):
            LiveSession.resume(self.resources,self.session.snapshot()['session_id'])
        with self.assertRaises(ValueError):
            LiveSession.resume(self.resources,'../escape')

    def test_behavior_starts_after_five_answers_and_resources_stay_frozen(self):
        for count in range(5):
            event=self.respond(count%2)
            self.assertEqual(event['after']['behavior'] is None,count<4)
        self.assertTrue(all(self.session.verify_frozen_resources().values()))
        with torch.no_grad():
            next(self.session.tracer.model.parameters()).add_(1)
        self.assertFalse(self.session.verify_frozen_resources()['model_weights_unchanged'])

    def test_actual_widget_callbacks_create_prepare_record_close_and_resume(self):
        from kt_project.live_ui import create_live_controls
        import matplotlib.pyplot as plt
        with patch('IPython.display.display') as show,patch('IPython.display.clear_output'):
            panel=create_live_controls(self.resources,origin='automated_smoke_test',directory=self.root/'ui')
            controls=panel._live_controls
            rendered_control_states=[]
            show.side_effect=lambda *_:rendered_control_states.append(
                [controls[name].disabled for name in ('start','prepare','correct','wrong','close')])
            self.assertTrue(controls['prepare'].disabled)
            controls['start'].click()
            session=controls['current']['session']
            self.assertEqual(session.snapshot()['response_origin'],'automated_smoke_test')
            controls['prepare'].click()
            self.assertTrue(controls['prepare'].disabled,controls['message'].value)
            self.assertFalse(controls['correct'].disabled)
            controls['correct'].click()
            self.assertEqual(len(session.snapshot()['events']),1)
            controls['close'].click()
            self.assertEqual(session.snapshot()['status'],'closed')
            controls['resume'].click()
            self.assertEqual(controls['current']['session'].snapshot()['session_id'],session.snapshot()['session_id'])
            self.assertTrue(controls['prepare'].disabled)
            self.assertTrue(rendered_control_states)
            self.assertTrue(all(all(states) for states in rendered_control_states),
                            'Grafik ve tablolar yenilenirken yeni giriş açılmamalı.')
            panel.close()
        plt.close('all')

    def test_timeline_uses_actual_datetimes_and_step_holds_between_events(self):
        from kt_project.live_ui import timeline_figure
        import matplotlib.pyplot as plt
        self.respond()
        self.clock.advance(7)
        record=self.session.close()
        fig=timeline_figure(record)
        line=fig.axes[0].lines[0]
        self.assertEqual(line.get_drawstyle(),'steps-post')
        self.assertEqual(line.get_xdata()[0],datetime.fromisoformat(record['initial_state_recorded_at']))
        self.assertEqual(line.get_xdata()[-1],datetime.fromisoformat(record['closed_at']))
        self.assertEqual(line.get_ydata()[-1],line.get_ydata()[-2])
        plt.close(fig)


if __name__=='__main__':
    unittest.main()
