"""Gerçek saatle çalışan, açıkça otomatik etiketli oturum işlev testi."""

import json
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from kt_project.live_session import LiveResources,LiveSession,parse_time
from kt_project.live_ui import timeline_figure


def main():
    torch.set_num_threads(4)
    resources=LiveResources.load(ROOT)
    session=LiveSession(resources,'Otomatik işlev testi',origin='automated_smoke_test')
    initial_prediction=[]
    for index,correct in enumerate([1,0,0,1,1,1]):
        skill=resources.default_skills[index%len(resources.default_skills)]
        candidates=sorted(q for q,entry in resources.catalog.items() if skill in entry['skills'])
        question=candidates[index//len(resources.default_skills)]
        pending=session.prepare(resources.inverse_questions[question])
        initial_prediction.append(pending['predicted_correct_probability'])
        # Gerçek işlem beklemesi: geçmiş tarih veya uydurma süre eklenmez.
        time.sleep(.15*(1+index%3))
        if index==2:
            session=LiveSession.resume(resources,session.snapshot()['session_id'])
        event=session.answer(pending['question_token'],correct,index%2)
        assert event['history_count_before']==index
        assert parse_time(event['predicted_at'])<=parse_time(event['answer_received_at'])<=parse_time(event['state_updated_at'])
        print(json.dumps({'event':event['sequence'],'answer_received_at':event['answer_received_at'],
            'correct':correct,'response_duration_seconds':event['response_duration_seconds']}),flush=True)
    record=session.close()
    checks=session.verify_frozen_resources()
    assert all(checks.values())
    restored=LiveSession.resume(resources,record['session_id'])
    assert restored.snapshot()==record
    out=ROOT/'outputs/live'
    out.mkdir(exist_ok=True)
    (out/'demo_session.json').write_text(session.path.read_text(encoding='utf-8'),encoding='utf-8')
    figure=timeline_figure(record)
    for suffix in ('png','svg'):
        figure.savefig(out/f'live_timeline.{suffix}',dpi=180)
    plt.close(figure)
    report={'source':'actual local wall-clock calls with explicitly automated response inputs',
        'human_learning_study':False,'record_file':session.path.relative_to(ROOT).as_posix(),
        'events':len(record['events']),'initial_history_count':0,'resume_verified':True,
        'source_event_times_fabricated':False,'timestamped_response_recording_implemented':True,
        'elapsed_session_seconds':(parse_time(record['closed_at'])-parse_time(record['created_at'])).total_seconds(),
        'each_prediction_precedes_answer':True,'all_resource_checks':checks,
        'time_used_as_model_feature':False,'between_event_semantics':'last estimate held; no interpolated learning'}
    report['implementation_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('src/kt_project/live_session.py','src/kt_project/live_ui.py','scripts/run_live_session_demo.py')}
    (ROOT/'outputs/validation/live_session_integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
