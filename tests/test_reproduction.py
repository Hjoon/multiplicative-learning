import copy
import json
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from run import records, export
from erbench import train
from erbench.config import ExperimentConfig
from erbench.data import DataBundle
from multlearning.models import TanhMLP
from multlearning.objectives import objective
from multlearning.experiment import evaluate
from multlearning.data import ResidentData
from multlearning.utils import bipolar_targets

ROOT=Path(__file__).resolve().parents[1]


def test_all_configs_expand():
    for path in (ROOT/'configs').glob('*.json'):
        jobs=list(records(json.loads(path.read_text())))
        assert jobs and len({r['run_id'] for r in jobs})==len(jobs)
    assert len(list(records(json.loads((ROOT/'configs/mse_ce.json').read_text()))))==240
    assert len(list(records(json.loads((ROOT/'configs/tp_dtp.json').read_text()))))==60
    assert len(list(records(json.loads((ROOT/'configs/fig2_optimizer_comparison.json').read_text()))))==180


def test_ce_uses_logits_under_saturation():
    m=TanhMLP(2,2,2,10)
    with torch.no_grad():
        for p in m.parameters():p.zero_()
        m.layers[-1].bias.fill_(20);m.layers[-1].bias[1]=21
    x=torch.ones(3,2);y=torch.ones(3,dtype=torch.long);t=bipolar_targets(y,10)
    data=ResidentData(x,y,t,x,y,t,2,10)
    cfg=ExperimentConfig(eval_batch_size=2)
    assert evaluate(m,data,cfg,'ce')['test_accuracy']==1
    assert evaluate(m,data,cfg,'mse')['test_accuracy']==0
    assert torch.equal(objective(m,x,y,t,'ce'),torch.nn.functional.cross_entropy(m.logits(x),y))


def test_original_runner_and_export(tmp_path,monkeypatch):
    g=torch.Generator().manual_seed(7)
    ds=TensorDataset(torch.rand(16,4,generator=g),torch.randint(0,10,(16,),generator=g))
    monkeypatch.setattr(train,'load_data',lambda cfg:DataBundle(ds,ds,4,10))
    for method in ['er0','er1','sgd','momentum','rmsprop','adam','pcg32_10']:
        cfg=ExperimentConfig(optimizer=method,device='cpu',num_workers=0,batch_size=8,hidden1=4,hidden2=3,max_steps=2,diag_every=0,output_root=str(tmp_path))
        rd,_,_=train.run_experiment(cfg)
        assert json.loads((rd/'status.json').read_text())['status']=='completed'
        assert train.run_experiment(cfg)[0]==rd
    export(tmp_path)
    assert len((tmp_path/'accuracy.csv').read_text().splitlines())>14


def test_all_control_methods(tmp_path):
    from multlearning import experiment as e
    for method in ['sgd','momentum','rmsprop','adam','er0','er1','tp','dtp']:
        for loss in (['mse','ce'] if method in ['sgd','momentum','rmsprop','adam'] else ['original']):
            cfg=e.ExperimentConfig(optimizer=method,num_workers=0)
            record=dict(config=e.asdict(cfg),loss=loss)
            rd=e.run(record,'cpu',tmp_path,smoke=True)
            assert json.loads((rd/'status.json').read_text())['step']==6
            assert e.run(record,'cpu',tmp_path,smoke=True)==rd


def test_verifier_requires_complete_runs(tmp_path):
    from verify import compare
    ref=dict(group='test',config=dict(dataset='mnist',optimizer='adam',seed=0),loss='mse',status='completed',final_step=5,final_accuracy=.9,best_accuracy=.9)
    assert compare([ref],tmp_path)['counts']=={'missing':1}
    assert not compare([ref],tmp_path)['passed']
    rd=tmp_path/'runs'/'example';rd.mkdir(parents=True)
    (rd/'status.json').write_text(json.dumps({'status':'completed'}))
    (rd/'record.json').write_text(json.dumps({'config':ref['config'],'loss':'mse'}))
    (rd/'metrics.json').write_text(json.dumps([{'step':4,'test_accuracy':.9}]))
    assert compare([ref],tmp_path)['counts']=={'budget_mismatch':1}
    (rd/'metrics.json').write_text(json.dumps([{'step':5,'test_accuracy':.9}]))
    assert compare([ref],tmp_path)['passed']
    (rd/'metrics.json').write_text(json.dumps([{'step':5,'test_accuracy':.8}]))
    assert compare([ref],tmp_path)['counts']=={'numerical_mismatch':1}


def test_isolated_trial_reuses_completed_result(tmp_path):
    from run import run_isolated
    from multlearning import experiment as e
    cfg=e.ExperimentConfig(optimizer='adam',output_root=str(tmp_path),device='cpu')
    record=dict(config=e.asdict(cfg),loss='mse',backend='control')
    rid=e.identity(cfg,'mse');rd=tmp_path/'runs'/rid;rd.mkdir(parents=True)
    state=rd/'status.json';state.write_text(json.dumps({'status':'completed'}))
    before=state.read_bytes()
    run_isolated(record)
    assert state.read_bytes()==before
    assert not (rd/'lock').exists()
