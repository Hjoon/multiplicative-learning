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


def test_verifier_keeps_execution_cores_separate(tmp_path):
    from verify import compare
    config=dict(dataset='mnist',optimizer='adam',seed=0,max_steps=5)
    for name,backend,value in [('scalar','control',.9),('original','original',.7)]:
        rd=tmp_path/'runs'/name;rd.mkdir(parents=True)
        (rd/'status.json').write_text(json.dumps({'status':'completed'}))
        (rd/'record.json').write_text(json.dumps(dict(config=config,loss='ce',backend=backend)))
        (rd/'metrics.json').write_text(json.dumps([dict(step=5,test_accuracy=value)]))
    ref=dict(group='test',backend='control',reference_run_id='scalar',config=config,loss='ce',status='completed',final_step=5,final_accuracy=.9,best_accuracy=.9)
    report=compare([ref],tmp_path,tolerance_pp=0)
    assert report['passed'] and report['counts']=={'within_tolerance':1}
    assert report['comparisons'][0]['run_id']=='scalar'


def test_verifier_does_not_accept_a_different_failure(tmp_path):
    from verify import compare
    rd=tmp_path/'runs'/'trial';rd.mkdir(parents=True)
    cfg=dict(dataset='mnist',optimizer='er1',seed=0)
    (rd/'config.json').write_text(json.dumps(cfg))
    (rd/'status.json').write_text(json.dumps(dict(status='failed',exception_type='OSError')))
    ref=dict(group='test',backend='original',config=cfg,loss='mse',status='failed',failure_type='_LinAlgError')
    assert compare([ref],tmp_path)['counts']=={'failure_type_mismatch':1}


def test_verifier_rejects_empty_or_ambiguous_reference(tmp_path):
    import pytest
    from verify import compare
    with pytest.raises(ValueError):compare([],tmp_path)
    cfg=dict(dataset='mnist',optimizer='adam',seed=0)
    for name in ['first','second']:
        rd=tmp_path/'runs'/name;rd.mkdir(parents=True)
        (rd/'record.json').write_text(json.dumps(dict(config=cfg,loss='mse')))
        (rd/'status.json').write_text(json.dumps(dict(status='completed')))
        (rd/'metrics.json').write_text(json.dumps([dict(step=5,test_accuracy=.9)]))
    ref=dict(group='test',backend='control',reference_run_id='absent',config=cfg,loss='mse',status='completed',final_step=5,final_accuracy=.9,best_accuracy=.9)
    assert compare([ref],tmp_path)['counts']=={'ambiguous_match':1}


def test_minimal_archive_excludes_models_and_paper_artifacts(tmp_path):
    import hashlib
    import tarfile
    from paper_tools.archive import create_archive
    from paper_tools.common import PAPER_SWEEPS
    from verify import compare
    results=tmp_path/'results';rd=results/'runs'/'trial';rd.mkdir(parents=True)
    cfg=dict(dataset='mnist',optimizer='adam',seed=0,max_steps=5,device='cpu',data_root='/private/data',output_root='/private/results')
    rec=dict(run_id='trial',backend='control',loss='ce',config=cfg)
    (results/'manifests').mkdir()
    for name in PAPER_SWEEPS:(results/'manifests'/(name+'.json')).write_text(json.dumps([rec]))
    (rd/'status.json').write_text(json.dumps(dict(status='completed',pid=99,host='private-host')))
    (rd/'record.json').write_text(json.dumps(dict(run_id='trial',loss='ce',config=cfg,environment=dict(torch='2.5.1',host='private-host'))))
    (rd/'metrics.json').write_text(json.dumps([dict(step=5,test_accuracy=.9)]))
    for filename in ['checkpoint.pt','model.pt','paper.pdf','figure.png','train_steps.csv']:(rd/filename).write_bytes(b'not needed')
    empty=tmp_path/'source';empty.mkdir()
    target=tmp_path/'minimal.tar.gz';meta=create_archive(results,target,source_root=empty)
    assert meta['unique_runs']==1
    unpack=tmp_path/'unpacked';unpack.mkdir()
    with tarfile.open(target) as archive:
        names=archive.getnames()
        assert not any(n.endswith(('.pt','.pdf','.png','train_steps.csv')) for n in names)
        manifest=json.load(archive.extractfile('ARCHIVE.json'))
        for name,digest in manifest['files_sha256'].items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
        for member in archive.getmembers():
            assert not member.name.startswith('/') and '..' not in Path(member.name).parts
            if member.isfile():
                path=unpack/member.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.extractfile(member).read())
    ref=dict(group='test',backend='control',reference_run_id='trial',config=cfg,loss='ce',status='completed',final_step=5,final_accuracy=.9,best_accuracy=.9)
    assert compare([ref],unpack/'results',tolerance_pp=0)['passed']
    assert 'private-host' not in (unpack/'results/runs/trial/record.json').read_text()
    assert '/private/' not in (unpack/'results/runs/trial/record.json').read_text()


def test_paper_reference_has_expected_trials_and_execution_cores():
    from verify import read_reference
    current=read_reference(ROOT/'validation/paper_reference.json.gz')
    assert current['metadata']['numerical_backend']=='scalar-individual'
    assert len(current['records'])==1861
    assert sum(r['status']=='failed' for r in current['records'])==338
    assert all(r['backend'] in ('original','control') and r['reference_run_id'] for r in current['records'])


def test_plotting_rejects_a_mislabeled_seed(tmp_path):
    import pytest
    from paper_tools.common import validate_run_config
    rd=tmp_path/'runs'/'trial';rd.mkdir(parents=True)
    actual=dict(dataset='mnist',optimizer='sgd',seed=1)
    (rd/'config.json').write_text(json.dumps(actual))
    record=dict(run_id='trial',backend='original',loss='mse',config={**actual,'seed':0})
    with pytest.raises(ValueError,match='configuration mismatch'):validate_run_config(tmp_path,record)
