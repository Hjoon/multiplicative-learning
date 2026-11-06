"""Sequential sweeps for the paper experiments; no cluster infrastructure."""
import argparse
import csv
from dataclasses import asdict
import itertools
import json
import subprocess
import sys
from pathlib import Path
from erbench.config import ExperimentConfig
from erbench.identity import experiment_id, is_complete
from erbench.train import run_experiment
from multlearning import experiment as control


def records(spec, device='cpu', output='results', data_root='data'):
    seen = set()
    grid = spec.get('grid', {})
    for setup, case, values in itertools.product(spec.get('setups', [{}]), spec.get('cases', [{}]), itertools.product(*[v if isinstance(v,list) else [v] for v in grid.values()])):
        raw = {**spec.get('base',{}), **setup, **case, **dict(zip(grid,values))}
        raw = {k:v for k,v in raw.items() if k not in ('tag','label','description')}
        raw.update(device=device, output_root=str(output), data_root=str(data_root))
        special = raw['optimizer'] in ('tp','dtp') or 'losses' in spec
        cfg = (control.ExperimentConfig if special else ExperimentConfig)(**raw)
        losses = spec['losses'] if 'losses' in spec else ['original' if special else 'mse']
        for loss in losses:
            rid = control.identity(cfg,loss) if special else experiment_id(cfg)
            if rid not in seen:
                seen.add(rid)
                yield dict(run_id=rid,backend='control' if special else 'original',config=asdict(cfg),loss=loss)


def run_isolated(record):
    """Release CUDA, DataLoader workers and file descriptors after each trial."""
    program = (
        "import json,sys; from erbench.config import ExperimentConfig; "
        "from erbench.train import run_experiment; "
        "from multlearning.experiment import run; "
        "r=json.loads(sys.argv[1]); "
        "run(r,r['config']['device'],r['config']['output_root']) "
        "if r['backend']=='control' else run_experiment(ExperimentConfig(**r['config']))"
    )
    subprocess.run([sys.executable, '-c', program, json.dumps(record)], check=True)


def export(output):
    rows=[]
    for rd in sorted((output/'runs').glob('*')):
        if not (rd/'status.json').exists(): continue
        state=json.loads((rd/'status.json').read_text())['status']
        if (rd/'record.json').exists():
            rec=json.loads((rd/'record.json').read_text());cfg=rec['config'];loss=rec['loss']
            if rec.get('smoke') or not (rd/'metrics.json').exists(): continue
            metrics=json.loads((rd/'metrics.json').read_text())
            series=[dict(step=r['step'],split=s,accuracy=r[s+'_accuracy']) for r in metrics for s in ('train','test')]
        elif (rd/'eval.csv').exists():
            cfg=json.loads((rd/'config.json').read_text());loss='mse'
            with (rd/'eval.csv').open() as f: series=list(csv.DictReader(f))
        else: continue
        for r in series:
            rows.append(dict(run_id=rd.name,dataset=cfg['dataset'],optimizer=cfg['optimizer'],seed=cfg['seed'],loss=loss,status=state,step=r['step'],split=r['split'],accuracy=r['accuracy']))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'accuracy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['run_id','dataset','optimizer','seed','loss','status','step','split','accuracy']);w.writeheader();w.writerows(rows)
    print(output/'accuracy.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['sweep','smoke','export'])
    p.add_argument('--config',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.command=='export': export(a.output);return
    if a.command=='smoke':
        for method in ['sgd','momentum','rmsprop','adam','er0','er1','tp','dtp']:
            for loss in (['mse','ce'] if method in ['sgd','momentum','rmsprop','adam'] else ['original']):
                cfg=control.ExperimentConfig(optimizer=method,num_workers=0)
                control.run(dict(config=asdict(cfg),loss=loss),a.device,a.output,smoke=True)
        print('12 synthetic runs completed (software check only)');return
    if a.config is None: p.error('sweep requires --config')
    if a.limit is not None and a.limit<1: p.error('--limit must be positive')
    spec=json.loads(a.config.read_text());jobs=list(records(spec,a.device,a.output,a.data_root))
    if a.limit: jobs=jobs[:a.limit]
    print(f'{len(jobs)} unique runs')
    if a.dry_run:return
    control.atomic(a.output/'manifests'/(a.config.stem+'.json'),jobs)
    failures=[]
    for i,r in enumerate(jobs,1):
        print(f'[{i}/{len(jobs)}] {r["run_id"]}',flush=True)
        if r['backend']=='original' and is_complete(ExperimentConfig(**r['config'])):
            print('[cache hit]',r['run_id'],flush=True)
            continue
        try:
            run_isolated(r)
        except Exception as exc:
            if not spec.get('continue_on_error',False):raise
            failures.append(dict(run_id=r['run_id'],error=str(exc)))
    control.atomic(a.output/'manifests'/(a.config.stem+'-failures.json'),failures)
    export(a.output)
    if failures:print(f'{len(failures)} failed runs; inspect failure records.')


if __name__=='__main__':main()
