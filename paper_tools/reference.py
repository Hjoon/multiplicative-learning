"""Explicit paper baseline capture, separate from verification."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
from .common import PAPER_SWEEPS, entries, read_json, run_dir


GROUPS=dict(fig2_optimizer_comparison='fig2',mse_ce='mse_ce',tp_dtp='tp_dtp',
            fig3a_batch_main='fig3a',fig3b_step_mix='fig3b',fig3c_ridge='fig3c',
            fig4_stability='fig4',table_stability_ablation='table_stability_ablation')


def capture_reference(results, destination):
    results,destination=Path(results),Path(destination)
    if destination.exists():raise FileExistsError('Refusing to overwrite the frozen paper baseline')
    records={}
    for sweep in PAPER_SWEEPS:
        for entry in entries(results,sweep):
            rid=entry['run_id']
            if rid in records:
                records[rid]['sweeps'].append(sweep);continue
            rd=run_dir(results,entry);state=read_json(rd/'status.json')
            if state['status'] not in ('completed','failed'):raise ValueError(f'Unfinished reference trial: {rid}')
            cfg={k:v for k,v in entry['config'].items() if k not in ('device','data_root','output_root','num_workers','run_name')}
            if entry['backend']=='original':
                path=rd/'eval.csv'
                if path.exists():
                    with path.open() as stream:series=[r for r in csv.DictReader(stream) if r['split']=='test']
                else:series=[]
            else:
                path=rd/'metrics.json'
                series=[dict(step=r['step'],accuracy=r['test_accuracy']) for r in read_json(path)] if path.exists() else []
            values=[float(r['accuracy']) for r in series]
            if state['status']=='completed' and (not values or int(float(series[-1]['step']))!=cfg['max_steps']):
                raise ValueError(f'Missing endpoint for reference trial {rid}')
            record=dict(group=GROUPS[sweep],sweeps=[sweep],backend=entry['backend'],
                        reference_run_id=rid,config=cfg,loss=entry['loss'],status=state['status'],
                        final_step=int(float(series[-1]['step'])) if series else None,
                        final_accuracy=values[-1] if values else None,
                        best_accuracy=max(values) if values else None)
            if state.get('exception_type'):record['failure_type']=state['exception_type']
            records[rid]=record
    data=dict(metadata=dict(name='paper-scalar-2026-11-09',schema_version=2,numerical_backend='scalar-individual',
                            adopted_on='2026-11-09',paper_sweeps=list(PAPER_SWEEPS),
                            meaning='Frozen regression baseline from the full scalar campaign used in the manuscript; capture is not an independent reproduction test.'),
              records=list(records.values()))
    destination.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
    destination.write_bytes(gzip.compress(payload,mtime=0) if destination.suffix=='.gz' else payload)
    return dict(trials=len(records),sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
