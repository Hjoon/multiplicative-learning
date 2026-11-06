"""Explicit manifests; deterministic epoch-boundary recovery; isolated workers."""
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import torch
from .config import ExperimentConfig
from .models import TanhMLP
from .utils import set_seed, resolve_device, bipolar_targets, sync_device
from .objectives import objective
from .er import er_step


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)); tmp.replace(path)

def source_digest():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob('*.py')):
        h.update(str(p.relative_to(Path(__file__).parent)).encode()); h.update(p.read_bytes())
    return h.hexdigest()

def identity(cfg, loss, smoke=False):
    c = asdict(cfg)
    for k in ('device','output_root','data_root','run_name','num_workers'):
        c.pop(k)
    # ER has no baseline objective dimension.
    loss = 'original' if cfg.optimizer.startswith('er') else loss
    digest = hashlib.sha256(json.dumps([c,loss,smoke,source_digest()],sort_keys=True).encode()).hexdigest()[:16]
    return f'{cfg.dataset}-{cfg.optimizer}-{loss}-s{cfg.seed}-{digest}'

def make_optimizer(model,cfg):
    opts={'sgd':lambda:torch.optim.SGD(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay),
          'momentum':lambda:torch.optim.SGD(model.parameters(),lr=cfg.lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay),
          'rmsprop':lambda:torch.optim.RMSprop(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay),
          'adam':lambda:torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)}
    return opts[cfg.optimizer]()

def evaluate(model,data,cfg,loss="mse"):
    result={}
    model.eval()
    with torch.no_grad():
        for split in ('train','test'):
            x=getattr(data,split+'_x'); y=getattr(data,split+'_y'); correct=0
            for start in range(0,len(y),cfg.eval_batch_size):
                # CE accuracy uses pre-tanh logits to avoid saturation ties.
                pred=(model.logits if loss == "ce" else model)(x[start:start+cfg.eval_batch_size])
                correct+=int((pred.argmax(1)==y[start:start+cfg.eval_batch_size]).sum())
            result[split+'_accuracy']=correct/len(y)
            result[split+'_error']=1-result[split+'_accuracy']
    return result

def run(record, device='cpu', output='results', smoke=False, stop_after_epochs=None):
    from .data import (load_data,make_resident_data,ResidentData,resident_steps_per_epoch,
                       make_shuffle_generator,resident_train_batches)
    cfg=ExperimentConfig(**record['config']); cfg.device=device
    loss=record['loss']; er=cfg.optimizer.startswith('er'); target_method=cfg.optimizer in ('tp','dtp')
    if not er and not target_method and loss not in ('mse','ce'): raise ValueError(loss)
    if smoke:
        cfg.hidden1=8;cfg.hidden2=4;cfg.batch_size=8;cfg.max_steps=6;cfg.eval_batch_size=16
    rid=identity(cfg,loss,smoke); rd=Path(output)/'runs'/rid;rd.mkdir(parents=True,exist_ok=True)
    if (rd/'status.json').exists() and json.loads((rd/'status.json').read_text())['status']=='completed': return rd
    lock=rd/'lock'
    try: lock.mkdir()
    except FileExistsError: raise RuntimeError(f'Run locked: {lock}; remove only after confirming its worker has stopped')
    step=0
    try:
        atomic(rd/'status.json',{'status':'running','pid':os.getpid()})
        set_seed(cfg.seed,cfg.deterministic); dev=resolve_device(device)
        if smoke:
            g=torch.Generator().manual_seed(901)
            def synthetic(n):
                x=torch.rand(n,12,generator=g).to(dev); y=torch.randint(0,10,(n,),generator=g).to(dev)
                return x,y,bipolar_targets(y,10)
            data=ResidentData(*synthetic(24),*synthetic(16),12,10)
        else:
            data=make_resident_data(cfg,load_data(cfg),dev,torch.float32)
        model=TanhMLP(data.input_dim,cfg.hidden1,cfg.hidden2,cfg.num_classes).to(dev)
        opt=None if er or target_method else make_optimizer(model,cfg)
        gen=make_shuffle_generator(cfg.seed+777,dev)
        spe=resident_steps_per_epoch(cfg,data.n_train); total=cfg.max_steps or math.ceil(cfg.epochs*spe)
        rows=[]; elapsed=0.; epoch=0
        checkpoint=rd/'checkpoint.pt'
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location=dev,weights_only=False)
            model.load_state_dict(state['model'])
            if opt: opt.load_state_dict(state['optimizer'])
            gen.set_state(state['generator'].cpu()); step=state['step'];epoch=state['epoch'];rows=state['rows'];elapsed=state['elapsed']
        else:
            rows=[{'step':0,'epoch':0.,**evaluate(model,data,cfg,loss)}]
        atomic(rd/'record.json',{'run_id':rid,'loss':loss,'config':asdict(cfg),'smoke':smoke,
            'source_digest':source_digest(),'environment':{'python':sys.version,'torch':torch.__version__,
            'platform':platform.platform(),'device':str(dev),'cuda':torch.version.cuda,
            'optimizer_defaults':None if opt is None else opt.defaults}})
        sync_device(dev); t0=time.perf_counter(); epochs_this_call=0
        target_schedule = set()
        if target_method:
            target_schedule = {0,*range(1,11),*range(20,101,10),*range(200,1001,100),*range(2000,10001,1000),total}
            if total>10000:
                target_schedule.update(int(round(v/100)*100) for v in torch.logspace(4,torch.log10(torch.tensor(float(total))).item(),12).tolist())
        while step<total:
            model.train()
            for x,y,target in resident_train_batches(data,cfg,gen):
                if step>=total: break
                if target_method:
                    from .targets import step as target_step
                    target_step(model,x,target,cfg,cfg.optimizer)
                elif er: er_step(model,x,y,target,cfg,collect_metrics=False)
                else:
                    opt.zero_grad(set_to_none=True); value=objective(model,x,y,target,loss)
                    if not torch.isfinite(value): raise FloatingPointError('Nonfinite training objective')
                    value.backward(); opt.step()
                step+=1
                if (step in target_schedule) if target_method else (step==total or step<10 or (step<100 and step%10==0) or step%100==0):
                    if not all(torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Nonfinite parameters')
                    rows.append({'step':step,'epoch':step/spe,**evaluate(model,data,cfg,loss)})
                    model.train()
            epoch+=1;epochs_this_call+=1;sync_device(dev)
            state={'model':model.state_dict(),'optimizer':None if opt is None else opt.state_dict(),'generator':gen.get_state(),
                   'step':step,'epoch':epoch,'rows':rows,'elapsed':elapsed+time.perf_counter()-t0}
            tmp=rd/'checkpoint.tmp';torch.save(state,tmp);tmp.replace(checkpoint)
            atomic(rd/'metrics.json',rows)
            if stop_after_epochs and epochs_this_call>=stop_after_epochs: break
        atomic(rd/'status.json',{'status':'completed' if step==total else 'interrupted','step':step,
            'elapsed_seconds':elapsed+time.perf_counter()-t0,'nonfinite_events':0})
        return rd
    except Exception:
        atomic(rd/'failure.json',{'step':step,'traceback':traceback.format_exc()})
        atomic(rd/'status.json',{'status':'failed','step':step});raise
    finally: lock.rmdir()
