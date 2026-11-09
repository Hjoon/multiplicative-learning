"""Manifest-only access to paper experiment runs."""
import hashlib
import json
from pathlib import Path

PAPER_SWEEPS = (
    'fig2_optimizer_comparison', 'mse_ce', 'tp_dtp', 'fig3a_batch_main',
    'fig3b_step_mix', 'fig3c_ridge', 'fig4_stability', 'table_stability_ablation',
)
PANEL_SWEEPS = dict(fig2='fig2_optimizer_comparison', fig3a='fig3a_batch_main',
                    fig3b='fig3b_step_mix', fig3c='fig3c_ridge', fig4='fig4_stability')


def read_json(path):
    return json.loads(Path(path).read_text())


def entries(results, name):
    path = Path(results) / 'manifests' / (name + '.json')
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f'Expected a nonempty sweep manifest: {path}')
    seen = set()
    for record in records:
        rid = record['run_id']
        if Path(rid).name != rid or rid in ('.', '..'):
            raise ValueError(f'Invalid run ID: {rid}')
        if rid in seen:
            raise ValueError(f'Duplicate run ID in {path}: {rid}')
        if record['backend'] not in ('original', 'control'):
            raise ValueError(f'Unsupported paper execution core: {record["backend"]}')
        seen.add(rid)
    return records


def run_dir(results, record):
    return Path(results) / 'runs' / record['run_id']


def record_input(path, results, checksums):
    path, results = Path(path), Path(results)
    checksums[str(path.relative_to(results))] = hashlib.sha256(path.read_bytes()).hexdigest()


def portable_config(config):
    cfg = dict(config)
    cfg.update(data_root='./data', output_root='./results', run_name=None)
    return cfg


def validate_run_config(results, record):
    """Reject mislabeled metrics before plotting or preserving a manifest entry."""
    rd=run_dir(results,record)
    if record['backend']=='control':
        stored=read_json(rd/'record.json')
        if stored.get('backend','control')!='control' or stored['loss']!=record['loss']:
            raise ValueError(f'Numerical backend/objective mismatch: {rd}')
        actual=stored['config']
    else:
        actual=read_json(rd/'config.json')
    operational={'device','data_root','output_root','num_workers','run_name','save_model'}
    expected={k:v for k,v in record['config'].items() if k not in operational}
    actual={k:v for k,v in actual.items() if k not in operational}
    if actual!=expected:raise ValueError(f'Manifest/configuration mismatch: {rd}')
