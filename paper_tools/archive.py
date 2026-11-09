"""Archive only the data needed to verify and redraw the adopted paper version."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from .common import PAPER_SWEEPS, entries, portable_config, read_json, run_dir, validate_run_config


ORIGINAL_SUMMARY_FIELDS = (
    'status', 'run_id', 'steps', 'epochs', 'samples_seen', 'input_dim',
    'final_train_loss', 'final_train_accuracy', 'final_test_loss', 'final_test_accuracy',
    'experiment_wall_seconds', 'mean_wall_ms_per_step', 'resident_data_bytes',
    'resident_data_mib', 'max_applied_relative_update', 'wall_timing_scope',
)
STATUS_FIELDS = ('status', 'step', 'steps', 'samples_seen', 'exception_type',
                 'nonfinite_events', 'elapsed_seconds')


def create_archive(results, destination, source_root=None):
    results = Path(results).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to replace an existing archive: {destination}')
    source_root = Path(source_root or Path(__file__).resolve().parents[1])
    members = {}
    seen = {}
    manifest_memberships = {}
    def add_json(name, value):
        members[name] = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    for sweep in PAPER_SWEEPS:
        records = entries(results, sweep)
        portable = []
        for record in records:
            rec = {**record, 'config': portable_config(record['config'])}
            portable.append(rec)
            rid = rec['run_id']
            if rid in seen and seen[rid] != rec:
                raise ValueError(f'Conflicting configurations for shared run {rid}')
            seen[rid] = rec
            manifest_memberships.setdefault(rid, []).append(sweep)
        add_json(f'results/manifests/{sweep}.json', portable)
    diagnostics = {r['run_id'] for r in entries(results, 'fig4_stability')}
    counts = {}
    for rid, record in sorted(seen.items()):
        validate_run_config(results, record)
        rd = run_dir(results, record)
        state = read_json(rd/'status.json')
        if state['status'] not in ('completed', 'failed'):
            raise ValueError(f'Unfinished run cannot be archived: {rid}')
        counts[state['status']] = counts.get(state['status'], 0) + 1
        prefix = f'results/runs/{rid}/'
        compact_state = {k: state[k] for k in STATUS_FIELDS if k in state}
        add_json(prefix+'status.json', compact_state)
        if record['backend'] == 'original':
            config = portable_config(read_json(rd/'config.json'))
            add_json(prefix+'config.json', config)
            for name in ('eval.csv', 'identity.json'):
                path = rd/name
                if path.exists():
                    members[prefix+name] = path.read_bytes()
                elif name == 'eval.csv' and state['status'] == 'completed':
                    raise FileNotFoundError(path)
            if (rd/'summary.json').exists():
                summary = read_json(rd/'summary.json')
                # Nonfinite diagnostic norms are unnecessary for the paper tables.
                import math
                minimal = {k: summary[k] for k in ORIGINAL_SUMMARY_FIELDS if k in summary}
                minimal = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                           for k, v in minimal.items()}
                add_json(prefix+'summary.json', minimal)
            if rid in diagnostics:
                members[prefix+'er_layers.csv'] = (rd/'er_layers.csv').read_bytes()
        else:
            rec = read_json(rd/'record.json')
            minimal = {k: rec[k] for k in ('run_id','loss','source_digest','smoke') if k in rec}
            minimal['config'] = portable_config(rec['config'])
            minimal['environment'] = {k: rec.get('environment', {})[k]
                                      for k in ('python','torch','cuda','device')
                                      if k in rec.get('environment', {})}
            minimal['backend'] = 'control'
            add_json(prefix+'record.json', minimal)
            if (rd/'metrics.json').exists():
                members[prefix+'metrics.json'] = (rd/'metrics.json').read_bytes()
            elif state['status'] == 'completed':
                raise FileNotFoundError(rd/'metrics.json')
        if state['status'] == 'failed':
            failure = read_json(rd/'failure.json') if (rd/'failure.json').exists() else state
            add_json(prefix+'failure.json', {k: failure[k] for k in ('step','exception_type','message') if k in failure})
    for rel in ('validation/paper_reference.json.gz', 'validation/paper_baseline.json',
                'validation/environment.json', 'validation/tool_checks.json', 'source_checksums.json'):
        path = source_root/rel
        if path.exists():
            members[rel] = path.read_bytes()
    source_hashes = {}
    for pattern in ('erbench/*.py','multlearning/*.py','paper_tools/*.py','configs/*.json'):
        for path in sorted(source_root.glob(pattern)):
            source_hashes[str(path.relative_to(source_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('run.py','verify.py','pyproject.toml','LICENSE','CITATION.cff'):
        path=source_root/name
        if path.exists():source_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(schema_version=1, numerical_backend='scalar-individual',
                    purpose='Local checkpoint-free paper regression and replot archive; not a release attachment',
                    paper_sweeps=list(PAPER_SWEEPS), unique_runs=len(seen), status_counts=counts,
                    excluded=['checkpoints','model weights','datasets','paper PDF','figure PDFs/PNGs',
                              'train_steps.csv','full tracebacks','compute_solver exploratory sweep'],
                    source_sha256=source_hashes,
                    files_sha256={name:hashlib.sha256(data).hexdigest() for name,data in sorted(members.items())})
    add_json('ARCHIVE.json', metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name+'.', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        with tarfile.open(temporary, 'w:gz') as archive:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
                archive.addfile(info, io.BytesIO(data))
        # Reopen and verify every stored member before publishing the local archive.
        with tarfile.open(temporary, 'r:gz') as archive:
            for name, digest in metadata['files_sha256'].items():
                if hashlib.sha256(archive.extractfile(name).read()).hexdigest()!=digest:
                    raise RuntimeError(f'Archive checksum mismatch: {name}')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_name(destination.name+'.sha256').write_text(f'{digest}  {destination.name}\n')
    return dict(path=str(destination), bytes=destination.stat().st_size, sha256=digest,
                unique_runs=len(seen), status_counts=counts, files=len(members))
