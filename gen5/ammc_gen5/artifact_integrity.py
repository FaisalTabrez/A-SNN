"""Fail-closed local provenance receipts (integrity, not digital signatures)."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def bind_protocol(path, value, label='Protocol'):
    """Never attach a new protocol to a pre-existing nonempty directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if identity(json.loads(path.read_text(encoding='utf-8'))) != identity(value):
            raise ValueError(f'{label} changed; choose a new output directory')
    elif any(path.parent.iterdir()):
        raise ValueError(f'Unbound nonempty {label} directory; choose a fresh directory')
    else:
        atomic_json(path, value)


def _files(root, names):
    root = Path(root).resolve()
    result = {}
    for name in names:
        original = root / name
        path = original.resolve()
        if original.is_symlink() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Unsafe or missing artifact: {name}')
        result[name] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
    return result


def seal_files(root, names, receipt, binding):
    atomic_json(Path(root) / receipt, {'schema': 2, 'binding': binding, 'files': _files(root, names)})


def verify_files(root, names, receipt, binding):
    path = Path(root) / receipt
    if not path.is_file():
        raise ValueError(f'Missing integrity receipt: {path}; unauthenticated artifacts cannot resume')
    saved = json.loads(path.read_text(encoding='utf-8'))
    expected = {'schema': 2, 'binding': binding, 'files': _files(root, names)}
    if saved != expected:
        raise ValueError(f'Artifact integrity/binding mismatch: {path}; preserve it and use a fresh directory')


def runtime_identity(torch=None):
    packages = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()
                if d.metadata.get('Name')}
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'packages': dict(sorted(packages.items()))}
    if torch is not None:
        result.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(),
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      cudnn_benchmark=torch.backends.cudnn.benchmark,
                      cudnn_deterministic=torch.backends.cudnn.deterministic,
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_tf32=torch.backends.cudnn.allow_tf32)
        driver = None
        if torch.cuda.is_available():
            try:
                driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                                  '--format=csv,noheader'], text=True, timeout=10).strip()
            except (OSError, subprocess.SubprocessError):
                driver = 'unavailable'
        result['driver'] = driver
    return result
