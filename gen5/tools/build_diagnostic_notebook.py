"""Build a no-training notebook for original Evidence-1 compiler discrepancies."""
from pathlib import Path
import textwrap
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[2]


def build():
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text): cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))
    md('''
    # Frozen Evidence-1 compiler diagnosis — no training
    This uses the exact original 73-source package and the validation checkpoints
    SHD310/SSC311. It does not evaluate test data, rerun the 80 fits, adjust a
    tolerance, or change the registered result. Backend comparisons are diagnostics,
    not permission to select a winner from test results. Use your existing raw-data
    folder's **original encoded caches**, not new v2 caches. No datasets are downloaded.
    ''')
    code('''
    # 1. Configuration. Supply the full release commit SHA printed by Codex.
    RELEASE_REVISION = ''
    BUNDLE = '/content/drive/MyDrive/A-SNN-research-restart/research_restart_results.zip'
    DATA_ROOT = '/content/drive/MyDrive/A-SNN-research-restart/data'
    OUTPUT = '/content/drive/MyDrive/A-SNN-research-restart/fidelity_diagnostic_01'
    USE_DRIVE = True
    PREFLIGHT_ONLY = True  # First check integrity; False explicitly executes CUDA diagnosis.
    ''')
    code('''
    # 2. Attach existing files; preserve the installed CUDA Torch build.
    import importlib.util, pathlib, re, subprocess, sys
    try: IN_COLAB = importlib.util.find_spec('google.colab') is not None
    except (ImportError, ModuleNotFoundError, ValueError): IN_COLAB = False
    if IN_COLAB and USE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
    import torch
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
    if not PREFLIGHT_ONLY: assert torch.cuda.is_available(), 'CUDA runtime required; no CPU fallback'
    assert pathlib.Path(BUNDLE).is_file(), 'Use your original result ZIP; do not retrain.'
    ''')
    code('''
    # 3. Immutable release checkout; no resets or overwrites.
    assert re.fullmatch('[0-9a-f]{40}', RELEASE_REVISION), 'Set the full release commit SHA.'
    REPO = pathlib.Path('/content/A-SNN-fidelity') if IN_COLAB else pathlib.Path.cwd() / 'A-SNN-fidelity'
    def command(args): subprocess.run(list(map(str, args)), check=True)
    if not REPO.exists():
        command(['git', 'clone', '--no-checkout', 'https://github.com/FaisalTabrez/A-SNN.git', REPO])
        command(['git', '-C', REPO, 'checkout', '--detach', RELEASE_REVISION])
    actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    assert actual == RELEASE_REVISION and not dirty, 'Choose a fresh checkout; this notebook does not reset files.'
    ''')
    code('''
    # 4. Verify all original bundle and source hashes, optionally execute validation only.
    args = [sys.executable, REPO / 'gen5/tools/evidence1_fidelity_diagnostic.py',
            '--bundle', BUNDLE, '--data-root', DATA_ROOT, '--output-dir', OUTPUT]
    if PREFLIGHT_ONLY: args.append('--preflight-only')
    command(args)
    # For the real diagnosis, select a CUDA runtime, set PREFLIGHT_ONLY=False and
    # choose a NEW empty OUTPUT. The completed preflight folder is preserved.
    ''')
    code('''
    # 5. Export only diagnostic reports, not original checkpoints/data.
    import json, zipfile
    folder = pathlib.Path(OUTPUT)
    result = folder / 'diagnostic_results.json'
    if result.exists():
        report = json.loads(result.read_text())
        assert report['test_evaluations'] == 0 and report['trained'] is False
        print(json.dumps(report, indent=2))
    else:
        print((folder / 'preflight.json').read_text())
    destination = folder / 'fidelity_diagnostic_reports.zip'
    assert not destination.exists(), 'Report export exists; preserve it.'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename in ['preflight.json', 'diagnostic_results.json']:
            if (folder / filename).exists(): archive.write(folder / filename, filename)
    if IN_COLAB:
        from google.colab import files
        files.download(str(destination))
    else: print(destination.resolve())
    ''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    for i, cell in enumerate(notebook.cells): cell.id = f'fidelity-{i:02d}'
    nbf.validate(notebook)
    destination = ROOT / 'gen5/notebooks/evidence1_fidelity_diagnostic.ipynb'
    nbf.write(notebook, destination)
    print(destination)


if __name__ == '__main__': build()
