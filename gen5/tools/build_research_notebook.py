"""Build the portable research-restart notebook with embedded runner snapshots."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = ["gen5/ammc_gen5/evidence1_audio.py", "gen5/ammc_gen5/artifact_integrity.py", "gen5/examples/evidence1_audio.py",
             "gen5/tools/fits_validation_reference.py", "gen5/tools/fly_reference.py"]


def build():
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))
    md("""
    # A-SNN research restart — Evidence-1 first
    Prepared 2026-10-13. Open in **Google Colab** or Jupyter and run top to bottom.
    Default is an inexpensive **synthetic smoke test**, not SHD/SSC evidence.
    For the registered study, select a **GPU** runtime, set `PROFILE='confirmation'`,
    `CONFIRM_FULL_RUN=True`, and preferably `USE_DRIVE=True` below, then Run all.
    Download, checksum verification, training, resumption, plots and ZIP export are included.

    The confirmation is 80 fits (2 datasets × 10 seeds × 4 models), 15 epochs each,
    plus frozen-checkpoint state ablations and compiler checks. It can outlast a
    Colab session. Epoch checkpoints persist if Drive is enabled; otherwise download
    the output folder before disconnecting. No TPU path or silent CPU confirmation.
    Allow at least 20 GB free disk and preferably 16+ GB host RAM for official audio;
    these are conservative planning allowances, not measured resource requirements.

    FiTS, timing-only data and whole-fly reproduction are separately opt-in.
    They do not alter the registered Evidence-1 protocol or reopen closed mechanisms.
    Whole-fly simulation is CPU/Brian2 work and can be slow even in its smoke profile.
    """)
    code("""
    # 1. Configuration — edit here before Run all.
    PROFILE = 'smoke'                  # 'smoke' or 'confirmation'
    CONFIRM_FULL_RUN = False           # True only for the full GPU study
    USE_DRIVE = False                  # Colab persistent data/checkpoints
    INSTALL_DEPENDENCIES = True
    DOWNLOAD_AUDIO = False             # confirmation automatically enables this
    DOWNLOAD_REFERENCES = False        # all pinned upstream code + open arXiv PDFs
    DOWNLOAD_TIMING_DATA = False       # Zenodo dataset, checksum-verified
    RUN_FITS_REFERENCE = False         # 100 epochs per dataset, validation-only selection
    RUN_FLY_REFERENCE = False          # whole-brain reference diagnostic, not our next mechanism
    FLY_PROFILE = 'smoke'              # 'reference' = author-like 30 × 1-second trials
    DOWNLOAD_RESULT_ZIP = False        # automatic browser download in Colab
    TIMING_DOWNLOAD_LIMIT_GB = 5.0      # fail before download if exceeded
    TIMING_FILE_NAMES = None           # None = all Zenodo files; or exact listed names
    LOCAL_WORK_DIR = None              # optional persistent path outside repository
    """)
    code("""
    # 2. Environment and dependencies. Keep an existing GPU PyTorch installation.
    import sys, subprocess, importlib.util, os, json, hashlib, base64
    from pathlib import Path
    def module_available(name):
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    IN_COLAB = module_available('google.colab')
    def command(args, cwd=None):
        print('Running:', ' '.join(map(str, args)))
        subprocess.run(list(map(str, args)), cwd=cwd, check=True)
    if INSTALL_DEPENDENCIES:
        if importlib.util.find_spec('torch') is None:
            command([sys.executable, '-m', 'pip', 'install', 'torch>=2.2,<3'])
        command([sys.executable, '-m', 'pip', 'install', 'numpy>=1.26,<3', 'scipy>=1.11,<2',
                 'h5py>=3.10,<4', 'pandas>=2.1,<4', 'matplotlib>=3.8,<4'])
        if RUN_FITS_REFERENCE:
            command([sys.executable, '-m', 'pip', 'install', 'pyyaml>=6,<7'])
        if RUN_FLY_REFERENCE:
            command([sys.executable, '-m', 'pip', 'install', 'brian2>=2.7,<3',
                     'pyarrow>=15,<25', 'joblib>=1.3,<2', 'brotli>=1,<2'])
    import torch, numpy as np, pandas as pd
    print('Python:', sys.version, '\\nPyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')
    assert PROFILE in ('smoke', 'confirmation')
    if PROFILE == 'confirmation':
        assert CONFIRM_FULL_RUN, 'Set CONFIRM_FULL_RUN=True to acknowledge the full study.'
        assert torch.cuda.is_available(), 'Select a GPU runtime; CPU runs cannot satisfy the systems gate.'
    """)
    code("""
    # 3. Persistent working directory and a pinned dependency checkout.
    BASE_REVISION = 'ee150bd1b0051830f8e3e959805374f4efac9deb'
    if IN_COLAB and USE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        WORK = Path('/content/drive/MyDrive/A-SNN-research-restart')
    else:
        WORK = Path(LOCAL_WORK_DIR).expanduser().resolve() if LOCAL_WORK_DIR else Path.cwd() / 'research_restart'
    WORK.mkdir(parents=True, exist_ok=True)
    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    def pinned_checkout(url, revision, path):
        path = Path(path).resolve()
        if not path.exists():
            command(['git', 'clone', '--no-checkout', url, path])
            command(['git', '-C', path, 'checkout', '--detach', revision])
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        if actual != revision or dirty:
            raise RuntimeError(f'Checkout must be clean at {revision}: {path}. Choose a fresh folder; no reset is performed.')
        return path
    # Existing local A-SNN can be used without modifying its Git state.
    candidates = [Path.cwd(), *Path.cwd().parents]
    REPO = next((p for p in candidates if (p / 'gen5/ammc_gen5/shd_benchmark.py').is_file()), None)
    if REPO is None:
        REPO = pinned_checkout('https://github.com/FaisalTabrez/A-SNN.git', BASE_REVISION,
                               Path('/content/A-SNN-evidence1') if IN_COLAB else WORK / 'A-SNN')
    REPO = REPO.resolve()
    DATA = WORK / 'data'
    OUTPUT = WORK / ('evidence1_v2_' + PROFILE)
    print('Repository:', REPO, '\\nPersistent work:', WORK, '\\nResults:', OUTPUT)
    """)
    md("""
    ## Portable implementation snapshot
    This notebook includes our new runner and adapters, so it works even when
    the pinned GitHub base predates them. Each embedded file is SHA-256 checked.
    It will create missing files but **refuses to overwrite a different local
    implementation**. Use a fresh checkout if you intentionally changed the code.
    Original A-SNN modules come from the checkout; all loaded dependency hashes
    are recorded in the run protocol. Local checkouts need not equal the pinned base.
    """)
    payload = {p: {"sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest(),
                   "base64": base64.b64encode((ROOT / p).read_bytes()).decode()} for p in SNAPSHOTS}
    code("SNAPSHOTS = " + repr(payload) + "\n" + textwrap.dedent("""
    for relative, snapshot in SNAPSHOTS.items():
        content = base64.b64decode(snapshot['base64'])
        assert hashlib.sha256(content).hexdigest() == snapshot['sha256']
        path = REPO / relative
        if path.exists():
            assert sha256(path) == snapshot['sha256'], f'Different existing file: {path}. Use a fresh checkout.'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    sys.path.insert(0, str(REPO / 'gen5'))
    from ammc_gen5.evidence1_audio import EvidenceConfig, make_model, ARMS
    from ammc_gen5.artifact_integrity import atomic_json, runtime_identity
    atomic_json(WORK / 'runtime_freeze.json', runtime_identity(torch))
    config = EvidenceConfig.confirmation() if PROFILE == 'confirmation' else EvidenceConfig()
    print(config)
    for dataset, classes in [('shd', 20), ('ssc', 35)]:
        if PROFILE == 'smoke':
            classes, inputs, duration = 3, 12, 1.0
        else:
            inputs, duration = 700, 1.4 if dataset == 'shd' else 1.0
        print(dataset, {arm: sum(p.numel() for p in make_model(arm, config, inputs, classes, duration).parameters()) for arm in ARMS})
    """).strip())
    code("""
    # 4. Official SHD/SSC download — MD5 archive check plus raw SHA-256 provenance.
    # No downloads are needed for synthetic smoke. Raw data is NOT in the result ZIP.
    import urllib.request, gzip, shutil
    def fetch(url, destination, checksum=None, algorithm='sha256'):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        def digest(path):
            h = hashlib.new(algorithm)
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
            return h.hexdigest()
        if destination.exists():
            if checksum and digest(destination) != checksum:
                raise ValueError(f'Checksum mismatch for existing {destination}; move it aside and retry.')
            return destination
        temporary = destination.with_suffix(destination.suffix + '.part')
        print('Downloading', url)
        request = urllib.request.Request(url, headers={'User-Agent': 'A-SNN-research-notebook/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as target:
            shutil.copyfileobj(response, target)
        if checksum and digest(temporary) != checksum:
            raise ValueError(f'Checksum mismatch: {temporary}; destination was not replaced.')
        temporary.replace(destination)
        return destination
    if DOWNLOAD_AUDIO or PROFILE == 'confirmation' or RUN_FITS_REFERENCE:
        base = 'https://zenkelab.org/datasets'
        with urllib.request.urlopen(base + '/md5sums.txt', timeout=120) as response:
            md5_text = response.read().decode()
        md5 = {row.split()[1].lstrip('*'): row.split()[0] for row in md5_text.splitlines() if len(row.split()) == 2}
        provenance = {'source': base, 'md5sums_text': md5_text, 'raw_sha256': {}}
        for dataset, splits in [('shd', ['train', 'test']), ('ssc', ['train', 'valid', 'test'])]:
            for split in splits:
                name = f'{dataset}_{split}.h5.gz'
                assert name in md5, f'Official checksum missing for {name}'
                archive = fetch(base + '/' + name, DATA / dataset / name, md5[name], 'md5')
                raw = archive.with_suffix('')
                # Hash the gzip content even for an existing raw file: it must match the verified archive.
                h = hashlib.sha256()
                with gzip.open(archive, 'rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
                if raw.exists():
                    assert sha256(raw) == h.hexdigest(), f'Raw file differs from official archive: {raw}'
                else:
                    temporary = raw.with_suffix('.h5.part')
                    with gzip.open(archive, 'rb') as stream, temporary.open('wb') as target: shutil.copyfileobj(stream, target)
                    temporary.replace(raw)
                provenance['raw_sha256'][str(raw.relative_to(DATA))] = h.hexdigest()
        (DATA / 'download_provenance.json').write_text(json.dumps(provenance, indent=2))
        print('Official audio verified:', provenance['raw_sha256'])
    else:
        print('Audio download skipped: synthetic smoke profile.')
    """)
    md("""
    ## Run the registered runner
    Validation selects the earliest best epoch. The final test set is evaluated
    only after training; zero-state/direct-only and globally deranged-state probes
    reuse that checkpoint. No architecture tuning is performed on test results.
    Restart this cell to resume at epoch boundaries. Changing source, configuration,
    data or environment requires a **new output directory**, not overwriting results.
    Confirmation is fixed at 64 bins, 700 channels, seeds 310–319, 133,631-parameter
    ceiling, AdamW, 15 epochs and batch 256. FiTS's 250-bin/140-channel recipe is
    intentionally **not** a matched control.
    """)
    code("""
    # 5. Execute. Logs stream live; Ctrl+C preserves the last completed epoch.
    command([sys.executable, REPO / 'gen5/examples/evidence1_audio.py', '--profile', PROFILE,
             '--output-dir', OUTPUT, '--data-root', DATA,
             '--device', 'cuda' if torch.cuda.is_available() else 'cpu'], cwd=REPO)
    """)
    code("""
    # 6. Results, confidence intervals and honest decision status.
    results = json.loads((OUTPUT / 'results.json').read_text())
    records = pd.DataFrame(results['records'])
    display(records[['dataset', 'seed', 'arm', 'accuracy', 'best_validation_accuracy', 'best_epoch', 'parameters', 'train_seconds']])
    print(json.dumps(results['summary'], indent=2))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    for ax, dataset in zip(axes, ['shd', 'ssc']):
        subset = records[records.dataset == dataset]
        subset.groupby('arm').accuracy.mean().reindex(ARMS).plot.bar(ax=ax)
        ax.set(title=dataset.upper() + (' — SYNTHETIC SMOKE' if PROFILE == 'smoke' else ' — test accuracy'), ylim=(0, 1), ylabel='Accuracy')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'accuracy_overview.png', dpi=150)
    plt.show()
    print('Smoke cannot unlock a research gate. Clean-clone replication and a reviewed decision lock remain required.')
    """)
    md("""
    ## Optional literature assets (code pinned, papers provenance-recorded)
    Sources: [FiTS](https://github.com/kaistmm/FiTS),
    [timing-only SHD](https://github.com/neural-reckoning/temporal-shd),
    [BulletTrains](https://github.com/ToddMorrill/snn-bullet-trains),
    [BrainTrace](https://github.com/chaobrain/braintrace),
    [dual-memory pathways](https://github.com/sunpengfei1122/Dual_memory_pathways),
    [fly LIF reference](https://github.com/philshiu/Drosophila_brain_model).
    Downloading does not execute or endorse their mechanisms. Review upstream
    licenses before integrating code. Closed dual-memory and sparse-scatter claims
    are not reopened by these assets. Male-CNS/Eon/ConnectomeX are background only;
    they are not interchangeable with the pinned Shiu/FlyWire reference here.
    """)
    code("""
    # 7. Download pinned code. Paper URLs track latest revisions, not immutable versions.
    REFERENCES = {
        'fits': ('kaistmm/FiTS', '8fe7e376bae13fe75f357dcfbff6adf9c45f1077'),
        'timing': ('neural-reckoning/temporal-shd', '65d8f4d9589d9e1f48c073f877ac44830afd61a2'),
        'bullettrains': ('ToddMorrill/snn-bullet-trains', 'a7717295b129d6afd45bb560e393893b0a344ad5'),
        'braintrace': ('chaobrain/braintrace', '365a88db39ca2cdbbe7291671d548347eeb25b86'),
        'dual_memory': ('sunpengfei1122/Dual_memory_pathways', '05cc0a3306b311e37ef9bc439ba7ac5186242672'),
        'fly': ('philshiu/Drosophila_brain_model', '91bdd1e7dcf193f3e7ca5a8933497fcef63b7960'),
    }
    UPSTREAM = {}
    requested = set(REFERENCES) if DOWNLOAD_REFERENCES else set()
    if RUN_FITS_REFERENCE: requested.add('fits')
    if RUN_FLY_REFERENCE: requested.add('fly')
    if DOWNLOAD_TIMING_DATA: requested.add('timing')
    for name in sorted(requested):
        owner, revision = REFERENCES[name]
        UPSTREAM[name] = pinned_checkout('https://github.com/' + owner + '.git', revision, WORK / 'references' / name)
    asset_manifest = {'repositories': {k: {'repository': REFERENCES[k][0], 'commit': REFERENCES[k][1]} for k in UPSTREAM}, 'papers': {}}
    if DOWNLOAD_REFERENCES:
        for name, identifier in [('FiTS', '2605.13071'), ('BeyondRateCoding', '2507.16043'),
                                 ('BulletTrains', '2603.13283'), ('DendriCL_preprint', '2607.02283'),
                                 ('ConnectomeX_preprint', '2610.08418')]:
            url = 'https://arxiv.org/pdf/' + identifier
            paper = fetch(url, WORK / 'papers' / (name + '.pdf'))
            assert paper.read_bytes()[:5] == b'%PDF-', f'Not a PDF: {paper}'
            asset_manifest['papers'][name] = {'url': url, 'sha256': sha256(paper),
                'revision_pinned': False, 'provenance': 'observed download hash, not immutable revision'}
    (WORK / 'reference_assets.json').write_text(json.dumps(asset_manifest, indent=2))
    print('Reference assets:', asset_manifest)
    """)
    code("""
    # 8. Timing-only SHD Zenodo download; no training here until E1 decision is locked.
    if DOWNLOAD_TIMING_DATA:
        with urllib.request.urlopen('https://zenodo.org/api/records/16153275', timeout=120) as response:
            metadata = json.load(response)
        files = metadata['files']
        display(pd.DataFrame([{'name': f['key'], 'bytes': f['size'], 'checksum': f['checksum']} for f in files]))
        selected = files if TIMING_FILE_NAMES is None else [f for f in files if f['key'] in TIMING_FILE_NAMES]
        if TIMING_FILE_NAMES is not None:
            assert set(TIMING_FILE_NAMES) == {f['key'] for f in selected}, 'Unknown timing-data file name'
        size = sum(f['size'] for f in selected) / 1e9
        assert size <= TIMING_DOWNLOAD_LIMIT_GB, f'{size:.2f} GB exceeds configured budget; inspect filenames and change the limit.'
        for file in selected:
            assert Path(file['key']).name == file['key'], 'Unsafe remote filename'
            algorithm, checksum = file['checksum'].split(':', 1)
            assert algorithm in ('md5', 'sha256')
            fetch(file['links']['self'], DATA / 'timing_only' / file['key'], checksum, algorithm)
        (DATA / 'timing_only/zenodo_metadata.json').write_text(json.dumps(metadata, indent=2))
        print('Verified timing-only files; archive extraction/training intentionally not automatic.')
    else:
        print('Timing-only dataset download disabled.')
    """)
    code("""
    # 9. Optional FiTS native-recipe references (NOT matched Evidence-1 arms).
    if RUN_FITS_REFERENCE:
        for dataset in ['shd', 'ssc']:
            command([sys.executable, REPO / 'gen5/tools/fits_validation_reference.py', '--upstream', UPSTREAM['fits'],
                     '--dataset', dataset, '--data-root', DATA, '--output-dir', WORK / 'fits_reference' / dataset,
                     '--seed', 310, '--epochs', 100], cwd=UPSTREAM['fits'])
            summary = json.loads((WORK / 'fits_reference' / dataset / 'validation_reference_summary.json').read_text())
            print(dataset, summary)
    else:
        print('FiTS native-recipe reference disabled; no test-only selection is used.')
    """)
    code("""
    # 10. Optional fly reference diagnostic with author-example comparison.
    # It does not train A-SNN or change its wiring. The actual topology-prior pilot is gated.
    if RUN_FLY_REFERENCE:
        command([sys.executable, REPO / 'gen5/tools/fly_reference.py', '--upstream', UPSTREAM['fly'],
                 '--output-dir', WORK / ('fly_' + FLY_PROFILE), '--profile', FLY_PROFILE], cwd=UPSTREAM['fly'])
        display(pd.read_csv(WORK / ('fly_' + FLY_PROFILE) / 'firing_rates_hz.csv').head())
    else:
        print('Whole-fly diagnostic disabled. Enable it only when you have CPU time and memory available.')
    """)
    md("""
    ## Download the deliverables
    The final ZIP includes checkpoints, per-seed records, selection histories,
    protocol/data/environment hashes, plots, and any enabled reference-run outputs.
    Raw audio, upstream checkouts, and paper PDFs are excluded to keep it portable.
    Returned checkpoints are trusted only if created by you with this notebook.
    Bring back the ZIP for interpretation; do not tune from the test accuracy table.
    A final E1 decision still requires clean-clone repetition and human review.

    Next queued branches: rate-matched temporal perturbations and sliding-window
    readout; FiTS matched port only under a new protocol; conditional associative
    scans; a small connectome-prior pilot with signed/degree-preserving rewiring
    controls; frozen-backbone explicit-context masks. No new mechanism is opened
    automatically by the smoke run or the reference downloads.
    """)
    code("""
    # 11. Rebuild E1 bundle after plotting, verify every hash, export a combined archive.
    from ammc_gen5.evidence1_audio import bundle, digest
    bundle(OUTPUT)
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())['files']
    for relative, record in manifest.items():
        file = OUTPUT / relative
        assert file.stat().st_size == record['bytes'] and digest(file) == record['sha256'], relative
    import zipfile
    archive = WORK / 'research_restart_results.zip'
    paths = list(OUTPUT.rglob('*'))
    for folder in [WORK / 'fits_reference', WORK / ('fly_' + FLY_PROFILE)]:
        if folder.exists(): paths.extend(folder.rglob('*'))
    paths.append(WORK / 'reference_assets.json')
    if (DATA / 'download_provenance.json').exists(): paths.append(DATA / 'download_provenance.json')
    members = sorted({p for p in paths if p.is_file() and p.suffix not in ('.zip', '.tmp', '.part')})
    combined_manifest = {str(p.relative_to(WORK)).replace(os.sep, '/'): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in members}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in members: z.write(p, p.relative_to(WORK))
        z.writestr('return_manifest.json', json.dumps(combined_manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        for relative, record in combined_manifest.items():
            assert hashlib.sha256(z.read(relative)).hexdigest() == record['sha256'], relative
    print('Verified return bundle:', archive, 'SHA-256:', sha256(archive))
    if IN_COLAB and DOWNLOAD_RESULT_ZIP:
        from google.colab import files
        files.download(str(archive))
    else:
        from IPython.display import FileLink
        display(FileLink(str(archive)))
        print('In Colab: enable DOWNLOAD_RESULT_ZIP or download this file from the Files sidebar/Drive.')
    """)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "evidence1_research_restart.ipynb"},
        "asnn": {"base_revision": "ee150bd1b0051830f8e3e959805374f4efac9deb", "generated_by": "gen5/tools/build_research_notebook.py"}})
    for index, cell in enumerate(notebook.cells):
        cell.id = f"restart-{index:02d}"
    nbf.validate(notebook)
    destination = ROOT / "gen5/notebooks/evidence1_research_restart.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    # Both user-facing entry points must be the identical generated artifact.
    (ROOT / 'evidence1_research_restart.ipynb').write_bytes(destination.read_bytes())
    print(destination)


if __name__ == "__main__":
    build()
