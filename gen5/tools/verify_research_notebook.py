"""Execute a smoke copy without package changes, network assets or real data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(root / 'gen5/notebooks/evidence1_research_restart.ipynb', as_version=4)
    nbformat.validate(notebook)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config = next(cell for cell in notebook.cells if cell.cell_type == 'code')
    config.source += '\n' + '\n'.join([
        "PROFILE = 'smoke'", 'INSTALL_DEPENDENCIES = False', 'USE_DRIVE = False',
        'DOWNLOAD_AUDIO = False', 'DOWNLOAD_REFERENCES = False', 'DOWNLOAD_TIMING_DATA = False',
        'RUN_FITS_REFERENCE = False', 'RUN_FLY_REFERENCE = False', 'DOWNLOAD_RESULT_ZIP = False',
        f'LOCAL_WORK_DIR = {str(args.work_dir.resolve())!r}',
    ])
    NotebookClient(notebook, timeout=180, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
    executed = args.work_dir / 'research_restart_smoke_executed.ipynb'
    nbformat.write(notebook, executed)
    results = json.loads((args.work_dir / 'evidence1_v2_smoke/results.json').read_text())
    assert results['summary']['decision'] == 'smoke_only'
    assert results['summary']['complete'] and len(results['records']) == 8
    print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells: {executed}')
    print(f'Eight synthetic trial records; decision={results["summary"]["decision"]}')
    print(f'Verified return ZIP: {args.work_dir / "research_restart_results.zip"}')


if __name__ == '__main__':
    main()
