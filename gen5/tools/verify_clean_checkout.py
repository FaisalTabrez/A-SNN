"""Verify exact committed software in a fresh checkout; not an independent research replication."""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True, help='Local repository or remote Git URL')
    parser.add_argument('--revision', required=True, help='Full immutable commit SHA')
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision): raise ValueError('Use a full immutable commit SHA')
    if args.work_dir.exists(): raise ValueError('Choose a nonexistent clean checkout path; no reset/delete is performed')
    def run(command, **kwargs): subprocess.run(command, check=True, **kwargs)
    run(['git', '-c', 'core.autocrlf=false', 'clone', '--no-local', '--no-checkout', args.repository, str(args.work_dir)])
    run(['git', '-C', str(args.work_dir), '-c', 'core.autocrlf=false', 'checkout', '--detach', args.revision])
    actual = subprocess.check_output(['git', '-C', str(args.work_dir), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != args.revision: raise ValueError('Wrong checkout identity')
    environment = dict(os.environ); environment.pop('PYTHONPATH', None)
    for command in [[sys.executable, 'gen5/tools/verify_research_release.py', '--require-tracked'],
                    [sys.executable, '-m', 'unittest', 'discover', '-s', 'gen5/tests', '-q'],
                    [sys.executable, 'gen5/tools/verify_research_notebook.py', '--work-dir', 'tmp/clean_notebook_smoke'],
                    [sys.executable, 'gen5/tools/verify_graphify_contract.py', '--require-research', '--require-corpus']]:
        run(command, cwd=args.work_dir, env=environment)
    receipt = {'revision': actual, 'source_clean': True, 'software_checks': 'passed',
               'environment': 'current Python dependencies reused; no clean-environment or independent-executor claim',
               'research_replication_certified': False}
    (args.work_dir / 'tmp/clean_checkout_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
