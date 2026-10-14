"""AST-only supplement for canonical Markdown classified by Graphify as papers.

Uses Graphify's extractor, merge/export and clustering, not hand-written graph
nodes or inferred findings. No semantic backend or API key is used.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gen5/tools'))
from verify_graphify_contract import canonical_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Re-extract all canonical Markdown, including changed sources already represented')
    args = parser.parse_args()
    from graphify.extract import extract
    from graphify.build import build_merge
    from graphify.cluster import cluster
    from graphify.export import to_json
    graph_path = ROOT / 'graphify-out/graph.json'
    payload = json.loads(graph_path.read_text(encoding='utf-8'))
    represented = {n.get('source_file', '').replace('\\', '/').lower() for n in payload['nodes']}
    missing = [p for p in canonical_files(ROOT) if p.suffix == '.md' and
               (args.force or p.relative_to(ROOT).as_posix().lower() not in represented)]
    if not missing:
        print('All canonical Markdown sources are represented structurally')
        return
    extraction = extract(missing, cache_root=ROOT)
    # Keep exact source identities; fuzzy cross-file merging can silently shrink
    # an otherwise complete refresh and erase distinct provenance anchors.
    graph = build_merge([extraction], graph_path=graph_path, root=ROOT, directed=False, dedup=False)
    communities = cluster(graph)
    if not to_json(graph, communities, graph_path):
        raise RuntimeError('Graphify refused graph export; existing graph retained')
    # Regenerate the report/visualization from the library-produced graph.
    subprocess.run([sys.executable, '-m', 'graphify', 'cluster-only', str(ROOT), '--no-label'], check=True)
    print('AST-only canonical Markdown supplementation:', len(missing), 'sources')
    print('Semantic freshness remains unverified')


if __name__ == '__main__': main()
