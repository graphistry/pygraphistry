#!/usr/bin/env python3
"""Extract caller information from Pysa call-graph.json

Usage:
    python3 ai/assets/pysa_extract_callers.py path/to/call-graph.json method1 method2 ...

Example:
    python3 ai/assets/pysa_extract_callers.py \\
        pysa_results/call-graph.json \\
        PlotterBase.PlotterBase.bind \\
        PlotterBase.PlotterBase.nodes
"""
import json
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

call_graph_path = sys.argv[1]
target_methods = sys.argv[2:]

call_graph = defaultdict(list)
with open(call_graph_path) as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('kind') != 'call_graph':
            continue

        caller = entry['data']['callable']
        for loc, call_info in entry['data'].get('calls', {}).items():
            for target in call_info.get('call', {}).get('calls', []):
                call_graph[target['target']].append({
                    'caller': caller,
                    'location': loc
                })

for method in target_methods:
    if method in call_graph:
        print(f"\n{method} callers ({len(call_graph[method])} total):")
        for call in call_graph[method][:10]:  # Show first 10
            print(f"  {call['caller']} at {call['location']}")
    else:
        print(f"\n{method}: No callers found")
