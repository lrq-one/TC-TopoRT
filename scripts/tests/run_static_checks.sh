#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

python -m compileall -q gwn scripts

while IFS= read -r script; do
    bash -n "${script}"
done < <(find scripts -type f -name '*.sh' | sort)

DRY_RUN=1 bash scripts/training/run_smrt_single_seed.sh 5 >/dev/null
python scripts/filtering/run_candidate_filtering.py --dry_run 1 >/dev/null

printf 'TC-TopoRT static checks: PASS\n'
