#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

test -f README.md
test -f LICENSE
test -f CITATION.cff
test -f configs/smrt.yaml
test -f configs/external_transfer.yaml
test -f configs/candidate_filtering.yaml
test -f data/README.md
test ! -e scripts/filtering/run_filtering_sensitivity.py

if git ls-files --error-unmatch 'data/candidate_filtering/*.csv' >/dev/null 2>&1; then
    echo "Candidate-level CSV found in the public Git index; these belong in Figshare/local storage." >&2
    exit 1
fi

for obsolete_filtering_pathspec in '*rank_guard*' '*guarded*' '*filtering_sensitivity*'; do
    if git ls-files --error-unmatch "${obsolete_filtering_pathspec}" >/dev/null 2>&1; then
        echo "Obsolete guarded/soft-filtering artifact found in the public Git index." >&2
        exit 1
    fi
done

python -m py_compile scripts/filtering/run_candidate_filtering.py
python scripts/filtering/run_candidate_filtering.py --help >/dev/null

if grep -RInE --exclude=run_static_checks.sh --exclude-dir=reviewer --exclude-dir=__pycache__ \
    'guard_k|hybrid_score|tau_seconds|9600|9900' \
    README.md configs scripts docs 2>/dev/null; then
    echo "Obsolete filtering implementation term found in the formal public paths." >&2
    exit 1
fi

echo "TC-TopoRT static checks: PASS"
