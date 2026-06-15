from pathlib import Path
import json
import pandas as pd
import numpy as np

RUN_DIRS = [
    "results_OOF_DualView_Stack_v1",
    "results_OOF_DualView_Stack_seed5",
    "results_OOF_DualView_Stack_seed79",
    "results_OOF_DualView_Stack_seed123",
    "results_OOF_DualView_Stack_seed256",
]

KEYS = {
    "mae": ["mae", "test_mae", "MAE"],
    "medae": ["medae", "median_ae", "MedAE", "test_medae"],
    "rmse": ["rmse", "RMSE", "test_rmse"],
    "r2": ["r2", "R2", "test_r2"],
    "bias": ["bias", "Bias", "test_bias"],
    "p95": ["p95", "p95_abs_err", "P95"],
    "p99": ["p99", "p99_abs_err", "P99"],
    "err_gt_100": ["err_gt_100", ">100", "gt100"],
    "err_gt_200": ["err_gt_200", ">200", "gt200"],
}

def norm_key(k):
    return str(k).strip().lower().replace(" ", "_").replace("-", "_")

def flatten_json(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, kk))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            kk = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_json(v, kk))
    else:
        out[prefix] = x
    return out

def pick_metric_from_dict(d, aliases):
    nd = {norm_key(k): v for k, v in d.items()}
    for a in aliases:
        aa = norm_key(a)
        for k, v in nd.items():
            if k == aa or k.endswith("." + aa) or aa in k:
                try:
                    return float(v)
                except Exception:
                    pass
    return np.nan

def scan_json(path):
    rows = []
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return rows

    flat = flatten_json(obj)
    row = {
        "source_file": str(path),
        "source_type": "json",
        "row_id": "",
        "name": path.stem,
    }
    for metric, aliases in KEYS.items():
        row[metric] = pick_metric_from_dict(flat, aliases)

    if not pd.isna(row.get("mae", np.nan)):
        rows.append(row)
    return rows

def scan_csv(path):
    rows = []
    try:
        df = pd.read_csv(path)
    except Exception:
        return rows

    lower_cols = {norm_key(c): c for c in df.columns}

    # 只要有 MAE 类列，就认为可能是结果文件
    mae_col = None
    for cand in ["mae", "test_mae", "MAE"]:
        cc = norm_key(cand)
        if cc in lower_cols:
            mae_col = lower_cols[cc]
            break
    if mae_col is None:
        return rows

    for i, r in df.iterrows():
        row = {
            "source_file": str(path),
            "source_type": "csv",
            "row_id": i,
            "name": "",
        }

        for name_col in ["name", "Name", "method", "Method", "stacker", "model"]:
            if name_col in df.columns:
                row["name"] = str(r[name_col])
                break
        if row["name"] == "":
            row["name"] = path.stem

        for metric, aliases in KEYS.items():
            val = np.nan
            for a in aliases:
                aa = norm_key(a)
                if aa in lower_cols:
                    try:
                        val = float(r[lower_cols[aa]])
                    except Exception:
                        val = np.nan
                    break
            row[metric] = val

        if not pd.isna(row["mae"]):
            rows.append(row)

    return rows

all_rows = []

for d in RUN_DIRS:
    root = Path(d)
    print("\n" + "=" * 120)
    print("[SCAN]", d, "exists=", root.exists())

    if not root.exists():
        continue

    files = list(root.rglob("*.csv")) + list(root.rglob("*.json"))
    print("[FILES]", len(files))

    for f in files:
        if f.suffix.lower() == ".json":
            all_rows.extend(scan_json(f))
        elif f.suffix.lower() == ".csv":
            all_rows.extend(scan_csv(f))

if not all_rows:
    print("[ERROR] no metric rows found")
    raise SystemExit

res = pd.DataFrame(all_rows)

# 从路径里标记 seed/run
def infer_run(source_file):
    s = str(source_file)
    for d in RUN_DIRS:
        if d in s:
            return d
    return ""

res["run_dir"] = res["source_file"].map(infer_run)

# 粗略标记是不是最终 stacker / test
def score_row(r):
    txt = (str(r.get("name", "")) + " " + str(r.get("source_file", ""))).lower()
    score = 0
    if "huber" in txt:
        score += 100
    if "stack" in txt:
        score += 50
    if "final" in txt:
        score += 20
    if "test" in txt:
        score += 10
    return score

res["priority"] = res.apply(score_row, axis=1)
res = res.sort_values(["run_dir", "priority", "mae"], ascending=[True, False, True])

out_dir = Path("final_smrt_results")
out_dir.mkdir(exist_ok=True)

res.to_csv(out_dir / "smrt_multiseed_all_metric_candidates.csv", index=False)

print("\n=== Candidate metric rows ===")
cols = ["run_dir", "name", "mae", "medae", "rmse", "r2", "bias", "p95", "p99", "err_gt_100", "err_gt_200", "priority", "source_file"]
print(res[cols].to_string(index=False))

print("\n=== Top candidates per run_dir ===")
tops = res.sort_values(["run_dir", "priority", "mae"], ascending=[True, False, True]).groupby("run_dir", as_index=False).head(8)
print(tops[cols].to_string(index=False))

print("\n[SAVE] final_smrt_results/smrt_multiseed_all_metric_candidates.csv")
