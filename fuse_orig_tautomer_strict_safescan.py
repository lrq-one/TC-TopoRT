import os
import numpy as np
import pandas as pd


def metrics(y, p):
    err = np.abs(y - p)
    return {
        "MAE": float(err.mean()),
        "MedAE": float(np.median(err)),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "P95": float(np.quantile(err, 0.95)),
        "P99": float(np.quantile(err, 0.99)),
        ">100": int((err > 100).sum()),
        ">200": int((err > 200).sum()),
        "N": int(len(err)),
    }


def load(path):
    return pd.read_csv(path)


base_dir = "results/TopoCellRT"
taut_dir = "results/TopoCellRT_tautomer_strict"
out_dir = "results/TopoCellRT_fuse_orig_tautomer_strict_safescan"
os.makedirs(out_dir, exist_ok=True)

val0 = load(f"{base_dir}/val_predictions.csv")
val1 = load(f"{taut_dir}/val_predictions.csv")
test0 = load(f"{base_dir}/test_predictions.csv")
test1 = load(f"{taut_dir}/test_predictions.csv")

yv = val0["y_true"].values
yt = test0["y_true"].values

p0v = val0["y_pred"].values
p1v = val1["y_pred"].values
p0t = test0["y_pred"].values
p1t = test1["y_pred"].values

base_val = metrics(yv, p0v)
base_test = metrics(yt, p0t)

rows = []

for alpha in np.linspace(0, 1, 1001):
    pv = alpha * p0v + (1 - alpha) * p1v
    pt = alpha * p0t + (1 - alpha) * p1t

    mv = metrics(yv, pv)
    mt = metrics(yt, pt)

    rows.append({
        "alpha": float(alpha),

        "val_MAE": mv["MAE"],
        "val_MedAE": mv["MedAE"],
        "val_RMSE": mv["RMSE"],
        "val_P95": mv["P95"],
        "val_P99": mv["P99"],
        "val_100": mv[">100"],
        "val_200": mv[">200"],

        "test_MAE": mt["MAE"],
        "test_MedAE": mt["MedAE"],
        "test_RMSE": mt["RMSE"],
        "test_P95": mt["P95"],
        "test_P99": mt["P99"],
        "test_100": mt[">100"],
        "test_200": mt[">200"],
    })

scan = pd.DataFrame(rows)
scan.to_csv(f"{out_dir}/alpha_scan.csv", index=False)

# 1. 纯 MAE 最优，当前版本
mae_best = scan.loc[scan["val_MAE"].idxmin()]

# 2. val MAE 优于 baseline，且 val P99 不超过 baseline P99
safe = scan[
    (scan["val_MAE"] < base_val["MAE"]) &
    (scan["val_P99"] <= base_val["P99"])
].copy()

if len(safe) > 0:
    # 在 P99 安全的候选里，选 val MAE 最低
    p99_safe_best = safe.loc[safe["val_MAE"].idxmin()]
else:
    p99_safe_best = None

# 3. val MAE 优于 baseline，且 val >200 不超过 baseline >200
tail = scan[
    (scan["val_MAE"] < base_val["MAE"]) &
    (scan["val_200"] <= base_val[">200"])
].copy()

if len(tail) > 0:
    # 在 >200 不变差的候选里，选 val MAE 最低
    tail_safe_best = tail.loc[tail["val_MAE"].idxmin()]
else:
    tail_safe_best = None

# 4. 保守折中：val MAE 排名前 20% 的候选里，选 val P99 最低
improved = scan[scan["val_MAE"] < base_val["MAE"]].copy()
if len(improved) > 0:
    cutoff = improved["val_MAE"].quantile(0.2)
    candidates = improved[improved["val_MAE"] <= cutoff]
    p99_min_best = candidates.loc[candidates["val_P99"].idxmin()]
else:
    p99_min_best = None


def print_row(name, row):
    if row is None:
        print(f"\n{name}: None")
        return

    print(f"\n{name}")
    print("alpha:", row["alpha"])
    print("VAL :",
          "MAE", row["val_MAE"],
          "MedAE", row["val_MedAE"],
          "RMSE", row["val_RMSE"],
          "P95", row["val_P95"],
          "P99", row["val_P99"],
          ">100", int(row["val_100"]),
          ">200", int(row["val_200"]))
    print("TEST:",
          "MAE", row["test_MAE"],
          "MedAE", row["test_MedAE"],
          "RMSE", row["test_RMSE"],
          "P95", row["test_P95"],
          "P99", row["test_P99"],
          ">100", int(row["test_100"]),
          ">200", int(row["test_200"]))


print("BASE VAL :", base_val)
print("BASE TEST:", base_test)

print_row("MAE_BEST", mae_best)
print_row("P99_SAFE_BEST", p99_safe_best)
print_row("TAIL_SAFE_BEST", tail_safe_best)
print_row("P99_MIN_AMONG_GOOD_MAE", p99_min_best)

with open(f"{out_dir}/safe_fusion_report.txt", "w") as f:
    f.write("BASE VAL : " + str(base_val) + "\n")
    f.write("BASE TEST: " + str(base_test) + "\n\n")
    for name, row in [
        ("MAE_BEST", mae_best),
        ("P99_SAFE_BEST", p99_safe_best),
        ("TAIL_SAFE_BEST", tail_safe_best),
        ("P99_MIN_AMONG_GOOD_MAE", p99_min_best),
    ]:
        f.write(name + "\n")
        if row is None:
            f.write("None\n\n")
        else:
            f.write(str(row.to_dict()) + "\n\n")
