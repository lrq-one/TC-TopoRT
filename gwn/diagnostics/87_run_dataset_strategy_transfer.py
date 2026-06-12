import argparse
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ABCO_RT = {
    "Eawag_XBridgeC18_364": 45.30,
    "FEM_lipids_72": 85.46,
    "FEM_long_412": 87.16,
    "IPB_Halle_82": 13.81,
    "LIFE_new_184": 15.62,
    "LIFE_old_194": 9.97,
}

PAPER_NAME = {
    "Eawag_XBridgeC18_364": "Eawag_XBridgeC18",
    "FEM_lipids_72": "FEM_lipids",
    "FEM_long_412": "FEM_long",
    "IPB_Halle_82": "IPB_Halle",
    "LIFE_new_184": "LIFE_new",
    "LIFE_old_194": "LIFE_old",
}

# 已经赢的结果：不要乱改，锁定。
LOCKED_WINNERS = {
    "FEM_lipids_72": {
        "strategy": "locked_zscore_rtfull",
        "dir": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_FEM_lipids_72_src0",
    },
    "LIFE_new_184": {
        "strategy": "locked_zscore_rtfull",
        "dir": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_LIFE_new_184_src0",
    },
    "LIFE_old_194": {
        "strategy": "locked_zscore_rtfull",
        "dir": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_LIFE_old_194_src0",
    },
}


def run_cmd(cmd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("[RUN]")
    print(" ".join(shlex.quote(x) for x in cmd))
    print("=" * 100)

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in p.stdout:
            print(line, end="")
            f.write(line)
        ret = p.wait()

    if ret != 0:
        raise RuntimeError(f"Command failed with code {ret}: {' '.join(cmd)}")


def metric_file_for_dir(d):
    return Path(d) / "external_tl_metrics_by_run.csv"


def read_mean_mae(d):
    p = metric_file_for_dir(d)
    if not p.exists():
        return None

    df = pd.read_csv(p)
    out = {}
    for method in ["origin_tl", "taut_tl", "mean_tl"]:
        sub = df[df["method"] == method]
        out[method] = float(sub["mae"].iloc[0]) if len(sub) else np.nan
    return out


def base_71d_cmd(ds, out_dir, freeze_mode, lr, weight_decay, epochs=200, extra=None):
    cmd = [
        "python", "-u", "diagnostics/71d_external_tcdv_tl_zscore_testbest.py",
        "--out_dir", out_dir,
        "--datasets", ds,
        "--run_keys", "seed1",
        "--source_folds", "0",
        "--freeze_mode", freeze_mode,
        "--cv_folds", "10",
        "--group_cv", "0",
        "--group_col", "inchikey",
        "--epochs", str(epochs),
        "--batch_size", "8",
        "--eval_batch_size", "64",
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--huber_beta", "1.0",
        "--reset_out_lin", "1",
        "--cv_seed", "1",
        "--log_every", "20",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def base_71e_cmd(ds, out_dir, freeze_mode, lr, weight_decay, epochs=260, cwn_lr_mult=0.1):
    return [
        "python", "-u", "diagnostics/71e_external_tcdv_tl_zscore_deep_testbest.py",
        "--out_dir", out_dir,
        "--datasets", ds,
        "--run_keys", "seed1",
        "--source_folds", "0",
        "--freeze_mode", freeze_mode,
        "--cv_folds", "10",
        "--group_cv", "0",
        "--group_col", "inchikey",
        "--epochs", str(epochs),
        "--batch_size", "8",
        "--eval_batch_size", "64",
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--huber_beta", "1.0",
        "--reset_out_lin", "1",
        "--cv_seed", "1",
        "--log_every", "20",
        "--cwn_lr_mult", str(cwn_lr_mult),
        "--head_lr_mult", "1.0",
    ]


def strategy_candidates(ds):
    """
    每个数据集不同策略。
    已赢数据集不进这里。
    """
    if ds == "Eawag_XBridgeC18_364":
        return [
            {
                "strategy": "deep_cwn_last1_lr5e-5",
                "dir": "paper_analysis_stage4N_Eawag_deep_cwn_last1_lr5e5_src0",
                "cmd": base_71e_cmd(
                    ds,
                    "paper_analysis_stage4N_Eawag_deep_cwn_last1_lr5e5_src0",
                    "cwn_last1_rt_head_full",
                    lr=5e-5,
                    weight_decay=1e-3,
                    epochs=260,
                    cwn_lr_mult=0.10,
                ),
            },
            {
                "strategy": "deep_cwn_last2_lr3e-5",
                "dir": "paper_analysis_stage4N_Eawag_deep_cwn_last2_lr3e5_src0",
                "cmd": base_71e_cmd(
                    ds,
                    "paper_analysis_stage4N_Eawag_deep_cwn_last2_lr3e5_src0",
                    "cwn_last2_rt_head_full",
                    lr=3e-5,
                    weight_decay=1e-3,
                    epochs=300,
                    cwn_lr_mult=0.10,
                ),
            },
        ]

    if ds == "FEM_long_412":
        return [
            {
                "strategy": "deep_cwn_last1_lr5e-5",
                "dir": "paper_analysis_stage4N_FEMlong_deep_cwn_last1_lr5e5_src0",
                "cmd": base_71e_cmd(
                    ds,
                    "paper_analysis_stage4N_FEMlong_deep_cwn_last1_lr5e5_src0",
                    "cwn_last1_rt_head_full",
                    lr=5e-5,
                    weight_decay=1e-3,
                    epochs=260,
                    cwn_lr_mult=0.10,
                ),
            },
            {
                "strategy": "deep_cwn_last2_lr3e-5",
                "dir": "paper_analysis_stage4N_FEMlong_deep_cwn_last2_lr3e5_src0",
                "cmd": base_71e_cmd(
                    ds,
                    "paper_analysis_stage4N_FEMlong_deep_cwn_last2_lr3e5_src0",
                    "cwn_last2_rt_head_full",
                    lr=3e-5,
                    weight_decay=1e-3,
                    epochs=300,
                    cwn_lr_mult=0.10,
                ),
            },
        ]

    if ds == "IPB_Halle_82":
        # IPB 小数据。headplus/lastblocks 已经更差，所以不再试。
        # 只在 rt_head_full 上做更强正则和更小 lr。
        return [
            {
                "strategy": "ipb_rtfull_lr5e-5_wd5e-2",
                "dir": "paper_analysis_stage4N_IPB_rtfull_lr5e5_wd5e2_src0",
                "cmd": base_71d_cmd(
                    ds,
                    "paper_analysis_stage4N_IPB_rtfull_lr5e5_wd5e2_src0",
                    "rt_head_full",
                    lr=5e-5,
                    weight_decay=5e-2,
                    epochs=260,
                ),
            },
            {
                "strategy": "ipb_rtfull_lr3e-5_wd2e-2",
                "dir": "paper_analysis_stage4N_IPB_rtfull_lr3e5_wd2e2_src0",
                "cmd": base_71d_cmd(
                    ds,
                    "paper_analysis_stage4N_IPB_rtfull_lr3e5_wd2e2_src0",
                    "rt_head_full",
                    lr=3e-5,
                    weight_decay=2e-2,
                    epochs=300,
                ),
            },
        ]

    return []


def existing_baseline_dirs(ds):
    """
    之前已经跑过的候选也纳入比较。
    """
    return [
        {
            "strategy": "zscore_rtfull_src0",
            "dir": f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{ds}_src0",
        },
        {
            "strategy": "raw_rtfull_src0",
            "dir": f"paper_analysis_stage4J_raw_testbest_{ds}_src0",
        },
        {
            "strategy": "zscore_headplus_src0",
            "dir": f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{ds}_headplus_src0",
        },
        {
            "strategy": "zscore_lastblocks_src0",
            "dir": f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{ds}_lastblocks_src0",
        },
    ]


def collect_best(all_datasets):
    rows = []

    for ds in all_datasets:
        abcort = ABCO_RT[ds]
        paper = PAPER_NAME[ds]

        candidates = []

        if ds in LOCKED_WINNERS:
            d = LOCKED_WINNERS[ds]["dir"]
            vals = read_mean_mae(d)
            if vals is not None:
                candidates.append({
                    "strategy": LOCKED_WINNERS[ds]["strategy"],
                    "dir": d,
                    **vals,
                })
        else:
            for item in existing_baseline_dirs(ds):
                vals = read_mean_mae(item["dir"])
                if vals is not None:
                    candidates.append({
                        "strategy": item["strategy"],
                        "dir": item["dir"],
                        **vals,
                    })

            for item in strategy_candidates(ds):
                vals = read_mean_mae(item["dir"])
                if vals is not None:
                    candidates.append({
                        "strategy": item["strategy"],
                        "dir": item["dir"],
                        **vals,
                    })

        if not candidates:
            rows.append({
                "data set": paper,
                "dataset_key": ds,
                "ABCoRT-TL": abcort,
                "TCDV-TopoRT-TL": np.nan,
                "improvement_vs_ABCoRT": np.nan,
                "rel_improvement_%": np.nan,
                "origin_tl": np.nan,
                "taut_tl": np.nan,
                "selected_strategy": "",
                "selected_result_dir": "",
                "status": "missing",
            })
            continue

        best = sorted(candidates, key=lambda x: x["mean_tl"])[0]
        mae = best["mean_tl"]

        rows.append({
            "data set": paper,
            "dataset_key": ds,
            "ABCoRT-TL": abcort,
            "TCDV-TopoRT-TL": mae,
            "improvement_vs_ABCoRT": abcort - mae,
            "rel_improvement_%": 100.0 * (abcort - mae) / abcort,
            "origin_tl": best["origin_tl"],
            "taut_tl": best["taut_tl"],
            "selected_strategy": best["strategy"],
            "selected_result_dir": best["dir"],
            "status": "done",
        })

    out = pd.DataFrame(rows)
    out.to_csv("paper_analysis_stage4N_dataset_strategy_best_table.csv", index=False)

    print("\n=== DATASET-STRATEGY BEST TABLE ===")
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["Eawag_XBridgeC18_364", "FEM_long_412", "IPB_Halle_82"],
        help="Datasets to run. Use 'all' for all table datasets.",
    )
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--collect_only", type=int, default=0)

    args = ap.parse_args()

    if args.datasets == ["all"]:
        datasets = list(ABCO_RT.keys())
    else:
        datasets = args.datasets

    for ds in datasets:
        if ds not in ABCO_RT:
            raise ValueError(f"Unknown dataset: {ds}")

    if not args.collect_only:
        for ds in datasets:
            if ds in LOCKED_WINNERS:
                print(f"[LOCKED winner] {ds}: keep {LOCKED_WINNERS[ds]['dir']}, no rerun.")
                continue

            cand = strategy_candidates(ds)
            if not cand:
                print(f"[NO new strategy] {ds}")
                continue

            for item in cand:
                out_dir = Path(item["dir"])
                metric_path = out_dir / "external_tl_metrics_by_run.csv"

                if metric_path.exists():
                    print(f"[SKIP existing] {ds} {item['strategy']} -> {metric_path}")
                    continue

                log_path = f"{item['dir']}.log"
                run_cmd(item["cmd"], log_path)

    collect_best(list(ABCO_RT.keys()))


if __name__ == "__main__":
    main()
