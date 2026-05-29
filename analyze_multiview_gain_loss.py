import os
import numpy as np
import pandas as pd


def analyze(split):
    if split == "test":
        audit_path = "SMRT_data/data/SMRT_test_tautomer_strict_audit.csv"
    else:
        # val 是从 train random_split 出来的，直接对齐 audit 会麻烦。
        # 这里先分析 test，最关键。
        return

    base = pd.read_csv(f"results/TopoCellRT/{split}_predictions.csv")
    taut = pd.read_csv(f"results/TopoCellRT_tautomer_strict/{split}_predictions.csv")
    fused = pd.read_csv(f"results/TopoCellRT_fuse_orig_tautomer_strict/{split}_predictions_fused.csv")
    audit = pd.read_csv(audit_path)

    y = base["y_true"].values
    e_base = np.abs(y - base["y_pred"].values)
    e_taut = np.abs(y - taut["y_pred"].values)
    e_fused = np.abs(y - fused["y_pred"].values)

    df = pd.DataFrame({
        "idx": np.arange(len(y)),
        "y_true": y,
        "pred_base": base["y_pred"].values,
        "pred_taut": taut["y_pred"].values,
        "pred_fused": fused["y_pred"].values,
        "err_base": e_base,
        "err_taut": e_taut,
        "err_fused": e_fused,
        "gain_fused_vs_base": e_base - e_fused,
        "gain_taut_vs_base": e_base - e_taut,
        "disagree_base_taut": np.abs(base["y_pred"].values - taut["y_pred"].values),
    })

    df = df.merge(
        audit[["idx", "orig_smile", "new_smile", "real_changed"]],
        on="idx",
        how="left"
    )

    os.makedirs("results/multiview_diagnostics", exist_ok=True)

    df.to_csv(f"results/multiview_diagnostics/{split}_multiview_gain_loss.csv", index=False)

    print("\n===", split, "===")
    print("mean gain fused vs base:", df["gain_fused_vs_base"].mean())
    print("median gain fused vs base:", df["gain_fused_vs_base"].median())

    for flag in [0, 1]:
        sub = df[df["real_changed"] == flag]
        print(f"\nreal_changed={flag}, n={len(sub)}")
        print("base MAE :", sub["err_base"].mean())
        print("taut MAE :", sub["err_taut"].mean())
        print("fused MAE:", sub["err_fused"].mean())
        print("fused gain:", sub["gain_fused_vs_base"].mean())
        print(">100 base/fused:", int((sub["err_base"] > 100).sum()), int((sub["err_fused"] > 100).sum()))
        print(">200 base/fused:", int((sub["err_base"] > 200).sum()), int((sub["err_fused"] > 200).sum()))

    improved = df.sort_values("gain_fused_vs_base", ascending=False).head(50)
    worsened = df.sort_values("gain_fused_vs_base", ascending=True).head(50)

    improved.to_csv(f"results/multiview_diagnostics/{split}_top50_improved_by_fusion.csv", index=False)
    worsened.to_csv(f"results/multiview_diagnostics/{split}_top50_worsened_by_fusion.csv", index=False)

    print("\nTop improved saved:")
    print(f"results/multiview_diagnostics/{split}_top50_improved_by_fusion.csv")
    print("Top worsened saved:")
    print(f"results/multiview_diagnostics/{split}_top50_worsened_by_fusion.csv")


analyze("test")
