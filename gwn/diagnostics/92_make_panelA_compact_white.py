import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")


def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "smiles" in df.columns and "smile" not in df.columns:
        df = df.rename(columns={"smiles": "smile"})
    return df


def canon_smiles(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def mol_from_smiles(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return None
        AllChem.Compute2DCoords(m)
        return m
    except Exception:
        return None


def formula(mol):
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return ""


def draw_mol_white_png(mol, out_png, width=520, height=360):
    mol = Chem.Mol(mol)
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()

    # 关键：不要透明背景
    opts.clearBackground = True
    opts.addAtomIndices = False
    opts.addStereoAnnotation = False
    opts.bondLineWidth = 2
    opts.padding = 0.05

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    png_bytes = drawer.GetDrawingText()
    out_png = Path(out_png)
    out_png.write_bytes(png_bytes)

    # 再强制铺白底，防止 AI / draw.io 把透明区域显示成黑色
    img = Image.open(out_png).convert("RGBA")
    white = Image.new("RGBA", img.size, "white")
    white.alpha_composite(img)
    white.convert("RGB").save(out_png, quality=95)


def draw_mol_white_svg(mol, out_svg, width=520, height=360):
    mol = Chem.Mol(mol)
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.addAtomIndices = False
    opts.addStereoAnnotation = False
    opts.bondLineWidth = 2
    opts.padding = 0.05

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # 给 SVG 手动加白色背景矩形，防止 AI 透明底变黑
    insert = f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>'
    svg = svg.replace(">", ">" + insert, 1)

    Path(out_svg).write_text(svg, encoding="utf-8")


def crop_white_margin(in_png, out_png, margin=20):
    img = Image.open(in_png).convert("RGB")
    arr = np.asarray(img)

    # 找非白区域
    mask = np.any(arr < 245, axis=2)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        img.save(out_png)
        return

    x0 = max(0, xs.min() - margin)
    x1 = min(img.width, xs.max() + margin)
    y0 = max(0, ys.min() - margin)
    y1 = min(img.height, ys.max() + margin)

    cropped = img.crop((x0, y0, x1, y1))
    cropped.save(out_png, quality=95)


def add_label_card(in_png, out_png, title, subtitle, border_color):
    img = Image.open(in_png).convert("RGB")

    # 固定卡片大小，方便放进 AI
    W, H = 620, 430
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    draw.rounded_rectangle(
        [3, 3, W - 4, H - 4],
        radius=16,
        outline=border_color,
        width=4,
        fill="white",
    )

    draw.text((W // 2, 25), title, anchor="mm", fill="#111111", font=font_title)

    # 缩放分子图到卡片中间
    max_w, max_h = 520, 270
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    new_size = (int(img.width * scale), int(img.height * scale))
    img2 = img.resize(new_size, Image.Resampling.LANCZOS)

    x = (W - img2.width) // 2
    y = 85 + (max_h - img2.height) // 2
    canvas.paste(img2, (x, y))

    draw.text((W // 2, H - 25), subtitle, anchor="mm", fill="#444444", font=font_sub)
    canvas.save(out_png, quality=95)


def make_triplet(paths, out_png):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    gap = 24
    pad = 20
    W = sum(i.width for i in imgs) + gap * 2 + pad * 2
    H = max(i.height for i in imgs) + pad * 2

    canvas = Image.new("RGB", (W, H), "white")
    x = pad
    for img in imgs:
        y = pad + (H - 2 * pad - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width + gap

    canvas.save(out_png, quality=95)


def load_data(origin_csv, taut_csv, rt_min):
    origin = norm_cols(pd.read_csv(origin_csv, engine="python"))
    taut = norm_cols(pd.read_csv(taut_csv, engine="python"))

    if "smile" not in origin.columns:
        raise ValueError(f"origin csv must have smile/smiles column, got {origin.columns.tolist()}")
    if "smile" not in taut.columns:
        raise ValueError(f"taut csv must have smile/smiles column, got {taut.columns.tolist()}")

    origin["rt"] = pd.to_numeric(origin["rt"], errors="coerce")
    valid = origin[origin["rt"] > float(rt_min)].copy().reset_index(drop=False)
    valid = valid.rename(columns={"index": "source_row"})

    if len(taut) == len(valid):
        valid["taut_smile"] = taut["smile"].astype(str).values
    elif len(taut) == len(origin):
        valid["taut_smile"] = taut.iloc[valid["source_row"].values]["smile"].astype(str).values
    else:
        raise ValueError(f"taut length mismatch: taut={len(taut)}, valid={len(valid)}, origin={len(origin)}")

    valid["origin_smile"] = valid["smile"].astype(str)
    valid["origin_can"] = valid["origin_smile"].map(canon_smiles)
    valid["taut_can"] = valid["taut_smile"].map(canon_smiles)
    valid["taut_changed"] = (valid["origin_can"].astype(str) != valid["taut_can"].astype(str)).astype(int)
    return valid


def mol_bbox_aspect(mol, width=520, height=360):
    tmp = Path("_tmp_rdkit_aspect.png")
    draw_mol_white_png(mol, tmp, width=width, height=height)
    img = Image.open(tmp).convert("RGB")
    arr = np.asarray(img)
    mask = np.any(arr < 245, axis=2)
    ys, xs = np.where(mask)
    tmp.unlink(missing_ok=True)

    if len(xs) == 0:
        return 999.0

    bw = max(1, xs.max() - xs.min() + 1)
    bh = max(1, ys.max() - ys.min() + 1)
    return float(bw / bh)


def select_compact_changed(df, max_heavy=28, min_heavy=10, max_aspect=2.2, top_k_print=20):
    candidates = []

    for i, row in df.iterrows():
        if int(row["taut_changed"]) != 1:
            continue

        mo = mol_from_smiles(row["origin_smile"])
        mt = mol_from_smiles(row["taut_smile"])
        if mo is None or mt is None:
            continue

        if formula(mo) != formula(mt):
            continue

        heavy = mo.GetNumHeavyAtoms()
        if heavy < min_heavy or heavy > max_heavy:
            continue

        asp = mol_bbox_aspect(mo)
        if asp > max_aspect:
            continue

        # 越接近 18 个重原子，越紧凑，优先
        score = abs(heavy - 18) + asp * 1.5
        candidates.append((score, i, heavy, asp, row))

    if not candidates:
        print("[WARN] no compact changed molecule found under current filters; relaxing aspect/heavy filters.")
        for i, row in df.iterrows():
            if int(row["taut_changed"]) != 1:
                continue
            mo = mol_from_smiles(row["origin_smile"])
            mt = mol_from_smiles(row["taut_smile"])
            if mo is None or mt is None:
                continue
            if formula(mo) != formula(mt):
                continue
            heavy = mo.GetNumHeavyAtoms()
            asp = mol_bbox_aspect(mo)
            score = abs(heavy - 18) + asp * 1.5
            candidates.append((score, i, heavy, asp, row))

    if not candidates:
        raise RuntimeError("No tautomer-changed molecule found.")

    candidates = sorted(candidates, key=lambda x: x[0])

    print("\n=== compact candidates ===")
    for score, i, heavy, asp, row in candidates[:top_k_print]:
        print(
            f"valid_idx={i:6d} source_row={int(row['source_row']):6d} "
            f"rt={float(row['rt']):8.2f} heavy={heavy:2d} aspect={asp:.2f} "
            f"origin={row['origin_can']} taut={row['taut_can']}"
        )

    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin_csv", default="data/SMRT_train.csv")
    ap.add_argument("--taut_csv", default="data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv")
    ap.add_argument("--out_dir", default="figure_assets/panelA_compact_white")

    ap.add_argument("--rt_min", type=float, default=300.0)
    ap.add_argument("--row_index", type=int, default=-1)

    ap.add_argument("--min_heavy", type=int, default=10)
    ap.add_argument("--max_heavy", type=int, default=28)
    ap.add_argument("--max_aspect", type=float, default=2.2)

    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height", type=int, default=360)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.origin_csv, args.taut_csv, args.rt_min)

    if args.row_index >= 0:
        row = df.iloc[int(args.row_index)]
        valid_idx = int(args.row_index)
        mo = mol_from_smiles(row["origin_smile"])
        mt = mol_from_smiles(row["taut_smile"])
        heavy = mo.GetNumHeavyAtoms()
        aspect = mol_bbox_aspect(mo)
    else:
        _, valid_idx, heavy, aspect, row = select_compact_changed(
            df,
            min_heavy=args.min_heavy,
            max_heavy=args.max_heavy,
            max_aspect=args.max_aspect,
        )
        mo = mol_from_smiles(row["origin_smile"])
        mt = mol_from_smiles(row["taut_smile"])

    print("\n=== selected ===")
    print("valid_idx:", valid_idx)
    print("source_row:", int(row["source_row"]))
    print("rt:", float(row["rt"]))
    print("heavy:", heavy)
    print("aspect:", aspect)
    print("origin_can:", row["origin_can"])
    print("taut_can:", row["taut_can"])
    print("formula_origin:", formula(mo))
    print("formula_taut:", formula(mt))

    # raw white
    raw_official = out_dir / "A0_official_raw_white.png"
    raw_origin = out_dir / "A1_original_raw_white.png"
    raw_taut = out_dir / "A2_tautomer_raw_white.png"

    draw_mol_white_png(mo, raw_official, args.width, args.height)
    draw_mol_white_png(mo, raw_origin, args.width, args.height)
    draw_mol_white_png(mt, raw_taut, args.width, args.height)

    draw_mol_white_svg(mo, out_dir / "A0_official_raw_white.svg", args.width, args.height)
    draw_mol_white_svg(mo, out_dir / "A1_original_raw_white.svg", args.width, args.height)
    draw_mol_white_svg(mt, out_dir / "A2_tautomer_raw_white.svg", args.width, args.height)

    # cropped raw white
    crop_official = out_dir / "A0_official_cropped_white.png"
    crop_origin = out_dir / "A1_original_cropped_white.png"
    crop_taut = out_dir / "A2_tautomer_cropped_white.png"

    crop_white_margin(raw_official, crop_official, margin=24)
    crop_white_margin(raw_origin, crop_origin, margin=24)
    crop_white_margin(raw_taut, crop_taut, margin=24)

    # labeled cards
    card_official = out_dir / "A0_official_SMRT_molecule_card.png"
    card_origin = out_dir / "A1_original_SMRT_graph_card.png"
    card_taut = out_dir / "A2_strict_tautomer_canonical_graph_card.png"

    add_label_card(
        crop_official,
        card_official,
        "Official SMRT molecule",
        "same compound, same RT label",
        "#666666",
    )
    add_label_card(
        crop_origin,
        card_origin,
        "Original SMRT graph",
        "benchmark-provided representation",
        "#5B8DB8",
    )
    add_label_card(
        crop_taut,
        card_taut,
        "Strict tautomer canonical graph",
        "deterministic tautomer-canonical representation",
        "#55A79D",
    )

    make_triplet(
        [card_official, card_origin, card_taut],
        out_dir / "panelA_triplet_white_compact.png",
    )

    meta = pd.DataFrame([{
        "valid_idx": int(valid_idx),
        "source_row": int(row["source_row"]),
        "rt": float(row["rt"]),
        "heavy_atoms": int(heavy),
        "aspect": float(aspect),
        "origin_smile": row["origin_smile"],
        "taut_smile": row["taut_smile"],
        "origin_can": row["origin_can"],
        "taut_can": row["taut_can"],
        "formula_origin": formula(mo),
        "formula_taut": formula(mt),
        "taut_changed": int(row["taut_changed"]),
    }])
    meta.to_csv(out_dir / "panelA_selected_metadata.csv", index=False)

    print("\n=== saved ===")
    for p in [
        card_official,
        card_origin,
        card_taut,
        out_dir / "panelA_triplet_white_compact.png",
        crop_origin,
        crop_taut,
        out_dir / "A1_original_raw_white.svg",
        out_dir / "A2_tautomer_raw_white.svg",
        out_dir / "panelA_selected_metadata.csv",
    ]:
        print(p)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
