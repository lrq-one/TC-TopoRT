import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

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


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def mol_from_any(text, kind="auto"):
    text = str(text).strip()
    if text == "" or text.lower() == "nan":
        return None

    if kind == "inchi":
        return Chem.MolFromInchi(text)

    if kind == "smiles":
        return Chem.MolFromSmiles(text)

    # auto
    if text.startswith("InChI="):
        return Chem.MolFromInchi(text)
    return Chem.MolFromSmiles(text)


def mol_to_can(mol):
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def get_formula(mol):
    if mol is None:
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)


def prepare_mol(mol):
    mol = Chem.Mol(mol)
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass
    return mol


def draw_mol_png(mol, out_png, width=520, height=360):
    mol = prepare_mol(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.addAtomIndices = False
    opts.addStereoAnnotation = False
    opts.bondLineWidth = 2
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    Path(out_png).write_bytes(drawer.GetDrawingText())


def draw_mol_svg(mol, out_svg, width=520, height=360):
    mol = prepare_mol(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.addAtomIndices = False
    opts.addStereoAnnotation = False
    opts.bondLineWidth = 2
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    Path(out_svg).write_text(svg, encoding="utf-8")


def add_label_and_border(in_png, out_png, label, border_color, note=None):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(in_png).convert("RGBA")

    pad = 16
    label_h = 48
    note_h = 26 if note else 0
    border = 5

    W = img.width + pad * 2
    H = img.height + pad * 2 + label_h + note_h

    canvas = Image.new("RGBA", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_label = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font_note = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font_label = ImageFont.load_default()
        font_note = ImageFont.load_default()

    # border
    draw.rounded_rectangle(
        [2, 2, W - 3, H - 3],
        radius=16,
        outline=border_color,
        width=border,
        fill="white",
    )

    # label
    draw.text((W // 2, 18), label, anchor="mm", fill="#111111", font=font_label)

    # molecule
    canvas.alpha_composite(img, (pad, pad + label_h))

    if note:
        draw.text((W // 2, H - 18), note, anchor="mm", fill="#444444", font=font_note)

    canvas.convert("RGB").save(out_png, quality=95)


def make_triplet(paths, out_png):
    from PIL import Image, ImageDraw, ImageFont

    imgs = [Image.open(p).convert("RGB") for p in paths]
    gap = 30
    pad = 30
    W = sum(i.width for i in imgs) + gap * (len(imgs) - 1) + pad * 2
    H = max(i.height for i in imgs) + pad * 2

    canvas = Image.new("RGB", (W, H), "white")
    x = pad
    for img in imgs:
        y = pad + (H - 2 * pad - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width + gap

    canvas.save(out_png, quality=95)


def load_origin_and_taut(origin_csv, taut_csv=None, rt_min=300.0):
    origin = norm_cols(pd.read_csv(origin_csv, engine="python"))

    smi_col = find_col(origin, ["smile", "smiles", "canonical_smiles", "origin_smiles"])
    inchi_col = find_col(origin, ["inchi", "inchikey_inchi"])
    rt_col = find_col(origin, ["rt", "retention_time", "time"])

    if smi_col is None and inchi_col is None:
        raise ValueError(f"Cannot find SMILES/InChI column in {origin_csv}. columns={origin.columns.tolist()}")

    if rt_col is None:
        print("[WARN] no rt column found; using all rows")
        valid = origin.copy()
    else:
        origin[rt_col] = pd.to_numeric(origin[rt_col], errors="coerce")
        valid = origin[origin[rt_col] > float(rt_min)].copy()

    valid = valid.reset_index(drop=False).rename(columns={"index": "_source_row"})

    taut = None
    if taut_csv and os.path.exists(taut_csv):
        taut = norm_cols(pd.read_csv(taut_csv, engine="python"))
        taut_smi_col = find_col(taut, ["smile", "smiles", "taut_smile", "_taut_smile"])
        orig_smi_col = find_col(taut, ["orig_smile", "origin_smile", "_origin_smile"])
        if taut_smi_col is None:
            raise ValueError(f"Cannot find tautomer smile column in {taut_csv}. columns={taut.columns.tolist()}")

        # 常见情况：taut csv 是 rt>300 后、按 origin valid 顺序重排的 70182 行
        if len(taut) == len(valid):
            valid["_taut_smile"] = taut[taut_smi_col].astype(str).values
            if orig_smi_col is not None:
                valid["_taut_orig_smile"] = taut[orig_smi_col].astype(str).values
        # 另一种情况：taut csv 和 raw origin 一样长
        elif len(taut) == len(origin):
            idx = valid["_source_row"].values.astype(int)
            valid["_taut_smile"] = taut.iloc[idx][taut_smi_col].astype(str).values
            if orig_smi_col is not None:
                valid["_taut_orig_smile"] = taut.iloc[idx][orig_smi_col].astype(str).values
        else:
            raise ValueError(
                f"taut csv length mismatch: taut={len(taut)}, valid_origin={len(valid)}, raw_origin={len(origin)}"
            )
    else:
        print("[WARN] taut_csv not found. Will generate tautomer view using RDKit TautomerEnumerator fallback.")
        valid["_taut_smile"] = None

    return valid, smi_col, inchi_col, rt_col


def rdkit_canonical_tautomer_smiles(mol):
    from rdkit.Chem.MolStandardize import rdMolStandardize
    enum = rdMolStandardize.TautomerEnumerator()
    taut = enum.Canonicalize(mol)
    return Chem.MolToSmiles(taut, canonical=True, isomericSmiles=True)


def choose_example(valid, smi_col, inchi_col, prefer_changed=True, min_heavy=10, max_heavy=36, row_index=-1):
    rows = []

    for i, row in valid.iterrows():
        if row_index >= 0 and int(i) != int(row_index):
            continue

        src = row[smi_col] if smi_col is not None else row[inchi_col]
        src_kind = "smiles" if smi_col is not None else "inchi"
        mol_o = mol_from_any(src, kind=src_kind)
        if mol_o is None:
            continue

        heavy = mol_o.GetNumHeavyAtoms()
        if row_index < 0 and not (min_heavy <= heavy <= max_heavy):
            continue

        orig_can = mol_to_can(mol_o)

        taut_smi = row.get("_taut_smile", None)
        if taut_smi is None or str(taut_smi).lower() == "none" or str(taut_smi).lower() == "nan":
            try:
                taut_smi = rdkit_canonical_tautomer_smiles(mol_o)
            except Exception:
                taut_smi = orig_can

        mol_t = mol_from_any(taut_smi, kind="smiles")
        if mol_t is None:
            continue

        taut_can = mol_to_can(mol_t)
        changed = int(orig_can != taut_can)

        rows.append((i, row, mol_o, mol_t, orig_can, taut_can, heavy, changed))

        if row_index >= 0:
            break

    if not rows:
        raise RuntimeError("No drawable molecule found. Try --rt_min 0 or --row_index with a known row.")

    if prefer_changed:
        changed_rows = [r for r in rows if r[-1] == 1]
        if changed_rows:
            # 选一个大小适中的 changed 分子
            changed_rows = sorted(changed_rows, key=lambda x: abs(x[6] - 22))
            return changed_rows[0]

    rows = sorted(rows, key=lambda x: abs(x[6] - 22))
    return rows[0]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--origin_csv", default="data/SMRT_train.csv")
    ap.add_argument("--taut_csv", default="data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv")
    ap.add_argument("--out_dir", default="figure_assets/panelA_rdkit_views")

    ap.add_argument("--rt_min", type=float, default=300.0)
    ap.add_argument("--row_index", type=int, default=-1, help="valid-row index after rt filter; -1 means auto select")
    ap.add_argument("--prefer_changed", type=int, default=1)

    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height", type=int, default=360)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid, smi_col, inchi_col, rt_col = load_origin_and_taut(
        origin_csv=args.origin_csv,
        taut_csv=args.taut_csv,
        rt_min=args.rt_min,
    )

    idx, row, mol_o, mol_t, orig_can, taut_can, heavy, changed = choose_example(
        valid=valid,
        smi_col=smi_col,
        inchi_col=inchi_col,
        prefer_changed=bool(args.prefer_changed),
        row_index=args.row_index,
    )

    rt_value = row[rt_col] if rt_col is not None and rt_col in row else ""

    print("\n=== selected molecule ===")
    print("valid_index:", idx)
    print("source_row:", int(row["_source_row"]))
    print("rt:", rt_value)
    print("heavy_atoms:", heavy)
    print("taut_changed:", changed)
    print("origin_can:", orig_can)
    print("taut_can:", taut_can)
    print("formula_origin:", get_formula(mol_o))
    print("formula_taut:", get_formula(mol_t))

    # raw molecule drawings
    raw_official = out_dir / "A0_official_smrt_molecule_raw.png"
    raw_original = out_dir / "A1_original_smrt_view_raw.png"
    raw_taut = out_dir / "A2_strict_tautomer_canonical_view_raw.png"

    draw_mol_png(mol_o, raw_official, width=args.width, height=args.height)
    draw_mol_png(mol_o, raw_original, width=args.width, height=args.height)
    draw_mol_png(mol_t, raw_taut, width=args.width, height=args.height)

    draw_mol_svg(mol_o, out_dir / "A0_official_smrt_molecule_raw.svg", width=args.width, height=args.height)
    draw_mol_svg(mol_o, out_dir / "A1_original_smrt_view_raw.svg", width=args.width, height=args.height)
    draw_mol_svg(mol_t, out_dir / "A2_strict_tautomer_canonical_view_raw.svg", width=args.width, height=args.height)

    # labeled PNGs for directly placing into draw.io / AI
    official = out_dir / "A0_official_smrt_molecule.png"
    original = out_dir / "A1_original_smrt_view.png"
    taut = out_dir / "A2_strict_tautomer_canonical_view.png"

    try:
        add_label_and_border(
            raw_official,
            official,
            label="Official SMRT molecule",
            border_color="#666666",
            note="same compound, same RT label",
        )
        add_label_and_border(
            raw_original,
            original,
            label="Original SMRT view",
            border_color="#5B8DB8",
            note="raw benchmark representation",
        )
        add_label_and_border(
            raw_taut,
            taut,
            label="Strict tautomer canonical view",
            border_color="#55A79D",
            note="deterministic tautomer representation",
        )
        make_triplet([official, original, taut], out_dir / "panelA_three_images_triplet.png")
    except Exception as e:
        print("[WARN] PIL label/border generation failed:", repr(e))
        print("Raw PNG/SVG files were still generated.")

    meta = {
        "valid_index": int(idx),
        "source_row": int(row["_source_row"]),
        "rt": float(rt_value) if str(rt_value) not in ["", "nan"] else None,
        "heavy_atoms": int(heavy),
        "taut_changed": int(changed),
        "origin_canonical_smiles": orig_can,
        "tautomer_canonical_smiles": taut_can,
        "formula_origin": get_formula(mol_o),
        "formula_taut": get_formula(mol_t),
    }

    pd.DataFrame([meta]).to_csv(out_dir / "panelA_selected_molecule_metadata.csv", index=False)

    print("\n=== saved files ===")
    for p in [
        official,
        original,
        taut,
        out_dir / "panelA_three_images_triplet.png",
        out_dir / "A0_official_smrt_molecule_raw.svg",
        out_dir / "A1_original_smrt_view_raw.svg",
        out_dir / "A2_strict_tautomer_canonical_view_raw.svg",
        out_dir / "panelA_selected_molecule_metadata.csv",
    ]:
        print(p)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
