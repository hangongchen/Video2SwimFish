"""Big table: one row per fish -> canonical frame crop | mesh side | mesh top | mesh+skeleton side | mesh+skeleton top.
One PNG per species (20 rows) + a combined PDF. Usage: python make_dataset_sheet.py [--out $EVAL_OUT/dataset_sheet]
Needs mesh renders $EVAL_OUT/dataset_sheet/mesh_views/<tag>_{side,top}.png (scripts/render_mesh_views.py) and
$ASSET_OUT/DATASET_REPORT.csv (scripts/dataset_report.py)."""
import argparse, glob, os, json, sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v2sf_paths as P
EVAL = str(P.EVAL_OUT)
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{EVAL}/dataset_sheet"); a = ap.parse_args()
CW, CH, LW, PAD = 300, 200, 190, 6
COLS = ["canonical frame (crop)", "mesh (lateral)", "mesh (dorsal)", "mesh + skeleton (dorsal)", "mesh + skeleton (lateral)"]
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15); FONTB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
except Exception: FONT = FONTB = ImageFont.load_default()
def fit(path, autocrop=False):
    tile = Image.new("RGB", (CW, CH), "white")
    if path and os.path.exists(path):
        im = Image.open(path).convert("RGB")
        if autocrop:   # tighten the uniform render background so the fish fills the tile
            import numpy as np
            arr = np.asarray(im).astype(int); bg = arr[0, 0]; mask = (np.abs(arr - bg).sum(-1) > 30)
            ys, xs = np.where(mask)
            if len(xs) > 10:
                m = 12; im = im.crop((max(0, xs.min() - m), max(0, ys.min() - m), min(im.width, xs.max() + m), min(im.height, ys.max() + m)))
        im.thumbnail((CW - 2 * PAD, CH - 2 * PAD))
        tile.paste(im, ((CW - im.width) // 2, (CH - im.height) // 2))
    else:
        ImageDraw.Draw(tile).text((CW // 2 - 30, CH // 2 - 8), "missing", fill="red", font=FONT)
    return tile
report = {}
try:
    import csv
    for r in csv.DictReader(open(f"{P.ASSET_OUT}/DATASET_REPORT.csv")): report[r["fish"]] = r
except Exception: pass
tags = sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(f"{P.ASSET_OUT}/*_fish*/"))
species = sorted({t.rsplit("_fish", 1)[0] for t in tags}); pages = []
for sp in species:
    rows = [t for t in tags if t.rsplit("_fish", 1)[0] == sp]
    HDR = 70; W = LW + CW * len(COLS); H = HDR + CH * len(rows)
    page = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(page)
    dr.text((PAD, 10), f"{sp}  ({len(rows)} fish)", fill="black", font=FONTB)
    for j, c in enumerate(COLS): dr.text((LW + j * CW + PAD, 44), c, fill="black", font=FONT)
    for i, t in enumerate(rows):
        y = HDR + i * CH; r = report.get(t, {})
        lab = [t.replace(sp + "_", ""), f"L={r.get('length_cm','?')} cm", f"bones={r.get('bones','?')}", f"cover={r.get('coverage','?')}", f"align={r.get('align_mean','?')} deg", f"contain={r.get('contain','?')}"]
        for k, s in enumerate(lab): dr.text((PAD, y + 8 + 26 * k), s, fill="black", font=FONTB if k == 0 else FONT)
        imgs = [f"{P.RAW_VIDEO_ROOT}/{sp}_canonical_frames_cropped/{t}_canonical.jpg",
                f"{EVAL}/dataset_sheet/mesh_views/{t}_side.png", f"{EVAL}/dataset_sheet/mesh_views/{t}_top.png",
                f"{P.ASSET_OUT}/{t}/_work/repair_side.png", f"{P.ASSET_OUT}/{t}/_work/repair_top.png"]
        for j, pth in enumerate(imgs): page.paste(fit(pth, autocrop=(j >= 1)), (LW + j * CW, y))
        dr.line([(0, y), (W, y)], fill=(200, 200, 200))
    os.makedirs(a.out, exist_ok=True); fn = f"{a.out}/sheet_{sp}.png"; page.save(fn, optimize=True); pages.append(page); print("wrote", fn)
pdf = f"{a.out}/dataset_sheet_all.pdf"; pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=100); print("wrote", pdf)
