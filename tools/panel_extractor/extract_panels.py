import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


# ----------------------------
# Path / Repo Root utilities
# ----------------------------

def find_repo_root(marker_dirname: str = "vaterundsohnApp") -> Path:
    """
    Walk upwards from this file until a folder named `marker_dirname` is found.
    Returns that folder as repo root.
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if p.name == marker_dirname:
            return p
    raise RuntimeError(
        f"Konnte Repo-Root '{marker_dirname}' nicht finden. "
        f"Script liegt unter: {here}. Bitte Repo-Ordnername prüfen."
    )


# ----------------------------
# Image helpers
# ----------------------------

def normalize_bw_bgr(img_bgr, black_p=1.0, white_p=99.0, ref_low=None, ref_high=None):
    """
    Normalize a B/W scan so whites are consistent.
    Uses percentile-based black/white points (contrast stretch).

    If ref_low/ref_high are provided, they are used (recommended: compute once per page,
    apply to all panels so the whole comic matches).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if ref_low is None or ref_high is None:
        low = float(np.percentile(gray, black_p))
        high = float(np.percentile(gray, white_p))
    else:
        low, high = float(ref_low), float(ref_high)

    # Avoid division by zero / degenerate scans
    if high - low < 1e-6:
        return img_bgr
   
    # Contrast stretch
    g = gray.astype(np.float32)
    g = (g - low) * (255.0 / (high - low))
    g = np.clip(g, 0, 255).astype(np.uint8)

    # Convert back to BGR so the rest of your pipeline stays unchanged
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def cv_imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    """
    Robust image loader for Windows paths containing umlauts / unicode.
    Uses np.fromfile + cv2.imdecode instead of cv2.imread.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def sort_reading_order(boxes):
    """Sort boxes in reading order: row by row, left to right."""
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows = []
    for b in boxes:
        x, y, w, h = b
        cy = y + h / 2.0
        placed = False
        for row in rows:
            if abs(row["cy"] - cy) < row["h_med"] * 0.7:
                row["boxes"].append(b)
                row["cy"] = float(np.mean([bb[1] + bb[3] / 2.0 for bb in row["boxes"]]))
                row["h_med"] = float(np.median([bb[3] for bb in row["boxes"]]))
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "h_med": float(h), "boxes": [b]})

    for row in rows:
        row["boxes"] = sorted(row["boxes"], key=lambda b: b[0])

    rows = sorted(rows, key=lambda r: r["cy"])
    out = []
    for r in rows:
        out.extend(r["boxes"])
    return out


def compute_safe_padding(boxes, base_pad=14):
    """Compute per-box padding that tries not to intrude into neighboring panels."""
    pads = []
    for i, (x, y, w, h) in enumerate(boxes):
        rgaps, lgaps, bgaps, tgaps = [], [], [], []
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if j == i:
                continue
            v_overlap = not (y + h < y2 or y2 + h2 < y)
            h_overlap = not (x + w < x2 or x2 + w2 < x)

            if v_overlap and x2 >= x + w:
                rgaps.append(x2 - (x + w))
            if v_overlap and x2 + w2 <= x:
                lgaps.append(x - (x2 + w2))
            if h_overlap and y2 >= y + h:
                bgaps.append(y2 - (y + h))
            if h_overlap and y2 + h2 <= y:
                tgaps.append(y - (y2 + h2))

        def min_gap(gaps, fallback=10**9):
            return min(gaps) if gaps else fallback

        right_gap = min_gap(rgaps)
        left_gap = min_gap(lgaps)
        bot_gap = min_gap(bgaps)
        top_gap = min_gap(tgaps)

        def safe(g):
            if g >= 10**8:
                return base_pad
            return max(0, min(base_pad, int(g / 2) - 2))

        pads.append((safe(left_gap), safe(top_gap), safe(right_gap), safe(bot_gap)))
    return pads


def save_webp_bgr(img_bgr, out_path, quality=92):
    """Save a BGR OpenCV image as WEBP."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.save(out_path, format="WEBP", quality=quality, method=6)


def save_cover_from_panel_bgr(panel_bgr, out_path, cover_width=420, quality=90):
    """
    Create a cover.webp from a panel (BGR image).
    Downscale to cover_width keeping aspect ratio.
    """
    ch, cw = panel_bgr.shape[:2]
    if cw <= 0 or ch <= 0:
        return

    if cw != cover_width:
        scale = cover_width / cw
        new_h = max(1, int(round(ch * scale)))
        panel_bgr = cv2.resize(panel_bgr, (cover_width, new_h), interpolation=cv2.INTER_AREA)

    save_webp_bgr(panel_bgr, out_path, quality=quality)


def detect_panel_boxes(
    img_bgr,
    base_pad=14,
    min_area_ratio=0.02,
    max_area_ratio=0.40,
    min_aspect=0.6,
    max_aspect=6.0,
):
    """Return sorted panel boxes and safe paddings. boxes are (x,y,w,h)."""
    H, W = img_bgr.shape[:2]
    page_area = W * H

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Filter: neighboring page strip (very tall & relatively narrow)
        if h > 0.75 * H and w < 0.35 * W:
            continue

        r = (w * h) / page_area
        if r < min_area_ratio or r > max_area_ratio:
            continue

        aspect = w / max(1, h)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        boxes.append((x, y, w, h))

    boxes = sort_reading_order(boxes) if boxes else []
    pads = compute_safe_padding(boxes, base_pad=base_pad) if boxes else []
    return boxes, pads


def extract_one_to_vus_folder(
    in_path: str,
    vus_id: str,
    out_root: str,
    target_width: int = 380,
    base_pad: int = 14,
    debug: bool = True,
    normalize: bool = True,
):
    """
    Write to out_root/<vus_id>/:
      000.webp, 001.webp, ...
      cover.webp
      comic.json
      optional _debug_boxes.webp
    Returns dict entry for index.json + panel filenames list.
    """
    img = cv_imread_unicode(in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Kann Datei nicht lesen: {in_path}")
    # Compute normalization reference once per page (so all panels match)
    gray_page = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref_low = float(np.percentile(gray_page, 1.0))
    ref_high = float(np.percentile(gray_page, 99.0))

    # Safety guard
    if ref_high <= ref_low:
        ref_low, ref_high = 0.0, 255.0

    title = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(out_root, vus_id)
    os.makedirs(out_dir, exist_ok=True)

    boxes, pads = detect_panel_boxes(img, base_pad=base_pad)

    # Debug output: draw boxes and indices
    if debug:
        dbg = img.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(
                dbg,
                f"{i:03d}",
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        save_webp_bgr(dbg, os.path.join(out_dir, "_debug_boxes.webp"), quality=90)

    panel_files = []
    H, W = img.shape[:2]

    # Ensure compact numbering (000..N) even if something gets skipped
    out_idx = 0

    # Keep the first (resized) panel as cover source
    cover_source_bgr = None

    for ((x, y, w, h), (pl, pt, pr, pb)) in zip(boxes, pads):
        x1 = max(0, x - pl)
        y1 = max(0, y - pt)
        x2 = min(W, x + w + pr)
        y2 = min(H, y + h + pb)

        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if cw <= 0 or ch <= 0:
            continue

        # Resize to target width, keep aspect ratio
        scale = target_width / cw
        new_h = max(1, int(round(ch * scale)))
        crop_resized = cv2.resize(crop, (target_width, new_h), interpolation=cv2.INTER_AREA)
        if normalize:
            crop_resized = normalize_bw_bgr(
                crop_resized,
                ref_low=ref_low,
                ref_high=ref_high
            )

        fname = f"{out_idx:03d}.webp"
        save_webp_bgr(crop_resized, os.path.join(out_dir, fname), quality=92)
        panel_files.append(fname)

        if out_idx == 0:
            cover_source_bgr = crop_resized.copy()

        out_idx += 1

    # Always write comic.json (even if 0 panels, so the viewer can show a message)
    comic_json = {"title": title, "panels": panel_files}
    with open(os.path.join(out_dir, "comic.json"), "w", encoding="utf-8") as f:
        json.dump(comic_json, f, ensure_ascii=False, indent=2)

    # Write cover.webp if we have at least one panel
    if cover_source_bgr is not None:
        save_cover_from_panel_bgr(
            cover_source_bgr,
            os.path.join(out_dir, "cover.webp"),
            cover_width=420,
            quality=90,
        )

    # Build index entry (keep existing path style for your website)
    index_entry = {
        "id": vus_id,
        "title": title,
        "cover": f"comics/{vus_id}/cover.webp",
        "tags": ["dummy_tag_1", "dummy_tag_2"],
    }

    return index_entry, panel_files


def batch_to_vus(in_dir: str, out_dir: str, debug: bool = True, start_index: int = 1):
    """
    For each input image (sorted), create vus_XXX folders sequentially.
    Creates out_dir/index.json with one entry per folder.
    Continues on errors instead of aborting the whole batch.
    """
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
    files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(exts)])

    if not files:
        print(f"Keine Bilder gefunden in: {in_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    index_entries = []
    vus_counter = start_index

    failed = []

    for f in files:
        in_path = os.path.join(in_dir, f)
        vus_id = f"vus_{vus_counter:03d}"

        try:
            entry, panels = extract_one_to_vus_folder(
                in_path=in_path,
                vus_id=vus_id,
                out_root=out_dir,
                debug=debug,
            )
            print(f"{f} -> {vus_id}: {len(panels)} Panels")
            index_entries.append(entry)
        except Exception as e:
            print(f"FEHLER bei {f}: {e}")
            failed.append({"file": f, "error": str(e)})

        vus_counter += 1

    # Write index.json
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index_entries, f, ensure_ascii=False, indent=2)

    # Write failures log (optional but handy)
    if failed:
        with open(os.path.join(out_dir, "failed.json"), "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)

    print(f"Fertig. index.json geschrieben mit {len(index_entries)} Einträgen.")
    if failed:
        print(f"Achtung: {len(failed)} Dateien konnten nicht verarbeitet werden. Siehe failed.json.")


if __name__ == "__main__":
    import argparse

    repo_root = find_repo_root("vaterundsohnApp")
    default_in = repo_root / "tools" / "input_scans"
    default_out = repo_root / "tools" / "Vater_und_Sohn_Bildstrecke" / "output"

    ap = argparse.ArgumentParser(
        description="Extract panels and create vus_XXX folders + comic.json + index.json"
    )
    ap.add_argument("--in_dir", default=str(default_in), help="Input folder with scans/photos")
    ap.add_argument("--out_dir", default=str(default_out), help="Output root folder (will contain vus_XXX folders)")
    ap.add_argument("--no_debug", action="store_true", help="Disable debug output")
    ap.add_argument("--start", type=int, default=1, help="Start index for vus_XXX (default 1)")
    
    args = ap.parse_args()

    print(f"Repo-Root: {repo_root}")
    print(f"Input:     {args.in_dir}")
    print(f"Output:    {args.out_dir}")

    batch_to_vus(args.in_dir, args.out_dir, debug=(not args.no_debug), start_index=args.start)
