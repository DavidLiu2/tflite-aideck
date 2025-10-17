#!/usr/bin/env python3
import os, json, random, argparse, math
from pathlib import Path
from PIL import Image

def load_coco(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    # Map category name -> id
    cat_by_name = {c["name"]: c["id"] for c in coco.get("categories", [])}
    # Map image_id -> file_name, (width, height)
    img_by_id = {im["id"]: (im["file_name"], im.get("width"), im.get("height"))
                 for im in coco.get("images", [])}
    # Group annotations by image_id
    anns_by_img = {}
    for a in coco.get("annotations", []):
        iid = a["image_id"]
        anns_by_img.setdefault(iid, []).append(a)
    return cat_by_name, img_by_id, anns_by_img

def clamp(v, lo, hi): return max(lo, min(hi, v))

def pad_square_crop(x, y, w, h, img_w, img_h, pad=0.1):
    # make a square around the bbox with padding
    cx = x + w/2.0
    cy = y + h/2.0
    side = max(w, h) * (1.0 + pad*2.0)
    half = side / 2.0
    left   = clamp(int(math.floor(cx - half)), 0, img_w-1)
    top    = clamp(int(math.floor(cy - half)), 0, img_h-1)
    right  = clamp(int(math.ceil (cx + half)), 1, img_w)
    bottom = clamp(int(math.ceil (cy + half)), 1, img_h)
    # Adjust to square bounds if we got clipped
    side_w = right - left
    side_h = bottom - top
    side2 = min(max(side_w, side_h), min(img_w, img_h))
    right  = clamp(left + side2, 1, img_w)
    bottom = clamp(top + side2, 1, img_h)
    return left, top, right, bottom

def main():
    ap = argparse.ArgumentParser(description="Seed rep_images from COCO 'person' annotations.")
    ap.add_argument("--ann_json", required=True,
                    help="COCO annotations JSON (e.g., data/coco/annotations/train_person.json)")
    ap.add_argument("--images_dir", required=True,
                    help="Directory containing the COCO images referenced in JSON")
    ap.add_argument("--out_dir", default="../data/rep_images",
                    help="Output directory for grayscale 320x320 PNGs")
    ap.add_argument("--num_full", type=int, default=400,
                    help="How many full-frame images to export (max available)")
    ap.add_argument("--crops_per_image", type=int, default=0,
                    help="Optional: up to N crops per image around person boxes")
    ap.add_argument("--crop_pad", type=float, default=0.15,
                    help="Padding ratio around bbox when making square crops")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_by_name, img_by_id, anns_by_img = load_coco(args.ann_json)
    if "person" not in cat_by_name:
        raise SystemExit("No 'person' category found in annotations.")
    person_id = cat_by_name["person"]

    # Collect image_ids that have at least one person
    person_img_ids = [iid for iid, anns in anns_by_img.items()
                      if any(a.get("category_id") == person_id for a in anns)]
    random.shuffle(person_img_ids)

    # Take up to num_full images
    pick_img_ids = person_img_ids[:args.num_full]

    # Export full-frame + optional crops
    full_count = 0
    crop_count = 0
    for idx, iid in enumerate(pick_img_ids, start=1):
        file_name, img_w, img_h = img_by_id[iid]
        src_path = Path(args.images_dir) / file_name
        if not src_path.exists():
            # Some COCO subsets store images in nested folders; try basename fallback
            src_path = Path(args.images_dir) / os.path.basename(file_name)
            if not src_path.exists():
                print(f"[warn] missing image: {file_name}")
                continue

        try:
            im = Image.open(src_path).convert("L")  # grayscale
        except Exception as e:
            print(f"[warn] failed to open {src_path}: {e}")
            continue

        # FULL-FRAME
        im_full = im.resize((320, 320), Image.NEAREST)
        out_full = out_dir / f"rep_{idx:04d}.png"
        im_full.save(out_full)
        full_count += 1

        # CROPS (optional)
        if args.crops_per_image > 0:
            w, h = im.size
            # person bboxes for this image
            pboxes = [a["bbox"] for a in anns_by_img.get(iid, [])
                      if a.get("category_id") == person_id and "bbox" in a]
            random.shuffle(pboxes)
            for k, bbox in enumerate(pboxes[:args.crops_per_image], start=1):
                x, y, bw, bh = bbox  # COCO bbox is [x,y,w,h]
                left, top, right, bottom = pad_square_crop(x, y, bw, bh, w, h, pad=args.crop_pad)
                if right - left < 4 or bottom - top < 4:
                    continue
                crop = im.crop((left, top, right, bottom)).resize((320, 320), Image.NEAREST)
                out_crop = out_dir / f"rep_{idx:04d}_c{k}.png"
                crop.save(out_crop)
                crop_count += 1

    print(f"Done. Wrote {full_count} full-frame images and {crop_count} crops to: {out_dir}")

if __name__ == "__main__":
    main()
