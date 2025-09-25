#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_coco_tf_record.py
Minimal COCO -> TFRecord converter for TF2 Object Detection API.

Usage:
  python create_coco_tf_record.py \
    --images_dir data/coco/images/train2017 \
    --annotations_json data/coco/annotations/train_person.json \
    --label_map_out annotations/label_map.pbtxt \
    --output_path annotations/train.record \
    --include_masks False \
    --filter_categories person \
    --num_shards 1 \
    --drop_crowd True \
    --min_area 0

Notes:
- If you pass --filter_categories, only those classes are kept.
- If you already filtered JSON to one class (e.g., person), you can omit --filter_categories.
- Writes/updates a label_map.pbtxt containing only kept categories (id/name preserved).
"""

import argparse
import io
import json
import os
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf

# ---------- TF Example helpers ----------
def _bytes_feature(v: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def _bytes_list_feature(vs: List[bytes]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=vs))

def _int64_feature(v: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

def _int64_list_feature(vs: List[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=vs))

def _float_list_feature(vs: List[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=vs))


# ---------- Core conversion ----------
def load_coco(annotations_json: str) -> Dict:
    with open(annotations_json, "r", encoding="utf-8") as f:
        return json.load(f)

def build_indices(coco: Dict):
    images_index = {im["id"]: im for im in coco["images"]}
    anns_by_image: Dict[int, List[Dict]] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    cat_index = {c["id"]: c for c in coco["categories"]}
    return images_index, anns_by_image, cat_index

def maybe_filter_categories(coco: Dict, keep_names: List[str]) -> Dict:
    if not keep_names:
        return coco
    keep_names_set = set(keep_names)
    keep_cat_ids = {c["id"] for c in coco["categories"] if c["name"] in keep_names_set}
    if not keep_cat_ids:
        raise ValueError(f"No categories matched filter {keep_names}")

    # filter annotations
    anns = [a for a in coco["annotations"] if a["category_id"] in keep_cat_ids]
    # keep only images that still have annotations
    img_ids_with_anns = {a["image_id"] for a in anns}
    imgs = [im for im in coco["images"] if im["id"] in img_ids_with_anns]
    cats = [c for c in coco["categories"] if c["id"] in keep_cat_ids]
    return {"images": imgs, "annotations": anns, "categories": cats}

def write_label_map(label_map_out: str, categories: List[Dict]):
    Path(label_map_out).parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_out, "w", encoding="utf-8") as f:
        for c in categories:
            f.write(f"item {{ id: {c['id']} name: '{c['name']}' }}\n")

def image_bytes_and_sha256(img_path: Path) -> Tuple[bytes, str]:
    with tf.io.gfile.GFile(str(img_path), "rb") as fid:
        encoded = fid.read()
    key = hashlib.sha256(encoded).hexdigest()
    return encoded, key

def coco_bbox_to_yxyx_norm(x, y, w, h, img_w, img_h):
    # COCO bbox is [x_min, y_min, width, height] in absolute pixels
    xmin = max(0.0, x) / img_w
    ymin = max(0.0, y) / img_h
    xmax = min(float(img_w), x + w) / img_w
    ymax = min(float(img_h), y + h) / img_h
    # Clamp
    xmin = min(max(xmin, 0.0), 1.0)
    ymin = min(max(ymin, 0.0), 1.0)
    xmax = min(max(xmax, 0.0), 1.0)
    ymax = min(max(ymax, 0.0), 1.0)
    return ymin, xmin, ymax, xmax  # TF OD API expects normalized y/x order lists

def make_tf_example(
    im: Dict,
    anns: List[Dict],
    cat_index: Dict[int, Dict],
    images_dir: Path,
    include_masks: bool = False,
    drop_crowd: bool = True,
    min_area: float = 0.0,
) -> tf.train.Example:
    file_name = im.get("file_name")
    img_w = int(im["width"])
    img_h = int(im["height"])
    img_path = images_dir / file_name

    encoded, key = image_bytes_and_sha256(img_path)

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []
    is_crowd_list = []
    area_list = []
    masks_encoded = []

    for a in anns:
        if drop_crowd and int(a.get("iscrowd", 0)) == 1:
            continue
        if float(a.get("area", 1e9)) < float(min_area):
            continue

        cat = cat_index[a["category_id"]]
        y1, x1, y2, x2 = coco_bbox_to_yxyx_norm(*a["bbox"], img_w, img_h)

        # Skip degenerate boxes
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            continue

        xmins.append(x1)
        xmaxs.append(x2)
        ymins.append(y1)
        ymaxs.append(y2)
        classes_text.append(cat["name"].encode("utf8"))
        classes.append(int(cat["id"]))
        is_crowd_list.append(int(a.get("iscrowd", 0)))
        area_list.append(float(a.get("area", 0.0)))

        if include_masks and "segmentation" in a and isinstance(a["segmentation"], list):
            # Optional: encode polygon masks as RLE or raw PNG. For simplicity we skip here.
            pass

    # If no boxes survived, return None to signal skip
    if len(classes) == 0:
        return None

    feature = {
        "image/height": _int64_feature(img_h),
        "image/width": _int64_feature(img_w),
        "image/filename": _bytes_feature(file_name.encode("utf8")),
        "image/source_id": _bytes_feature(str(im["id"]).encode("utf8")),
        "image/key/sha256": _bytes_feature(key.encode("utf8")),
        "image/encoded": _bytes_feature(encoded),
        "image/format": _bytes_feature(file_name.split(".")[-1].encode("utf8")),

        "image/object/bbox/xmin": _float_list_feature(xmins),
        "image/object/bbox/xmax": _float_list_feature(xmaxs),
        "image/object/bbox/ymin": _float_list_feature(ymins),
        "image/object/bbox/ymax": _float_list_feature(ymaxs),
        "image/object/class/text": _bytes_list_feature(classes_text),
        "image/object/class/label": _int64_list_feature(classes),
        "image/object/is_crowd": _int64_list_feature(is_crowd_list),
        "image/object/area": _float_list_feature(area_list),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def shard_writers(output_path: str, num_shards: int):
    base = Path(output_path)
    if num_shards <= 1:
        return [tf.io.TFRecordWriter(str(base))]
    writers = []
    for i in range(num_shards):
        shard_path = f"{base.stem}-{i:05d}-of-{num_shards:05d}{base.suffix or '.record'}"
        writers.append(tf.io.TFRecordWriter(str(base.with_name(shard_path))))
    return writers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Directory with images referenced by COCO JSON")
    ap.add_argument("--annotations_json", required=True, help="Path to COCO annotations JSON")
    ap.add_argument("--output_path", required=True, help="Output TFRecord path (or prefix if sharded)")
    ap.add_argument("--label_map_out", required=True, help="Path to write label_map.pbtxt")
    ap.add_argument("--include_masks", type=lambda s: s.lower()=="true", default=False)
    ap.add_argument("--filter_categories", type=str, default="", help="Comma-separated class names to keep (optional)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--drop_crowd", type=lambda s: s.lower()=="true", default=True)
    ap.add_argument("--min_area", type=float, default=0.0)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    coco = load_coco(args.annotations_json)

    keep = [s.strip() for s in args.filter_categories.split(",") if s.strip()]
    if keep:
        coco = maybe_filter_categories(coco, keep)

    images_index, anns_by_image, cat_index = build_indices(coco)
    # Persist label map containing only kept categories
    write_label_map(args.label_map_out, list(cat_index.values()))

    img_ids = list(images_index.keys())
    random.shuffle(img_ids)

    writers = shard_writers(args.output_path, args.num_shards)
    n_written = 0
    n_skipped = 0

    for idx, img_id in enumerate(img_ids):
        im = images_index[img_id]
        anns = anns_by_image.get(img_id, [])
        tf_ex = make_tf_example(
            im, anns, cat_index, images_dir,
            include_masks=args.include_masks,
            drop_crowd=args.drop_crowd,
            min_area=args.min_area,
        )
        if tf_ex is None:
            n_skipped += 1
            continue
        if args.num_shards > 1:
            shard_id = idx % args.num_shards
            writers[shard_id].write(tf_ex.SerializeToString())
        else:
            writers[0].write(tf_ex.SerializeToString())
        n_written += 1

    for w in writers:
        w.close()

    print(f"Done. Wrote {n_written} examples, skipped {n_skipped} (no valid boxes).")
    if args.num_shards > 1:
        print(f"Sharded into {args.num_shards} files with pattern based on: {args.output_path}")

if __name__ == "__main__":
    main()
