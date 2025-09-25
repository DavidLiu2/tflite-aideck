# coco_person_filter.py
import json, sys

def filter_to_person(src_json, dst_json, drop_crowd=True, min_area=0):
    with open(src_json, 'r') as f:
        coco = json.load(f)

    person_cat_ids = {c['id'] for c in coco['categories'] if c['name'] == 'person'}
    anns = []
    img_has_person = set()

    for a in coco['annotations']:
        if a['category_id'] in person_cat_ids:
            if drop_crowd and a.get('iscrowd', 0) == 1:
                continue
            if a.get('area', 1e9) < min_area:
                continue
            anns.append(a)
            img_has_person.add(a['image_id'])

    imgs = [im for im in coco['images'] if im['id'] in img_has_person]
    cats = [c for c in coco['categories'] if c['name'] == 'person']

    out = dict(images=imgs, annotations=anns, categories=cats)
    with open(dst_json, 'w') as f:
        json.dump(out, f)
    print(f"Images kept: {len(imgs)}; Annotations kept: {len(anns)}; Categories: {len(cats)}")

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    # Example: python coco_person_filter.py instances_train2017.json train_person.json
    filter_to_person(src, dst, drop_crowd=True, min_area=0)
