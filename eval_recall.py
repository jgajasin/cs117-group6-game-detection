"""Recall of a YOLO model on a COCO-format labeled image set.

Metric (same as proposal slide 1): a labeled object counts as found if YOLO
predicts a box of the SAME class with IoU >= 0.5, at the default confidence
threshold. Each prediction can match at most one labeled object. Crowd
annotations (iscrowd=1) are not counted. Reported separately for people and
animals (horse, cow, sheep pooled), plus per class.

Usage (COCO baseline):
    python eval_recall.py --images data/coco_subset/images \
        --ann data/coco_subset/annotations.json \
        --out results/baseline_coco_val.json
Works the same for game screenshots once they're exported from CVAT as COCO JSON.
"""
import argparse, json, os
from collections import defaultdict

CLASSES = ["person", "horse", "cow", "sheep"]
GROUPS = {"people": ["person"], "animals": ["horse", "cow", "sheep"]}


def iou(a, b):  # boxes as x1, y1, x2, y2
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gts, preds, thr):
    """gts: [(cls, box)], preds: [(cls, conf, box)] -> list of bools, one per gt.
    Greedy: highest-confidence prediction first, takes the best unmatched
    same-class gt with IoU >= thr."""
    found = [False] * len(gts)
    for cls, _, pbox in sorted(preds, key=lambda p: -p[1]):
        best, best_iou = None, thr
        for j, (gcls, gbox) in enumerate(gts):
            if found[j] or gcls != cls:
                continue
            v = iou(pbox, gbox)
            if v >= best_iou:
                best, best_iou = j, v
        if best is not None:
            found[best] = True
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)

    coco = json.load(open(a.ann))
    catname = {c["id"]: c["name"] for c in coco["categories"]}
    gts = defaultdict(list)
    for x in coco["annotations"]:
        name = catname.get(x["category_id"])
        if name in CLASSES and not x.get("iscrowd", 0):
            bx, by, bw, bh = x["bbox"]  # COCO bbox = x, y, width, height
            gts[x["image_id"]].append((name, [bx, by, bx + bw, by + bh]))

    found_n, total_n = defaultdict(int), defaultdict(int)
    per_image = []
    for im in coco["images"]:
        path = os.path.join(a.images, im["file_name"])
        r = model.predict(path, imgsz=a.imgsz, verbose=False)[0]  # default conf
        names = r.names
        preds = [(names[int(c)], float(s), b.tolist())
                 for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)
                 if names[int(c)] in CLASSES]
        g = gts.get(im["id"], [])
        f = match(g, preds, a.iou)
        for (cls, _), ok in zip(g, f):
            total_n[cls] += 1
            found_n[cls] += ok
        per_image.append({"file": im["file_name"], "labeled": len(g), "found": sum(f)})

    def rec(classes):
        t = sum(total_n[c] for c in classes)
        fnd = sum(found_n[c] for c in classes)
        return {"recall": round(fnd / t, 4) if t else None, "found": fnd, "labeled": t}

    result = {
        "model": a.model, "split": a.split, "num_images": len(coco["images"]),
        "metric": f"recall, same class, IoU>={a.iou}, default conf "
                  f"({model.predictor.args.conf}), imgsz={a.imgsz}",
        "groups": {g: rec(cs) for g, cs in GROUPS.items()},
        "per_class": {c: rec([c]) for c in CLASSES},
        "per_image": per_image,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
