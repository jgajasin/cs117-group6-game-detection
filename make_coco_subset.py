"""Build the COCO val2017 real-photo baseline subset.

Picks images that contain people and images that contain horses/cows/sheep,
downloads them, and writes a COCO-format annotation file with just those images
(and just those 4 classes) for the eval script.

Usage:
    python make_coco_subset.py --n-person 40 --n-animal 35 --seed 117
Outputs:
    data/coco_subset/images/*.jpg
    data/coco_subset/annotations.json   (COCO format, 4 classes only)
    data/coco_subset/image_ids.txt      (commit this; it makes the subset reproducible)
"""
import argparse, io, json, os, random, urllib.request, zipfile

ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_MEMBER = "annotations/instances_val2017.json"
CLASSES = ["person", "horse", "cow", "sheep"]
ANIMALS = {"horse", "cow", "sheep"}


def load_val_annotations(cache):
    if not os.path.exists(cache):
        print("Downloading COCO annotations (~240 MB, one time)...")
        data = urllib.request.urlopen(ANN_ZIP_URL).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z, open(cache, "wb") as f:
            f.write(z.read(ANN_MEMBER))
    with open(cache) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-person", type=int, default=40)
    ap.add_argument("--n-animal", type=int, default=35)
    ap.add_argument("--seed", type=int, default=117)
    ap.add_argument("--out", default="data/coco_subset")
    ap.add_argument("--cache", default="instances_val2017.json")
    ap.add_argument("--no-download-images", action="store_true")
    a = ap.parse_args()

    coco = load_val_annotations(a.cache)
    cat_id = {c["name"]: c["id"] for c in coco["categories"] if c["name"] in CLASSES}
    name_of = {v: k for k, v in cat_id.items()}

    # which of our classes appear (non-crowd) in each image
    present = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in name_of and not ann.get("iscrowd", 0):
            present.setdefault(ann["image_id"], set()).add(name_of[ann["category_id"]])

    animal_imgs = sorted(i for i, s in present.items() if s & ANIMALS)
    person_only = sorted(i for i, s in present.items() if s == {"person"})

    rng = random.Random(a.seed)
    chosen_animal = rng.sample(animal_imgs, min(a.n_animal, len(animal_imgs)))
    chosen_person = rng.sample(person_only, min(a.n_person, len(person_only)))
    chosen = sorted(set(chosen_animal) | set(chosen_person))

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    keep = set(chosen)
    images = [im for im in coco["images"] if im["id"] in keep]
    subset = {
        "info": {"description": f"COCO val2017 subset, seed={a.seed}"},
        "licenses": coco.get("licenses", []),
        "categories": [c for c in coco["categories"] if c["name"] in CLASSES],
        "images": images,
        # keep iscrowd annotations too, so the eval can skip them instead of
        # counting them as misses
        "annotations": [x for x in coco["annotations"]
                        if x["image_id"] in keep and x["category_id"] in name_of],
    }
    with open(os.path.join(a.out, "annotations.json"), "w") as f:
        json.dump(subset, f)
    with open(os.path.join(a.out, "image_ids.txt"), "w") as f:
        f.write("\n".join(map(str, chosen)) + "\n")

    if not a.no_download_images:
        for k, im in enumerate(images, 1):
            dst = os.path.join(a.out, "images", im["file_name"])
            if not os.path.exists(dst):
                urllib.request.urlretrieve(im["coco_url"], dst)
            print(f"\r{k}/{len(images)} images", end="")
        print()

    counts = {c: 0 for c in CLASSES}
    for x in subset["annotations"]:
        if not x.get("iscrowd", 0):
            counts[name_of[x["category_id"]]] += 1
    print(f"{len(images)} images ({len(chosen_person)} people-only, "
          f"{len(chosen_animal)} with animals)")
    print("objects (non-crowd):", counts)


if __name__ == "__main__":
    main()
