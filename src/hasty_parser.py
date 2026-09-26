"""
Parses a Hasty JSON v1.1 export into a simple, flat structure:

    images = [
        {
            "image_name": "IMG_0001.jpg",
            "width": 640,
            "height": 480,
            "points": {
                "vanishing_point": [(x, y)],
                "vanishing_point_right": [(x, y)],
                "left_intercept": [(x, y)],
                "right_intercept": [(x, y)],
            },
        },
        ...
    ]

Confirmed against a real Hasty export: labels each have a "keypoints" list
of {x, y, ...} dicts. If a future schema change breaks this, run
inspect_json.py and edit the 3 constants below to match.
"""
import json

from src.config import CLASS_NAME_ALIASES, CLASS_NAMES

# ---------------------------------------------------------------------------
# EDIT THESE THREE IF inspect_json.py SHOWS A DIFFERENT STRUCTURE
# ---------------------------------------------------------------------------
KEYPOINT_LIST_KEY = "keypoints"
POINT_X_KEY = "x"
POINT_Y_KEY = "y"
# ---------------------------------------------------------------------------


def _canonical_class(raw_class_name):
    if raw_class_name in CLASS_NAME_ALIASES:
        return CLASS_NAME_ALIASES[raw_class_name]
    lowered = raw_class_name.strip().lower()
    for name in CLASS_NAMES:
        if lowered == name.lower():
            return name
    return None


def load_hasty_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    parsed_images = []
    skipped_labels = set()

    for img in data.get("images", []):
        points = {name: [] for name in CLASS_NAMES}

        for label in img.get("labels", []):
            raw_class = label.get("class_name", "")
            canonical = _canonical_class(raw_class)
            if canonical is None:
                skipped_labels.add(raw_class)
                continue

            kp_list = label.get(KEYPOINT_LIST_KEY)
            if not kp_list:
                continue

            for pt in kp_list:
                x = pt.get(POINT_X_KEY)
                y = pt.get(POINT_Y_KEY)
                if x is None or y is None:
                    continue
                points[canonical].append((float(x), float(y)))

        parsed_images.append({
            "image_name": img["image_name"],
            "width": img["width"],
            "height": img["height"],
            "points": points,
        })

    if skipped_labels:
        print(f"[hasty_parser] Note: ignored {len(skipped_labels)} unmatched "
              f"class_name(s) not in CLASS_NAME_ALIASES: {sorted(skipped_labels)}. "
              f"If one of these should be a keypoint class, add it to "
              f"CLASS_NAME_ALIASES in src/config.py.")

    return parsed_images


if __name__ == "__main__":
    from src.config import JSON_PATH
    imgs = load_hasty_json(JSON_PATH)
    print(f"Parsed {len(imgs)} images.")
    n_with_all = sum(
        1 for im in imgs if all(len(im["points"][c]) > 0 for c in CLASS_NAMES)
    )
    print(f"Images with all {len(CLASS_NAMES)} keypoints present: {n_with_all}")
    if imgs:
        print("Example:", json.dumps(imgs[0], indent=2)[:1500])
