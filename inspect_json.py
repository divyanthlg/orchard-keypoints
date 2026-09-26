"""
Run this FIRST whenever you change your Hasty schema, before touching any
other code.

    python inspect_json.py /path/to/orchard_keypoints.json

Prints the top-level keys, the label_classes list, and one full image
entry (with its labels) so you can see exactly how Hasty nested the
keypoint coordinates and confirm the class_name strings match
CLASS_NAME_ALIASES in src/config.py (especially vanishing_point_right).
"""
import json
import sys


def main(path):
    with open(path, "r") as f:
        data = json.load(f)

    print("=" * 70)
    print("TOP-LEVEL KEYS:", list(data.keys()))
    print("=" * 70)

    if "label_classes" in data:
        print("\nLABEL CLASSES:")
        print(json.dumps(data["label_classes"], indent=2))

    images = data.get("images", [])
    print(f"\nNumber of images in export: {len(images)}")

    if not images:
        print("No images found under data['images'] - check the top-level "
              "keys printed above, the schema may differ from what this "
              "script assumes.")
        return

    example = None
    for img in images:
        if img.get("labels"):
            example = img
            break
    example = example or images[0]

    print("\n" + "=" * 70)
    print("ONE FULL IMAGE ENTRY (image_name, width/height, and its labels):")
    print("=" * 70)
    print(json.dumps(dict(example), indent=2)[:6000])
    print("\n(truncated if longer than 6000 chars)")

    print("\n" + "=" * 70)
    print("Confirm the class_name values above match src/config.py's "
          "CLASS_NAME_ALIASES exactly -- especially 'vanishing_point_right' "
          "if you named it something else in Hasty.")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_json.py /path/to/orchard_keypoints.json")
        sys.exit(1)
    main(sys.argv[1])
