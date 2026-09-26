# Orchard row-following keypoint detector

Trains a CNN to detect 4 keypoints per image, then estimates robot state
(heading angle, distance to left row, distance to right row) using a
calibrated camera-geometry pipeline based on CropFollow++ (Ch. 4 of the
Veeranampalayam Sivakumar thesis).

## Keypoints

```
vanishing_point        (LV) -- single instance, anchors the LEFT row line
vanishing_point_right   (RV) -- single instance, anchors the RIGHT row line
left_intercept          (LI) -- single instance, where the left row line exits the image
right_intercept         (RI) -- single instance, where the right row line exits the image
```

## Model

ResNet-18 encoder (ImageNet pretrained) + U-Net-style decoder (bilinear
upsampling, skip connections) producing 4 heatmap channels at 1/4 input
resolution. Trained with KL-divergence between a spatial-softmax of the
predicted heatmap and a normalized target Gaussian (matches CropFollow++'s
training recipe). Decoded with spatial soft-argmax.

## State estimation pipeline

Rather than using the 4 raw keypoints directly, the pipeline:

1. Fits **LLS** (line through `left_intercept` and `vanishing_point`) and
   **RLS** (line through `right_intercept` and `vanishing_point_right`).
2. Computes **AVP** ("actual vanishing point") as the intersection of LLS
   and RLS -- a geometrically-derived single vanishing point, since pitch
   and heading are properties of the one physical camera, not one value
   per row line.
3. Computes **ALVP**/**ARVP** by tracing the ray from AVP through
   `left_intercept`/`right_intercept` and finding where it exits the image
   rectangle. Used only as a validity check: LLS must exit via the left or
   bottom edge, RLS via the right or bottom edge. If not, the frame is
   flagged `[could not recognize]` and skipped.
4. AVP drives a roll -> pitch -> heading homography correction sequence
   (Eq. 22-23), then the corrected LLS/RLS lines are projected onto the
   bottom row to compute the distance ratio `d = d_L / (d_L + d_R)`
   (Eq. 24), converted to metric distances via a known row spacing.

See `src/camera_geometry.py` for the full implementation and derivation
notes.

## REQUIRED before real-world use

This pipeline needs camera calibration to produce meaningful numbers.
Every one of these is a placeholder in `src/config.py` right now:

- `FOCAL_LENGTH_PX` -- get from a checkerboard calibration
  (`cv2.calibrateCamera`), not the lens spec sheet.
- `PRINCIPAL_POINT` -- also from that same calibration.
- `CAMERA_ROLL_DEG` -- should come from a live IMU reading per frame on a
  real robot, not a config constant.
- `ROW_SPACING_M` -- the known real-world row spacing for your field.

Running with the placeholders is fine for development/testing the
pipeline mechanics, but angle and distance outputs will not be accurate
until these are set properly. See the comments on each in `config.py` for
exactly what to do.

## Setup

```bash
git clone <this-repo-url>
cd orchard_keypoints_project
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check GPU availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Data

This repo does not include the dataset or trained checkpoints (see
`.gitignore`). You need:
- `orchard_images/` -- your image folder
- `orchard_keypoints.json` -- your Hasty JSON v1.1 export with the 4
  keypoint classes above

Point `src/config.py`'s `DATA_ROOT` at the folder containing both.

## Usage

**0. Confirm your JSON schema** (do this whenever the schema changes):
```bash
python inspect_json.py /path/to/orchard_keypoints.json
```
Compare the printed `class_name` values against `CLASS_NAME_ALIASES` in
`src/config.py`.

**1. Sanity-check the data loader:**
```bash
python -m src.dataset
```
Saves `sanity_check_0/1/2.png` -- confirm the heatmaps land on the actual
row edges before training.

**2. Train:**
```bash
python -m src.train
```
Checkpoints save to `checkpoints/` (best + last). Per-epoch loss and a
per-channel breakdown print to stdout.

**3. Run inference on one image:**
```bash
python -m src.infer path/to/image.png --checkpoint checkpoints/best.pt
```
Prints keypoints, per-keypoint confidence, and the estimated robot state.
Saves `<image>_keypoints.png` (annotated detections) and
`<image>_steps.png` (a 5-panel step-by-step figure showing the roll/pitch/
heading correction process, styled after the thesis's Fig. 16).

**4. Debug a model that isn't detecting some classes:**
```bash
python debug_heatmaps.py path/to/image.png --checkpoint checkpoints/best.pt
```
Shows raw per-channel confidence before any threshold, plus saves a
heatmap overlay per class.

**5. Batch-process a whole folder:**
```bash
python -m src.batch_infer /path/to/images_folder
```
Processes every image in the folder, saving annotated copies plus a
`batch_results.csv` (per-image keypoints, confidence, and state) into a
new sibling folder. Use `--output` to control the destination.

## Files

```
inspect_json.py          prints your JSON's real structure
debug_heatmaps.py        per-channel raw confidence + heatmap overlays for one image
requirements.txt

src/config.py             paths, hyperparameters, class names, camera calibration constants
src/hasty_parser.py       parses Hasty JSON -> per-image list of (class, x, y)
src/dataset.py            PyTorch Dataset: image + 4-channel gaussian heatmaps
src/model.py              ResNet18 encoder + U-Net decoder + heatmap head
src/losses.py             KL-divergence heatmap loss (spatial softmax)
src/decode.py             spatial soft-argmax decoding
src/confidence.py         per-keypoint confidence check (CropFollow++ Eq. 25) + calibration helpers
src/camera_geometry.py    AVP/ALVP/ARVP + roll/pitch/heading/distance-ratio pipeline (Eq. 22-24)
src/visualize_steps.py    Fig.16-style step-by-step derivation figure
src/train.py              training loop
src/infer.py              single-image inference + state estimation + visualization
src/batch_infer.py        folder-level batch inference + CSV summary
```

## Known limitations / TODO

- Camera calibration constants are placeholders (see "REQUIRED" section above).
- `CAMERA_ROLL_DEG` is a static config value; needs to become a live IMU
  reading in an actual robot control loop.
- The confidence formula in `src/confidence.py` (Eq. 25 from the thesis)
  is close to a step function at realistic pixel-variance scales -- see
  that file's docstring for a calibrated alternative if you need a
  smoothly-graded confidence score (e.g. for sensor fusion).
- Trained on a small dataset relative to the original thesis (check
  `orchard_keypoints.json`'s image count); expect to need more annotated
  data for robust generalization.
