import os

# ---------------------------------------------------------------------------
# Paths -- EDIT THESE
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.expanduser("~/projects/orchard_keypoints")
JSON_PATH = os.path.join(DATA_ROOT, "orchard_keypoints.json")
IMAGES_DIR = os.path.join(DATA_ROOT, "orchard_images")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical channel order. Every keypoint gets bucketed into one of these
# 4 channels. Index order matters -- it's the model's output channel order.
#
# Two separate vanishing points now, one per row line:
#   vanishing_point       -- anchors the LEFT row line (paired with left_intercept)
#   vanishing_point_right -- anchors the RIGHT row line (paired with right_intercept)
# This doesn't assume the two row lines share a single convergence point,
# which is more robust to curved rows / imperfect camera alignment.
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "vanishing_point",        # 0, single instance -- left row line
    "vanishing_point_right",  # 1, single instance -- right row line
    "left_intercept",         # 2, single instance
    "right_intercept",        # 3, single instance
]

# All four are single-instance -- decoded with spatial soft-argmax,
# trained with KL-divergence, same recipe as CropFollow++ (Ch.4).
SINGLE_INSTANCE_CLASSES = {
    "vanishing_point", "vanishing_point_right", "left_intercept", "right_intercept",
}

# Maps the class_name strings that actually appear in your Hasty JSON
# export to the canonical names used in CLASS_NAMES above. Run
# inspect_json.py on your updated export to confirm these match exactly
# (especially vanishing_point_right -- adjust here if you named the Hasty
# schema/class something slightly different).
CLASS_NAME_ALIASES = {
    "vanishing_point": "vanishing_point",
    "vanishing_point_right": "vanishing_point_right",
    "left_intercept": "left_intercept",
    "right_intercept": "right_intercept",
}

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Model / training hyperparameters (matches CropFollow++ Ch.4 recipe)
# ---------------------------------------------------------------------------
INPUT_SIZE = 512          # images are resized to INPUT_SIZE x INPUT_SIZE (square)
OUTPUT_STRIDE = 4         # heatmap resolution = INPUT_SIZE / OUTPUT_STRIDE
GAUSSIAN_SIGMA = 1.0      # sigma=1 in heatmap-space pixels

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5
VAL_FRACTION = 0.15
NUM_WORKERS = 4
SEED = 42

DEVICE = "cuda"  # falls back to cpu automatically in train.py/infer.py if unavailable

# ---------------------------------------------------------------------------
# Camera calibration for the CropFollow++ pipeline (Eq. 22-24), implemented
# in src/camera_geometry.py. These are PLACEHOLDERS for development on a
# laptop with no real robot/camera attached. On a real robot, EVERY ONE OF
# THESE must be replaced as described below -- the pipeline's angle and
# distance outputs scale directly with how correct these are.
# ---------------------------------------------------------------------------

# FOCAL_LENGTH_PX: focal length in PIXELS (not mm, not the lens spec sheet
# value). None makes infer.py/batch_infer.py fall back to using the image
# width as a rough guess (this corresponds to a made-up ~53deg horizontal
# FOV and is almost certainly wrong for your actual camera/lens).
#
# REAL ROBOT TODO: run a standard checkerboard calibration once per
# camera+lens combination using cv2.calibrateCamera (OpenCV's
# calibrateCamera tutorial), which returns a 3x3 intrinsic matrix K. Take
# fx (and fy, which should be close to fx for square pixels) from that
# matrix as FOCAL_LENGTH_PX. Re-calibrate if you change resolution, lens,
# or swap the physical camera.
FOCAL_LENGTH_PX = None

# PRINCIPAL_POINT: the (cx, cy) pixel where the optical axis actually hits
# the sensor. None makes the code assume it's exactly image center, which
# is a common approximation but rarely exactly true.
#
# REAL ROBOT TODO: also comes out of the same cv2.calibrateCamera call
# (the (0,2) and (1,2) entries of the intrinsic matrix K). Use that instead
# of guessing image center.
PRINCIPAL_POINT = None

# CAMERA_ROLL_DEG: the camera's roll angle relative to level ground, in
# degrees. 0.0 here means "assume the camera never rolls," which is only
# true if the robot chassis is perfectly rigid and the ground is flat.
#
# REAL ROBOT TODO: read this live, every frame, from the robot's IMU
# (typically the IMU's own roll estimate after its internal sensor fusion/
# filtering). Replace this constant with a live IMU topic/reading in your
# actual inference loop -- do not hardcode it once real hardware is
# involved, since roll will vary as the robot drives over uneven ground.
CAMERA_ROLL_DEG = 0.0

# ROW_SPACING_M: the known real-world perpendicular distance between crop
# rows, in meters. None means the pipeline only reports the dimensionless
# distance ratio d = d_L/(d_L+d_R), not an actual metric distance.
#
# REAL ROBOT TODO: measure this once for the field/orchard you're
# deploying in (it's an agronomic property of how the rows were planted,
# not something the camera can infer on its own), and set it here or load
# it from a per-field config at deployment time.
ROW_SPACING_M = None
