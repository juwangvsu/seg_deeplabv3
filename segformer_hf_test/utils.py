import numpy as np
# ============================
CITYSCAPES_CLASSES = 19
CITYSCAPES_IGNORE_INDEX = 255

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# labelId -> trainId mapping (vectorized). Unspecified ids map to 255.
_LABELID_TO_TRAINID = {
    0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255,
    7:0, 8:1, 9:255, 10:255,
    11:2, 12:3, 13:4, 14:255, 15:255, 16:255,
    17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15,
    29:255, 30:255, 31:16, 32:17, 33:18,
    -1:255  # sometimes unlabeled is -1
}

def map_labelIds_to_trainIds(arr: np.ndarray) -> np.ndarray:
    out = np.full_like(arr, CITYSCAPES_IGNORE_INDEX)
    for lid, tid in _LABELID_TO_TRAINID.items():
        out[arr == lid] = tid
    return out.astype(np.uint8)
