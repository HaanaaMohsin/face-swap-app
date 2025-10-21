import cv2
import numpy as np
from typing import Tuple

# Initialize Haar cascade once to keep it lightweight at runtime
_SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


def _clamp_roi(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


essential_float_dtype = np.float32


def _extract_bbox(face_obj: object) -> Tuple[int, int, int, int]:
    """Extract integer bbox (x1, y1, x2, y2) from an InsightFace face object."""
    bbox = getattr(face_obj, 'bbox', None)
    if bbox is None:
        raise ValueError("Face object does not have 'bbox' attribute")
    bbox = np.array(bbox).astype(int)
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def is_smiling(image_rgb: np.ndarray, face_obj: object) -> bool:
    """
    Heuristic smile detection using OpenCV's Haar cascade on the face ROI.

    Parameters:
    - image_rgb: RGB or BGR image (H, W, 3). Works with either.
    - face_obj: InsightFace face object with .bbox

    Returns:
    - bool indicating whether a smile was detected
    """
    if image_rgb is None or image_rgb.size == 0:
        return False

    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = _extract_bbox(face_obj)
    x1, y1, x2, y2 = _clamp_roi(x1, y1, x2, y2, w, h)

    roi = image_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    # Convert to grayscale for Haar cascade regardless of input color space
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if image_rgb.shape[2] == 3 else roi

    min_size = max(16, int(min(roi_gray.shape[0], roi_gray.shape[1]) * 0.18))
    smiles = _SMILE_CASCADE.detectMultiScale(
        roi_gray,
        scaleFactor=1.3,
        minNeighbors=15,
        minSize=(min_size, min_size),
    )

    return len(smiles) > 0


def emotion_label(is_smile: bool) -> str:
    return "Happy" if is_smile else "Neutral"


def _make_rect_mask(shape: Tuple[int, int], rect: Tuple[int, int, int, int], sigma: int) -> np.ndarray:
    """Create a blurred rectangular mask within given image shape."""
    h, w = shape
    x1, y1, x2, y2 = rect
    mask = np.zeros((h, w), dtype=essential_float_dtype)
    mask[y1:y2, x1:x2] = 1.0
    # Ensure kernel size is odd and proportional to rect size
    k = max(3, int(max((y2 - y1), (x2 - x1)) * 0.15))
    if k % 2 == 0:
        k += 1
    k = max(k, sigma if sigma % 2 == 1 else sigma + 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _compute_expression_rects(face_bbox: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Derive approximate eyes and mouth rectangles from the face bbox.
    Returns (eyes_rect, mouth_rect) as (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = face_bbox
    fw = x2 - x1
    fh = y2 - y1

    # Eyes region: upper-middle portion of face
    eyes_x1 = int(x1 + 0.12 * fw)
    eyes_x2 = int(x1 + 0.88 * fw)
    eyes_y1 = int(y1 + 0.28 * fh)
    eyes_y2 = int(y1 + 0.50 * fh)

    # Mouth region: lower portion of face
    mouth_x1 = int(x1 + 0.18 * fw)
    mouth_x2 = int(x1 + 0.82 * fw)
    mouth_y1 = int(y1 + 0.62 * fh)
    mouth_y2 = int(y1 + 0.95 * fh)

    return (eyes_x1, eyes_y1, eyes_x2, eyes_y2), (mouth_x1, mouth_y1, mouth_x2, mouth_y2)


def blend_expression_regions(swapped_img: np.ndarray, target_img: np.ndarray, face_obj: object,
                             mouth_weight: float = 0.3, eyes_weight: float = 0.15) -> np.ndarray:
    """
    Blend target's eyes and mouth back onto the swapped image to better preserve emotion.

    Parameters:
    - swapped_img: swapped image (same color space as target_img)
    - target_img: original target image
    - face_obj: InsightFace face object located on target_img
    - mouth_weight: blending weight for mouth region [0..1]
    - eyes_weight: blending weight for eyes region [0..1]

    Returns:
    - blended image
    """
    if swapped_img is None or target_img is None:
        return swapped_img

    assert swapped_img.shape == target_img.shape, "Images must have the same shape for blending"

    h, w = target_img.shape[:2]
    x1, y1, x2, y2 = _extract_bbox(face_obj)
    x1, y1, x2, y2 = _clamp_roi(x1, y1, x2, y2, w, h)

    eyes_rect, mouth_rect = _compute_expression_rects((x1, y1, x2, y2))

    eyes_mask = _make_rect_mask((h, w), eyes_rect, sigma=9) * float(max(0.0, min(1.0, eyes_weight)))
    mouth_mask = _make_rect_mask((h, w), mouth_rect, sigma=9) * float(max(0.0, min(1.0, mouth_weight)))

    alpha = np.clip(eyes_mask + mouth_mask, 0.0, 1.0).astype(essential_float_dtype)

    if swapped_img.dtype != np.uint8:
        swapped = (swapped_img * 255.0).astype(np.uint8)
    else:
        swapped = swapped_img
    target = target_img

    alpha_3 = np.expand_dims(alpha, axis=2)
    blended = swapped.astype(essential_float_dtype) * (1.0 - alpha_3) + target.astype(essential_float_dtype) * alpha_3
    return np.clip(blended, 0, 255).astype(np.uint8)
