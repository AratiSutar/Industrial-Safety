import cv2
import numpy as np

from safety.crops import MAX_SIDE, MIN_SIDE, fit_for_jev, head_region, ppe_crops
from safety.detector import L_EYE, R_EYE, Person


def person_with_eyes(eye_y=100.0):
    kp, kc = np.zeros((17, 2)), np.full(17, 0.1)
    kp[L_EYE], kp[R_EYE] = (140, eye_y), (160, eye_y)
    kc[L_EYE] = kc[R_EYE] = 0.9
    return Person(1, (100.0, 60.0, 200.0, 460.0), 0.9, kp, kc, (150, 460), True)


def test_small_crop_is_scaled_up_to_jev_minimum():
    out = fit_for_jev(np.zeros((40, 60, 3), np.uint8))
    assert min(out.shape[:2]) >= MIN_SIDE and max(out.shape[:2]) <= MAX_SIDE


def test_large_crop_is_scaled_down():
    out = fit_for_jev(np.zeros((3000, 2000, 3), np.uint8))
    assert max(out.shape[:2]) <= MAX_SIDE and min(out.shape[:2]) >= MIN_SIDE


def test_very_thin_crop_is_padded():
    out = fit_for_jev(np.zeros((20, 2000, 3), np.uint8))
    assert min(out.shape[:2]) >= MIN_SIDE and max(out.shape[:2]) <= MAX_SIDE


def test_hard_hat_crop_stops_above_the_eyes():
    _, _, _, bottom = head_region(person_with_eyes(eye_y=100), 0.5)
    assert bottom < 100


def test_hard_hat_crop_stays_above_eyebrows_when_nose_is_visible():
    from safety.detector import NOSE

    p = person_with_eyes(eye_y=100)
    p.keypoints[NOSE], p.keypoint_conf[NOSE] = (150, 120), 0.9   # eye-to-nose 20 px
    _, _, _, bottom = head_region(p, 0.5)
    assert bottom <= 100 - 0.6 * 20


def test_ppe_crops_are_valid_jpegs_within_limits():
    frame = np.random.default_rng(0).integers(0, 255, (500, 300, 3), dtype=np.uint8)
    crops = ppe_crops(frame, person_with_eyes(), 0.5)
    assert set(crops) == {"hard_hat", "vest", "boots"}
    for jpeg in crops.values():
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert MIN_SIDE <= min(img.shape[:2]) and max(img.shape[:2]) <= MAX_SIDE


def test_face_missed_by_keypoints_is_still_hidden_in_hard_hat_crop():
    """Real-image case: the eye keypoint was placed too low, so the crop reached the eyebrows.
    The face detector's box is blurred in the frame the crop is cut from, so the face stays hidden."""
    from safety.privacy import blur_faces

    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (500, 300, 3), dtype=np.uint8)
    p = person_with_eyes(eye_y=115)                          # true eyes are at y=95..105
    true_face = (125.0, 85.0, 175.0, 140.0)                  # what the face detector found
    blurred = blur_faces(frame, [p], 0.5, faces=[true_face])
    x1, y1, x2, y2 = (int(v) for v in head_region(p, 0.5))
    overlap = (slice(max(y1, 85), min(y2, 140)), slice(max(x1, 125), min(x2, 175)))
    assert not np.array_equal(blurred[overlap], frame[overlap])   # face part of the crop is pixelated
    crops = ppe_crops(blurred, p, 0.5)
    assert crops["hard_hat"] is not None


def test_vest_crop_is_wide_enough_for_a_person_seen_from_the_side():
    from safety.crops import torso_region
    from safety.detector import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

    p = person_with_eyes()                              # body box 100 px wide
    for i, xy in ((L_SHOULDER, (150, 150)), (R_SHOULDER, (152, 150)), (L_HIP, (149, 280)), (R_HIP, (151, 280))):
        p.keypoints[i] = xy                             # side view: all on one vertical line
        p.keypoint_conf[i] = 0.9
    x1, _, x2, _ = torso_region(p, 0.5)
    assert x2 - x1 >= 70


def test_body_part_outside_frame_gives_none():
    frame = np.zeros((100, 300, 3), np.uint8)  # person box goes far below the frame
    crops = ppe_crops(frame, person_with_eyes(eye_y=70), 0.5)
    assert crops["boots"] is None
