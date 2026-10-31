import numpy as np

from safety.detector import L_ANKLE, NOSE, R_ANKLE, Person, foot_point, is_full_body
from safety.privacy import blur_faces, face_box

BOX = (100.0, 50.0, 200.0, 450.0)


def keypoints(conf: float = 0.0):
    return np.zeros((17, 2)), np.full(17, conf)


def test_foot_is_midpoint_of_both_ankles():
    kp, kc = keypoints()
    kp[L_ANKLE], kp[R_ANKLE] = (130, 440), (170, 444)
    kc[L_ANKLE] = kc[R_ANKLE] = 0.9
    foot, from_ankles = foot_point(kp, kc, BOX, 0.5)
    assert foot == (150, 442) and from_ankles


def test_foot_uses_single_visible_ankle():
    kp, kc = keypoints()
    kp[L_ANKLE] = (130, 440)
    kc[L_ANKLE] = 0.9
    foot, from_ankles = foot_point(kp, kc, BOX, 0.5)
    assert foot == (130, 440) and from_ankles


def test_foot_falls_back_to_bottom_of_box_when_ankles_hidden():
    kp, kc = keypoints(conf=0.1)
    foot, from_ankles = foot_point(kp, kc, BOX, 0.5)
    assert foot == (150, 450) and not from_ankles


def test_full_body_needs_head_and_ankle():
    kp, kc = keypoints()
    kc[NOSE] = 0.9
    person = Person(1, BOX, 0.9, kp, kc, (150, 450), foot_from_ankles=False)
    assert not is_full_body(person, 0.5)
    person.foot_from_ankles = True
    assert is_full_body(person, 0.5)


def test_face_box_falls_back_to_head_zone_when_face_uncertain():
    kp, kc = keypoints(conf=0.1)
    assert face_box(kp, kc, BOX, 0.5) == (100, 50, 200, 50 + 400 * 0.3)


def test_blur_changes_face_but_not_original():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (500, 300, 3), dtype=np.uint8)
    kp, kc = keypoints(conf=0.1)
    person = Person(1, BOX, 0.9, kp, kc, (150, 450), False)
    out = blur_faces(frame, [person], 0.5)
    assert not np.array_equal(out[50:170, 100:200], frame[50:170, 100:200])  # head zone pixelated
    assert np.array_equal(out[300:, :], frame[300:, :])                       # rest untouched
    assert out is not frame
