import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(eye):
    p1, p2, p3, p4, p5, p6 = eye

    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear
