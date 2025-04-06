import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle in degrees between three points a, b, c.
    
    Args:
        a (tuple): First point coordinates (x, y).
        b (tuple): Middle point coordinates (x, y).
        c (tuple): Third point coordinates (x, y).
    
    Returns:
        float: Angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)