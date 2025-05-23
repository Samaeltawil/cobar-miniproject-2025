import numpy as np


def compute_action_from_odor(obs):
    odor = obs["odor_intensity"][0]  # Attractive odours
    #normalize the odor intensity
    odor = (odor - np.min(odor)) / (np.max(odor) - np.min(odor))
    left = odor[0] + odor[2]  # Left palp + left antenna
    right = odor[1] + odor[3]  # Right palp + right antenna
    total = right + left + 1e-8
    turn  = right - left # Avoid division by zero
    cone_threshold = 2*1e-1
    if np.abs(turn) > cone_threshold :
        if right>left:
            right_drive = 1
            left_drive =-1
        else:
            
            right_drive = -1
            left_drive =1
    else:
        # Gradient following
        left_ratio = left / total
        right_ratio = right / total
        base_drive = 1.0
        turn_bias = (right_ratio - left_ratio)
        left_drive = np.clip(base_drive - turn_bias, 0.0, 1.0)
        right_drive = np.clip(base_drive + turn_bias, 0.0, 1.0)
    return np.array([float(right_drive), float(left_drive)])