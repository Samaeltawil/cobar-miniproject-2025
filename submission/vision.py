import numpy as np
from scipy.ndimage import label, center_of_mass
from enum import Enum
class Direction(Enum):
    CLEAR = 0
    LEFT = 1
    RIGHT = 2
    BACK = 3

class ObstacleAvoid():
    def __init__(self, threshold=50, right_edge_limit=100, left_edge_start=410):
        self.threshold = threshold
        self.right_edge_limit = right_edge_limit  # For detecting obs_redge
        self.left_edge_start = left_edge_start 
        self._debug_once = True    # For detecting obs_ledge
        self._react_thresh = self.threshold * 0.85
        self._edge_band   = 120 
    
    
    def get_decision(self, vision_left, vision_right):

        

        mask_left = vision_left < self._react_thresh
        mask_right = vision_right < self._react_thresh

        _, obstacles_left = self.detect_obstacles(mask_left, vision_left, is_left=True)
        _, obstacles_right = self.detect_obstacles(mask_right, vision_right, is_left=False)

        score_left = self.compute_total_score(obstacles_left)
        score_right = self.compute_total_score(obstacles_right)

        # Use configurable edge ranges
        obs_redge = np.any(vision_right[:, :self._edge_band] < self._react_thresh)
        obs_ledge = np.any(vision_left[:, -self._edge_band:] < self._react_thresh)

        # Full logic with combined conditions

        if max(score_left, score_right) < 4500:
            decision = Direction.CLEAR
        elif (score_left > score_right) or (obs_ledge):
            decision = Direction.RIGHT
        else:
            decision = Direction.LEFT
        
        

        # print(f"[Vision] Scores — Left: {score_left:.2f}, Right: {score_right:.2f}, "
        #       f"obs_redge: {obs_redge}, obs_ledge: {obs_ledge} → Decision: {decision}")

        return decision

    def detect_obstacles(self, mask, vision, is_left):
        labeled, num_features = label(mask)
        obstacles = []
        img_w = vision.shape[1]
        inner_col = img_w - 1 if is_left else 0

        for i in range(1, num_features + 1):
            region = labeled == i
            height = np.sum(region)
            intensity = np.max(vision[region])
            weight = height * (255 - intensity)
            center = center_of_mass(region)
            dist_inner = abs(center[1] - inner_col)
            proximity_score = weight / (1 + dist_inner)

            obstacles.append({
                "index": i,
                "height": height,
                "intensity": intensity,
                "weight": weight,
                "center": center,
                "dist_inner": dist_inner,
                "score": proximity_score
            })

        return labeled, obstacles

    def compute_total_score(self, obstacles):
        return sum(obs["score"] for obs in obstacles)

    # ------------------------------------------------------------------
    #  Looming-ball detector: looks for a RED blob in ONE eye’s RGB frame
    # ------------------------------------------------------------------
    def vision_ball(self, rgb_frame) -> Direction:
        """
        Parameters
        ----------
        rgb_frame : ndarray  (H × W × 3, uint8)
            A colour image from either the left or right retina.

        Returns
        -------
        Direction
            LEFT   – ball in left half  → dodge right
            RIGHT  – ball in right half → dodge left
            CLEAR  – no ball (or still too small)
        """
        # --- DEBUG: print the largest red blob each call ---
        red_mask = (
            (rgb_frame[:, :, 0] > 150) &  # loosen thresholds for test
            (rgb_frame[:, :, 1] < 110) &
            (rgb_frame[:, :, 2] < 110)
        )
        print("ball-px:", red_mask.sum())
        
        
        # 1. Binary mask of “red enough” pixels
        red_mask = (
            (rgb_frame[:, :, 0] > 180) &    # R channel high
            (rgb_frame[:, :, 1] <  80) &    # G low
            (rgb_frame[:, :, 2] <  80)      # B low
        )

        # print("ball pixels:", red_mask.sum())
        if red_mask.sum() < 400:            
            return Direction.CLEAR
        else:
            return Direction.BACK

