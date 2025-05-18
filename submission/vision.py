# import numpy as np
# from scipy.ndimage import label, center_of_mass
# from collections import deque
# import random

# class ObstacleAvoid():
#     def __init__(self, threshold=50, right_edge_limit=100, left_edge_start=410):
#         self.threshold = threshold
#         self.right_edge_limit = right_edge_limit
#         self.left_edge_start = left_edge_start

#         # Memory mechanisms
#         self.last_turn = None
#         self.score_history_left = deque(maxlen=5)
#         self.score_history_right = deque(maxlen=5)
#         self.brightness_history_left = deque(maxlen=5)
#         self.brightness_history_right = deque(maxlen=5)

#     def get_decision(self, vision_left, vision_right):
#         mask_left = vision_left < self.threshold
#         mask_right = vision_right < self.threshold

#         _, obstacles_left = self.detect_obstacles(mask_left, vision_left, is_left=True)
#         _, obstacles_right = self.detect_obstacles(mask_right, vision_right, is_left=False)

#         score_left = self.compute_total_score(obstacles_left, is_left=True)
#         score_right = self.compute_total_score(obstacles_right, is_left=False)

#         # Store in memory
#         self.score_history_left.append(score_left)
#         self.score_history_right.append(score_right)
#         self.brightness_history_left.append(np.mean(vision_left))
#         self.brightness_history_right.append(np.mean(vision_right))

#         avg_score_left = np.mean(self.score_history_left)
#         avg_score_right = np.mean(self.score_history_right)
#         avg_brightness_left = np.mean(self.brightness_history_left)
#         avg_brightness_right = np.mean(self.brightness_history_right)

#         # Edge detection
#         obs_redge = np.any(vision_right[:, :self.right_edge_limit] < self.threshold)
#         obs_ledge = np.any(vision_left[:, self.left_edge_start:] < self.threshold)

#         # STUCK DETECTION
#         stuck = (
#             avg_score_left > 8000 and avg_score_right > 8000 and
#             obs_redge and obs_ledge
#         )

#         # Decision logic
#         decision = None

#         if stuck:
#             decision = random.choice(["pause_and_scan", "turn_left", "turn_right"])
#         elif obs_ledge and not obs_redge:
#             decision = "turn_right"
#         elif obs_redge and not obs_ledge:
#             decision = "turn_left"
#         elif avg_score_left > avg_score_right + 1000:
#             decision = "turn_right"
#         elif avg_score_right > avg_score_left + 1000:
#             decision = "turn_left"
#         elif max(score_left, score_right) < 6000 and not (obs_ledge or obs_redge):
#             decision = "clear"
#         else:
#             # Brightness bias if scores are similar
#             if abs(score_left - score_right) < 800:
#                 if avg_brightness_right > avg_brightness_left:
#                     decision = "turn_right"
#                 elif avg_brightness_left > avg_brightness_right:
#                     decision = "turn_left"
#                 else:
#                     decision = random.choice(["turn_left", "turn_right"])
#             else:
#                 decision = "slow_down"

#         # Hysteresis: stick to last turn with some probability
#         if decision in ["turn_left", "turn_right"]:
#             if self.last_turn and decision != self.last_turn:
#                 if random.random() < 0.6:  # 60% stickiness
#                     decision = self.last_turn
#             self.last_turn = decision

#         # print(f"[Vision] Scores — Left: {score_left:.2f}, Right: {score_right:.2f}, "
#         #       f"Brightness — Left: {avg_brightness_left:.1f}, Right: {avg_brightness_right:.1f}, "
#         #       f"obs_redge: {obs_redge}, obs_ledge: {obs_ledge}, Stuck: {stuck} → Decision: {decision}")

#         return decision

#     def detect_obstacles(self, mask, vision, is_left):
#         labeled, num_features = label(mask)
#         obstacles = []
#         img_w = vision.shape[1]
#         inner_col = img_w - 1 if is_left else 0

#         for i in range(1, num_features + 1):
#             region = labeled == i
#             height = np.sum(region)
#             intensity = np.max(vision[region])
#             weight = height * (255 - intensity)
#             center = center_of_mass(region)
#             dist_inner = abs(center[1] - inner_col)

#             # Exponential proximity score (closer obstacles matter more)
#             proximity_score = weight * np.exp(-dist_inner / (img_w * 0.3))

#             obstacles.append({
#                 "index": i,
#                 "height": height,
#                 "intensity": intensity,
#                 "weight": weight,
#                 "center": center,
#                 "dist_inner": dist_inner,
#                 "score": proximity_score
#             })

#         return labeled, obstacles

#     def compute_total_score(self, obstacles, is_left):
#         return sum(obs["score"] for obs in obstacles)

import numpy as np
from scipy.ndimage import label, center_of_mass

class ObstacleAvoid():
    def __init__(self, threshold=50, right_edge_limit=100, left_edge_start=410):
        self.threshold = threshold
        self.right_edge_limit = right_edge_limit  # For detecting obs_redge
        self.left_edge_start = left_edge_start    # For detecting obs_ledge

    def get_decision(self, vision_left, vision_right):
        mask_left = vision_left < self.threshold
        mask_right = vision_right < self.threshold

        _, obstacles_left = self.detect_obstacles(mask_left, vision_left, is_left=True)
        _, obstacles_right = self.detect_obstacles(mask_right, vision_right, is_left=False)

        score_left = self.compute_total_score(obstacles_left)
        score_right = self.compute_total_score(obstacles_right)

        # Use configurable edge ranges
        obs_redge = np.any(vision_right[:, :self.right_edge_limit] < self.threshold)
        obs_ledge = np.any(vision_left[:, self.left_edge_start:] < self.threshold)

        # Full logic with combined conditions
        if (score_left > score_right + 500) or (obs_ledge):
            decision = "turn_right"
        elif (score_right > score_left + 500) or (obs_ledge and obs_redge) or (obs_redge):
            decision = "turn_left"
        elif max(score_left, score_right) < 6000:
            decision = "clear"
        else:
            decision = "slow_down"

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