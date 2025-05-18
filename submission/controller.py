# from cobar_miniproject.base_controller import Action, BaseController, Observation
# from .utils import get_cpg, step_cpg
# from .vision import ObstacleAvoid
# import numpy as np
# from flygym.vision import Retina
# from collections import deque


# def compute_odor_turn_bias(obs):
#     odor = obs["odor_intensity"][0]
#     odor = (odor - np.min(odor)) / (np.max(odor) - np.min(odor) + 1e-8)
#     left = odor[0] + odor[2]  # Left palp + antenna
#     right = odor[1] + odor[3]  # Right palp + antenna
#     total = left + right + 1e-8
#     turn_bias = (right - left) / total  # Positive: turn right, Negative: turn left
#     return np.clip(turn_bias, -1.0, 1.0)


# class Controller(BaseController):
#     def __init__(self, timestep=1e-4, seed=0):
#         from flygym.examples.locomotion import PreprogrammedSteps
#         super().__init__()
#         self.quit = False
#         self.cpg_network = get_cpg(timestep=timestep, seed=seed)
#         self.preprogrammed_steps = PreprogrammedSteps()
#         self.navigator = ObstacleAvoid(threshold=50)
#         self.retina = Retina()

#         self.last_vision_decision = "clear"
#         self.phase = "orienting"
#         self.phase_timer = 0
#         self.orientation_duration = 200

#         self.odor_buffer = deque(maxlen=5)

#     def get_actions(self, obs: Observation) -> Action:
#         turn_bias = compute_odor_turn_bias(obs)
#         self.odor_buffer.append(turn_bias)
#         smoothed_turn_bias = np.mean(self.odor_buffer)

#         if obs['vision_updated']:
#             vision_left = self.retina.hex_pxls_to_human_readable(obs["vision"][0], color_8bit=True).max(axis=-1)
#             vision_right = self.retina.hex_pxls_to_human_readable(obs["vision"][1], color_8bit=True).max(axis=-1)
#             self.last_vision_decision = self.navigator.get_decision(vision_left, vision_right)

#         if self.phase == "orienting":
#             self.phase_timer += 1
#             if abs(smoothed_turn_bias) > 0.1:
#                 if smoothed_turn_bias > 0:
#                     action_array = np.array([1.0, -0.2])  # Turn right in place
#                 else:
#                     action_array = np.array([-0.2, 1.0])  # Turn left in place
#             else:
#                 action_array = np.array([0.0, 0.0])
#             if self.phase_timer > self.orientation_duration:
#                 self.phase = "moving"
#         else:
#             # Determine visual drive first
#             if self.last_vision_decision == "turn_left":
#                 visual_drive = np.array([0.2, 1.0])
#             elif self.last_vision_decision == "turn_right":
#                 visual_drive = np.array([1.0, 0.2])
#             elif self.last_vision_decision == "slow_down":
#                 visual_drive = np.array([0.4, 0.4])
#             else:
#                 visual_drive = np.array([1.0, 1.0])

#             # Apply odor turn if strong enough
#             odor_turn_threshold = 0.3
#             if abs(smoothed_turn_bias) > odor_turn_threshold:
#                 if smoothed_turn_bias > 0:
#                     odor_drive = np.array([1.0, -0.2])  # Turn right in place
#                 else:
#                     odor_drive = np.array([-0.2, 1.0])  # Turn left in place
#                 # Odor dominates turning if it's strong
#                 action_array = 0.6 * visual_drive + 0.4 * odor_drive
#             else:
#                 # Mild odor effect, just a small steer
#                 base_drive = 1.0
#                 left = np.clip(base_drive - smoothed_turn_bias, 0.0, 1.0)
#                 right = np.clip(base_drive + smoothed_turn_bias, 0.0, 1.0)
#                 odor_drive = np.array([right, left])
#                 action_array = 0.7 * visual_drive + 0.3 * odor_drive

#             action_array = np.clip(action_array, 0.0, 1.0)

#         joint_angles, adhesion = step_cpg(
#             cpg_network=self.cpg_network,
#             preprogrammed_steps=self.preprogrammed_steps,
#             action=action_array,
#         )

#         return {
#             "joints": joint_angles,
#             "adhesion": adhesion,
#         }

#     def reset(self, **kwargs):
#         self.cpg_network.reset()
#         self.phase = "orienting"
#         self.phase_timer = 0
#         self.last_vision_decision = "clear"
#         self.odor_buffer.clear()

#     def done_level(self, obs: Observation):
#         return self.quit


from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from .vision import ObstacleAvoid
import numpy as np
from flygym.vision import Retina
from collections import deque


def compute_odor_turn_bias(obs):
    odor = obs["odor_intensity"][0]
    odor = (odor - np.min(odor)) / (np.max(odor) - np.min(odor) + 1e-8)
    left = odor[0] + odor[2]  # Left palp + antenna
    right = odor[1] + odor[3]  # Right palp + antenna
    total = left + right + 1e-8
    turn_bias = (right - left) / total  # Positive: turn right, Negative: turn left
    return np.clip(turn_bias, 0, 1.0)


class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        from flygym.examples.locomotion import PreprogrammedSteps
        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.navigator = ObstacleAvoid(threshold=50)
        self.retina = Retina()

        self.last_vision_decision = "clear"
        self.phase = "orienting"  # either "orienting" or "moving"
        self.phase_timer = 0
        self.orientation_duration = 200  # steps for initial orientation

        # Odor memory buffer to smooth out fluctuations
        self.odor_buffer = deque(maxlen=5)

    def get_actions(self, obs: Observation) -> Action:
        # Compute and store odor-based turn bias
        turn_bias = compute_odor_turn_bias(obs)
        self.odor_buffer.append(turn_bias)
        smoothed_turn_bias = np.mean(self.odor_buffer)

        # Vision processing
        if obs['vision_updated']:
            vision_left = self.retina.hex_pxls_to_human_readable(obs["vision"][0], color_8bit=True).max(axis=-1)
            vision_right = self.retina.hex_pxls_to_human_readable(obs["vision"][1], color_8bit=True).max(axis=-1)
            self.last_vision_decision = self.navigator.get_decision(vision_left, vision_right)
        else:
            # If vision is not updated, we rely more on odor
            if abs(smoothed_turn_bias) > 0.05:
                self.last_vision_decision = "bias_odor"

        # Phase switching logic
        if self.phase == "orienting":
            self.phase_timer += 1
            if abs(smoothed_turn_bias) > 0.1:
                action_array = np.array([1.0 - smoothed_turn_bias, 1.0 + smoothed_turn_bias]) * 0.5
            else:
                action_array = np.array([0.0, 0.0])

            if self.phase_timer > self.orientation_duration:
                self.phase = "moving"
        else:
            # Odor + Vision based decision
            if self.last_vision_decision == "turn_left":
                action_array = np.array([0.1, 1.0])
            elif self.last_vision_decision == "turn_right":
                action_array = np.array([1.0, 0.1])
            elif self.last_vision_decision == "slow_down":
                action_array = np.array([0.4, 0.4])
            elif self.last_vision_decision == "bias_odor":
                # Favor odor-based turning
                base_drive = 0.8
                left_drive = np.clip(base_drive - smoothed_turn_bias, 0.0, 1.0)
                right_drive = np.clip(base_drive + smoothed_turn_bias, 0.0, 1.0)
                action_array = np.array([right_drive, left_drive])
            else:
                # Default: move forward while slightly biasing based on odor
                base_drive = 1.0
                left_drive = np.clip(base_drive - 0.5 * smoothed_turn_bias, 0.0, 1.0)
                right_drive = np.clip(base_drive + 0.5 * smoothed_turn_bias, 0.0, 1.0)
                action_array = np.array([right_drive, left_drive])
            
            action_array *= 0.8

        # Generate motor output
        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=action_array,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.phase = "orienting"
        self.phase_timer = 0
        self.last_vision_decision = "clear"
        self.odor_buffer.clear()

    def done_level(self, obs: Observation):
        return self.quit