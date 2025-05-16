from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from .vision import ObstacleAvoid
import numpy as np
from flygym.vision import Retina
from .olfactory import compute_action_from_odor


class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.navigator = ObstacleAvoid(threshold=50)
        self.retina = Retina()

    def reset(self, **kwargs):
        self.cpg_network.reset()

    def done_level(self, obs: Observation):
        return self.quit

    def get_actions(self, obs: Observation) -> Action:
        # Convert vision to grayscale
        vision_left = self.retina.hex_pxls_to_human_readable(obs["vision"][0], color_8bit=True).max(axis=-1)
        vision_right = self.retina.hex_pxls_to_human_readable(obs["vision"][1], color_8bit=True).max(axis=-1)

        # Get decision from vision module
        decision = self.navigator.get_decision(vision_left, vision_right)

        if decision == "turn_left":
            action_array = np.array([0, 1.0])
        elif decision == "turn_right":
            action_array = np.array([1.0, 0])
        elif decision == "slow_down":
            action_array = compute_action_from_odor(obs)*3
        else:
            action_array = compute_action_from_odor(obs)

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=action_array,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }