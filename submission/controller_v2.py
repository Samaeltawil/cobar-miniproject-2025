import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg

def compute_action_from_odor(obs, timestep):
    odor = obs["odor_intensity"][0]  # Attractive odours
    #normalize the odor intensity
    odor = (odor - np.min(odor)) / (np.max(odor) - np.min(odor))
    left = odor[0] + odor[2]  # Left palp + left antenna
    right = odor[1] + odor[3]  # Right palp + right antenna
    total = right + left + 1e-8
    turn  = right - left # Avoid division by zero
    cone_threshold = 2*1e-1
    print(turn)
    if np.abs(turn) > cone_threshold :
        print("state turn")
        if right>left:
            right_drive = 1
            left_drive =-1
        else:
            
            right_drive = -1
            left_drive =1
    else:
        print("state forward")
        # Gradient following
        left_ratio = left / total
        right_ratio = right / total
        base_drive = 1.0
        turn_bias = (right_ratio - left_ratio)
        left_drive = np.clip(base_drive - turn_bias, 0.0, 1.0)
        right_drive = np.clip(base_drive + turn_bias, 0.0, 1.0)
    return np.array([right_drive, left_drive])

class Controller(BaseController):
    def __init__(
        self,
        timestep=1e-4,
        seed=0,
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        

    def get_actions(self, obs: Observation) -> Action:
        
        action = compute_action_from_odor(obs, self.cpg_network.timestep)

        #print("action", action)

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action= action,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
