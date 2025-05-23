import numpy as np
from flygym.examples.locomotion import CPGNetwork, PreprogrammedSteps
from flygym.preprogrammed import all_leg_dofs
from flygym.vision import Retina
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import step_cpg
from .olfactory import compute_action_from_odor
from .vision import ObstacleAvoid, Direction

class Controller(BaseController):
    def __init__(self,timestep=1e-4,seed=0, **kwargs):
        super().__init__()

       
        self.preprogrammed_steps = PreprogrammedSteps()
        self.timestep = kwargs.get("timestep", 1e-4)
        self.scale_factor = kwargs.get("scale_factor", 0.25)
        self.intrinsic_freqs = kwargs.get("intrinsic_freqs", np.ones(6) * 36)
        self.intrinsic_amps = kwargs.get("intrinsic_amps", np.ones(6) * 6)
        # Further reduced Kp_dist and Kp_heading for even smoother control and better physics stability
        self.Kp_dist = kwargs.get("Kp_dist", 0.03) # Was 0.05
        self.Ki_dist = kwargs.get("Ki_dist", 0.05)
        self.Kp_heading = kwargs.get("Kp_heading", 1.0) # Was 1.5
        self.Ki_heading = kwargs.get("Ki_heading", 0.8)

        phase_biases = kwargs.get("phase_biases", np.pi * np.array([
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
        ]))

        coupling_weights = kwargs.get("coupling_weights", (phase_biases > 0) * 10)
        convergence_coefs = kwargs.get("convergence_coefs", np.ones(6) * 20)

        self.cpg_network = CPGNetwork(
            timestep=self.timestep,
            intrinsic_freqs=self.intrinsic_freqs,
            intrinsic_amps=self.intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            init_phases=kwargs.get("init_phases"),
            init_magnitudes=kwargs.get("init_magnitudes"),
            seed=kwargs.get("seed", 0),
        )

        self.quit = False
        self.has_reached_ball = False
        self.cumulative_error_dist = 0.0
        self.cumulative_error_heading = 0.0
        self.fly_pos = np.array([0.0, 0.0])
        self.heading = 0.0

        self.navigator = ObstacleAvoid(threshold=50)
        self.retina = Retina()

    def reset(self, seed=None, init_phases=None, init_magnitudes=None, **kwargs):
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
        self.cpg_network.reset(init_phases, init_magnitudes)

        self.cumulative_error_dist = 0.0
        self.cumulative_error_heading = 0.0
        self.fly_pos = np.array([0.0, 0.0])
        self.heading = 0.0
        self.quit = False
        self.has_reached_ball = False

    def update_position(self, obs):
        theta = obs["heading"]
        direction = np.array([np.cos(theta), np.sin(theta)])
        speed = np.linalg.norm(obs["velocity"])
        self.fly_pos += speed * direction * self.timestep * self.scale_factor
        self.heading = theta

    def done_level(self, obs: Observation) -> bool:
        return self.quit

    def get_cpg_joint_angles(self, action):
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs
        self.cpg_network.step() # This is where the CPG network is advanced

        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg, self.cpg_network.curr_phases[i], self.cpg_network.curr_magnitudes[i]
            )
            joints_angles.append(my_joints_angles)
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )
            adhesion_onoff.append(my_adhesion_onoff)

        return {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }

    def get_actions(self, obs: Observation) -> Action:
        self.update_position(obs)

        if not self.has_reached_ball:
            if obs.get("reached_odour", False):
                print("Ball reached! Switching to PID path integration.")
                self.has_reached_ball = True
                joints, adhesion = step_cpg(self.cpg_network, self.preprogrammed_steps, np.array([0.0, 0.0]))
                return {"joints": joints, "adhesion": adhesion}

            action_array = compute_action_from_odor(obs)

            if action_array.sum()>0.1:
                ball_left = self.navigator.vision_ball(obs["raw_vision"][0])
                ball_right = self.navigator.vision_ball(obs["raw_vision"][1])

                if ball_left == Direction.BACK or ball_right == Direction.BACK:
                    print("Looming ball detected! Initiating evasive backward action.")
                    action_array[0] = -1.0
                    action_array[1] = -1.0
                    joint_angles, adhesion = step_cpg(self.cpg_network, self.preprogrammed_steps, action_array)

                else: 
                    vision_left = self.retina.hex_pxls_to_human_readable(obs["vision"][0], color_8bit=True).max(axis=-1)
                    vision_right = self.retina.hex_pxls_to_human_readable(obs["vision"][1], color_8bit=True).max(axis=-1)
                    decision = self.navigator.get_decision(vision_left, vision_right)
                
                    if decision == Direction.RIGHT:
                        action_array[0] = 1.0
                        action_array[1] = 0.5
                    elif decision == Direction.LEFT:
                        action_array[0] = 0.5
                        action_array[1] = 1.0
                

            joint_angles, adhesion = step_cpg(self.cpg_network, self.preprogrammed_steps, action_array)
            return {"joints": joint_angles, "adhesion": adhesion}

        # PID-controlled return to origin using get_cpg_joint_angles
        error = -self.fly_pos
        error_dist = np.linalg.norm(error)
        error_heading = np.arctan2(error[1], error[0]) - self.heading
        error_heading = ((error_heading + np.pi) % (2 * np.pi)) - np.pi

        self.cumulative_error_dist += error_dist * self.timestep
        self.cumulative_error_heading += error_heading * self.timestep
        self.cumulative_error_dist = np.clip(self.cumulative_error_dist, -10.0, 10.0)
        self.cumulative_error_heading = np.clip(self.cumulative_error_heading, -np.pi, np.pi)

        speed_control = np.sqrt(self.Kp_dist * error_dist + self.Ki_dist * self.cumulative_error_dist)
        speed_heading = self.Kp_heading * error_heading + self.Ki_heading * self.cumulative_error_heading

        # Clip speed_left and speed_right to prevent overly aggressive control signals
        # Adjust these bounds based on what works best for simulation stability and fly movement
        speed_left = np.clip(speed_control * (1 - speed_heading / 2), 0.0, 2.0) # Example bounds, adjust as needed
        speed_right = np.clip(speed_control * (1 + speed_heading / 2), 0.0, 2.0) # Example bounds, adjust as needed

        if error_dist < 1: # Changed back to 1 for consistency with original code
            print("Returned to origin. Finished.")
            self.quit = True
        
        # Call get_cpg_joint_angles once and store the result
        cpg_actions = self.get_cpg_joint_angles(np.array([speed_left, speed_right]))
        
        # If you need to print the actions for debugging, print the stored variable
        # print(cpg_actions) 
        
        return cpg_actions