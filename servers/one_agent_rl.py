import holoocean
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

SCENARIO_NAME = "OpenWater-TorpedoProfilingSonar" 
AGENT_NAME = "auv0"
SONAR_SENSOR_NAME = "ProfilingSonar" 

LOG_DIR = "holoocean_rl_cnn_logs"  
MODEL_DIR = "holoocean_rl_cnn_models" 
TOTAL_TIMESTEPS = 500_000  
SAVE_FREQ = 25_000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class HoloOceanMinefieldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, scenario_name, agent_name, sonar_sensor_name, render_mode=None):
        super(HoloOceanMinefieldEnv, self).__init__()

        self.scenario_name = scenario_name
        self.agent_name = agent_name
        self.sonar_sensor_name = sonar_sensor_name
        self.render_mode = render_mode
        self.env = None 
        self.sonar_config = None
        self.plot_elements = None
        self.last_dist_to_goal = None 

  
        try:
            print("Initializing temporary HoloOcean env to get sonar config...")
            temp_env = holoocean.make(self.scenario_name)
            agent_cfg = None
            for ag_cfg in temp_env.config['agents']:
                if ag_cfg['agent_name'] == self.agent_name:
                    agent_cfg = ag_cfg
                    break
            if agent_cfg is None:
                raise ValueError(f"Agent '{self.agent_name}' not found in scenario.")

            found_sonar = False
            for sensor in agent_cfg['sensors']:
                if sensor['sensor_name'] == self.sonar_sensor_name:
                    self.sonar_config = sensor["configuration"]
                    found_sonar = True
                    print(f"Found Sonar Config: {self.sonar_config}")
                    break
            if not found_sonar:
                raise ValueError(f"Sonar sensor '{self.sonar_sensor_name}' not found for agent '{self.agent_name}'")
            temp_env.close() 
            print("Temporary env closed.")
        except Exception as e:
            print(f"Error during initial HoloOcean setup for sonar config: {e}")
            print("Falling back to default sonar config.")
            self.sonar_config = {'Azimuth': 90, 'RangeMin': 0.5, 'RangeMax': 50, 'RangeBins': 100, 'AzimuthBins': 90}


        self.azi = self.sonar_config['Azimuth']
        self.minR = self.sonar_config['RangeMin']
        self.maxR = self.sonar_config['RangeMax']
        self.binsR = self.sonar_config['RangeBins']
        self.binsA = self.sonar_config['AzimuthBins']

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]),
                                       high=np.array([1.0, 1.0]),
                                       dtype=np.float32)
        self.max_thrust = 50 
        self.max_yaw_torque = 20 

        sonar_obs_shape = (1, self.binsR, self.binsA)
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=sonar_obs_shape,
                                            dtype=np.float32)
        print(f"Observation Space Shape (CNN): {self.observation_space.shape}")

        self.current_step = 0
        self.max_episode_steps = 500

        if self.render_mode == 'human':
            self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8,5))
        ax.set_theta_zero_location("N")
        ax.set_thetamin(-self.azi/2)
        ax.set_thetamax(self.azi/2)
        theta = np.linspace(-self.azi/2, self.azi/2, self.binsA)*np.pi/180
        r_polar = np.linspace(self.minR, self.maxR, self.binsR)
        T, R_polar = np.meshgrid(theta, r_polar)
        z = np.zeros_like(T)
        plt.grid(False)
        plot = ax.pcolormesh(T, R_polar, z, cmap='gray', shading='auto', vmin=0, vmax=1)
        plt.tight_layout()
        self.plot_elements = {'fig': fig, 'ax': ax, 'plot': plot, 'T': T, 'R_polar': R_polar}
        fig.canvas.flush_events()

    def _process_sonar_data(self, sonar_image_2d):
        max_val = np.max(sonar_image_2d)
        if max_val > 0:
            sonar_image_2d = sonar_image_2d / max_val
        
        return np.expand_dims(sonar_image_2d, axis=0).astype(np.float32)

    def _get_observation(self, holoocean_state):
        sonar_data_raw = np.zeros((self.binsR, self.binsA)) 
        if self.sonar_sensor_name in holoocean_state[self.agent_name]:
            sonar_data_raw = holoocean_state[self.agent_name][self.sonar_sensor_name]

        processed_sonar = self._process_sonar_data(sonar_data_raw)
        return processed_sonar

    def _calculate_reward(self, sonar_image_2d, holoocean_state):

        reward = 0.0

        mid_range_start = self.binsR // 4
        mid_range_end = 3 * self.binsR // 4
        mid_azi_start = self.binsA // 3
        mid_azi_end = 2 * self.binsA // 3
        
        target_region = sonar_image_2d[mid_range_start:mid_range_end, mid_azi_start:mid_azi_end]
        

        intensity_mean = np.mean(target_region)
        intensity_max = np.max(target_region)

        if intensity_max > 0.7: 
             reward += 5.0 * intensity_mean 


        reward -= 0.1


        if "CollisionSensor" in holoocean_state[self.agent_name]:
            if holoocean_state[self.agent_name]["CollisionSensor"]:
                reward -= 50.0
                

        # Small penalty for excessive control effort (encourages smoother actions)
        # command = holoocean_state[self.agent_name].get("AppliedCommand", np.zeros(6))
        # reward -= 0.01 * (np.abs(command[0]/self.max_thrust) + np.abs(command[5]/self.max_yaw_torque))

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.env is None:
            print("Making HoloOcean Environment...")
            self.env = holoocean.make(self.scenario_name)
            print("HoloOcean Environment Created.")
        else:
            print("Resetting HoloOcean Environment...")
            self.env.reset()
            print("HoloOcean Environment Reset.")

        self.current_step = 0
        self.last_dist_to_goal = None

 
        self.env.act(self.agent_name, np.zeros(6)) 
        holoocean_state = self.env.tick()

        observation = self._get_observation(holoocean_state)
        info = {}

        if self.render_mode == 'human' and self.plot_elements:
            self.render(holoocean_state)

        return observation, info

    def step(self, action):
        thrust = action[0] * self.max_thrust
        yaw_torque = action[1] * self.max_yaw_torque
        command = np.array([thrust, 0, 0, 0, 0, yaw_torque], dtype=np.float32)

        self.env.act(self.agent_name, command)
        holoocean_state = self.env.tick()

        observation = self._get_observation(holoocean_state)


        raw_sonar = np.zeros((self.binsR, self.binsA))
        if self.sonar_sensor_name in holoocean_state[self.agent_name]:
            raw_sonar = holoocean_state[self.agent_name][self.sonar_sensor_name]

        reward = self._calculate_reward(raw_sonar, holoocean_state)

        self.current_step += 1
        terminated = False
        truncated = False

        if "CollisionSensor" in holoocean_state[self.agent_name]:
            if holoocean_state[self.agent_name]["CollisionSensor"]:
                terminated = True 

        if self.current_step >= self.max_episode_steps:
            truncated = True

        if self.render_mode == 'human' and self.plot_elements:
            self.render(holoocean_state)

        return observation, reward, terminated, truncated, {}

    def render(self, holoocean_state_for_render=None):
        if self.render_mode == 'human' and self.plot_elements:
            s = np.zeros((self.binsR, self.binsA)) # Default
            if holoocean_state_for_render and self.sonar_sensor_name in holoocean_state_for_render[self.agent_name]:
                s = holoocean_state_for_render[self.agent_name][self.sonar_sensor_name]
            
            max_val = np.max(s)
            s_vis = s / max_val if max_val > 0 else s
            
            self.plot_elements['plot'].set_array(s_vis.ravel())
            self.plot_elements['fig'].canvas.draw()
            self.plot_elements['fig'].canvas.flush_events()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.render_mode == 'human' and self.plot_elements:
            plt.ioff()
            plt.close(self.plot_elements['fig'])
            self.plot_elements = None
        print("HoloOcean Environment Closed.")


if __name__ == "__main__":
    env = HoloOceanMinefieldEnv(
        scenario_name=SCENARIO_NAME,
        agent_name=AGENT_NAME,
        sonar_sensor_name=SONAR_SENSOR_NAME,
        render_mode=None 
    )
    env = DummyVecEnv([lambda: env]) 

    print("Defining PPO agent with CnnPolicy...")
    model = PPO("CnnPolicy", 
                env,
                verbose=1,
                tensorboard_log=LOG_DIR,
                learning_rate=3e-4, 
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                # policy_kwargs can be used to customize CNN architecture, e.g.:
                # policy_kwargs=dict(
                #     features_extractor_class=NatureCNN, # Default
                #     features_extractor_kwargs=dict(features_dim=512),
                # )
               )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // env.num_envs, 1),
        save_path=MODEL_DIR,
        name_prefix=f"ppo_cnn_{SCENARIO_NAME.split('/')[-1].lower()}" # Use scenario name for prefix
    )

    print(f"Starting PPO training (CNN) for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        final_model_path = os.path.join(MODEL_DIR, f"ppo_cnn_{SCENARIO_NAME.split('/')[-1].lower()}_final.zip")
        model.save(final_model_path)
        print(f"Training finished. Final model saved to {final_model_path}")
        env.close()

    print("\n--- RL Training Script Finished ---")

    # Example: How to load and run the trained CNN model
    # print("\n--- Loading and testing trained CNN agent ---")
    # loaded_model = PPO.load(final_model_path)
    # eval_env = HoloOceanMinefieldEnv(SCENARIO_NAME, AGENT_NAME, SONAR_SENSOR_NAME, render_mode='human')
    # obs, _ = eval_env.reset()
    # for _ in range(2000):
    #     action, _states = loaded_model.predict(obs.reshape(1, 1, env.get_attr('binsR')[0], env.get_attr('binsA')[0]), deterministic=True) # Reshape for predict if needed
    #     obs, rewards, terminated, truncated, info = eval_env.step(action[0]) # Get action from batch
    #     done = terminated or truncated
    #     if done:
    #         print("Evaluation episode finished.")
    #         obs, _ = eval_env.reset()
    # eval_env.close()