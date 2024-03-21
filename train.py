from FlightEnv.env import FlightEnv
from stable_baselines3 import PPO
import gymnasium as gym
import torch

def main():
    log_dir = "logs/"

    env = FlightEnv()
    policy_kargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
    )
    model = PPO("MlpPolicy", env, 
                learning_rate=1e-4,
                policy_kwargs=policy_kargs, 
                tensorboard_log=log_dir, verbose=1)
    model.learn(total_timesteps=100000, reset_num_timesteps=True, tb_log_name="ppo_flight_env")
    
    model.save("QuadrotorPPO")
    


if __name__ == "__main__":
    main()