from FlightEnv.env import FlightEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import torch


import argparse

def train():
    log_dir = "logs/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    env = FlightEnv()
    policy_kargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 128], vf=[256, 128])
    )
    model = PPO("MlpPolicy", env, 
                learning_rate=1e-4,
                ent_coef=0.01,
                policy_kwargs=policy_kargs, 
                tensorboard_log=log_dir, verbose=1,
                device=device)
    # need to add entropy coefficient -> force the agent to explore 0.01 -> track KL divergence, too high means overexploration
    checkpoint_callback = CheckpointCallback(save_freq=1e6, save_path="temp_checkpoints", name_prefix="QuadrotorPPO")
    
    model.learn(total_timesteps=13*1e6, reset_num_timesteps=True, tb_log_name="ppo_flight_env", callback=checkpoint_callback)
    
    model.save("QuadrotorPPO")

def load():
    model = PPO.load("QuadrotorPPO")
    env = FlightEnv(render=True)
    obs, _ = env.reset()
    
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.load:
        load()
