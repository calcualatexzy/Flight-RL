from FlightEnv.env import FlightEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
import gymnasium as gym
import torch

import multiprocessing
import argparse

def train():
    try:
        log_dir = "logs/"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_envs = 16
        multiprocessing.freeze_support()
        env = SubprocVecEnv([lambda: FlightEnv() for _ in range(num_envs)])
        env = VecMonitor(env)
        # env = FlightEnv()
        policy_kargs = dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64])
        )

        model = PPO("MlpPolicy", env,
                batch_size=256,
                learning_rate=1e-3,
                ent_coef=0.01,
                policy_kwargs=policy_kargs, 
                tensorboard_log=log_dir, verbose=1,
                device=device)
        
        checkpoint_callback = CheckpointCallback(save_freq=1e6, save_path="temp_checkpoints", name_prefix="QuadrotorPPO")
    
        model.learn(total_timesteps=12*1e7, reset_num_timesteps=True, tb_log_name="ppo_flight_env", callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted, saving model...")
    finally:
        model.save("QuadrotorPPO")
        print("Model saved.")
    model.save("QuadrotorPPO")

def load():
    model = PPO.load("QuadrotorPPO")
    env = FlightEnv(render=True)
    obs, _ = env.reset()
    rewards = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        rewards += reward
        env.render()
    print("Total reward: ", rewards)
    
def retrain():
    try:
        log_dir = "logs/"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_envs = 16
        multiprocessing.freeze_support()
        env = SubprocVecEnv([lambda: FlightEnv() for _ in range(num_envs)])
        env = VecMonitor(env)
        # env = FlightEnv()
        model = PPO.load("QuadrotorPPO", env=env, device=device)
        env.reset()

        checkpoint_callback = CheckpointCallback(save_freq=1e6, save_path="temp_checkpoints", name_prefix="QuadrotorPPO")

        model.learn(total_timesteps=20*1e7, reset_num_timesteps=False, tb_log_name="ppo_flight_env", callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted, saving model...")
    finally:
        model.save("QuadrotorPPO")
        print("Model saved.")
    model.save("QuadrotorPPO")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--retrain", action="store_true")   
    args = parser.parse_args()

    if args.train:
        train()
    elif args.load:
        load()
    elif args.retrain:
        retrain()
    