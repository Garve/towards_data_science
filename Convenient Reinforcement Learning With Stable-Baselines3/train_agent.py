import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

if __name__ == "__main__":
    env_name = "SpaceInvadersNoFrameskip-v4"  # change to another Atari game if you want
    env = SubprocVecEnv([lambda: AtariWrapper(gym.make(env_name)) for _ in range(4)])  # train 4 game envs in parallel, scale down images for faster training
    env = VecFrameStack(env, n_stack=4)  # don't only use a still image for training, but the last 4 frames
    env = VecTransposeImage(env)  # technical magic for putting the channels of the animation in the first coordinate, i.e., turning HxWxC into CxHxW

    config = {
        "batch_size": 32,
        "buffer_size": 10000,
        "exploration_final_eps": 0.02,
        "exploration_fraction": 0.1,
        "gamma": 0.99,
        "gradient_steps": 4,
        "learning_rate": 1e-4,
        "learning_starts": 10000,
        "target_update_interval": 1000,
        "train_freq": 4,
    }

    eval_callback = EvalCallback(
        eval_env=env,
        eval_freq=1000,
        n_eval_episodes=10,
        best_model_save_path=f"./logs/{env_name}",
        log_path=f"./logs/{env_name}",
    )
    save_callback = CheckpointCallback(save_freq=10000, save_path=f"./logs/{env_name}")

    model = DQN("CnnPolicy", env, verbose=0, **config)  # CnnPolicy creates default convolutional neural net for processing screen pixels
    model.learn(total_timesteps=10_000_000, progress_bar=True, callback=[eval_callback, save_callback])
