from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy



# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
env = make_atari_env('Boxing-v0', n_envs=8, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)
model = DQN('CnnPolicy', env, verbose=1, tensorboard_log= "./logs/atari_breakout")
model.learn(total_timesteps=1000000)
model.save("atari_breakout")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic= True)
print(mean_reward, std_reward)
