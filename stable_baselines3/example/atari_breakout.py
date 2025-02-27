from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import exputils as eu
import exputils.data.logging as log
from torchinfo import summary

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
def default_config():

    return eu.AttrDict(
        seed = 42 + 1,
        gamma =  0.99,
        epsilon = 1,
        eps_min = 0.01,
        lr = 0.00025,
        rbf_on = False,
        rbf = eu.AttrDict(
            n_neurons_per_input = 5,
            ranges = [-1.0, 1.0],
            sigma = None,
            is_trainable = True,		
        )
)
def run(config = None, **kwargs):
    config = eu.combine_dicts(kwargs, config, default_config())
    env = make_atari_env('ALE/Breakout-v5', n_envs=8, seed=0)
    #eu.activate_tensorboard()
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)
    if config.rbf_on:
        model = DQN('CNNRBFPolicy', env, verbose=1, learning_rate= config.lr, gamma= config.gamma,tensorboard_log= "./logs/atari_breakout_with_rbf", optimize_memory_usage= True, 
                config = config.rbf)
    else:
        model = DQN('CnnPolicy', env, verbose = 0, learning_rate= config.lr, gamma = config.gamma, tensorboard_log = "./logs/atari_breakout_without_rbf", optimize_memory_usage= True)
    model.learn(total_timesteps=2_500_000, tb_log_name="atari_breakout")
    model_stats = summary(model.policy, input_size=(32, 4, 84, 84), col_names=["kernel_size", "output_size", "num_params", "mult_adds"], depth= 5)
    print(model_stats)
    log.add_scalar("total number of parameters", model_stats.total_params)
    log.add_scalar("total number of multiplications and additions", model_stats.total_mult_adds)
    log.add_scalar("total number of trainable parameters", model_stats.trainable_params)
    log.save()
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100, deterministic= True)
    log.add_scalar("mean_reward", mean_reward)
    log.add_scalar("std_reward", std_reward)
    log.save()
