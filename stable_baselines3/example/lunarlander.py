import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import exputils as eu
import exputils.data.logging as log

def default_config():
    return eu.AttrDict(
        seed = 42 + 1,
        gamma =  0.99,
        epsilon = 1,
        eps_min = 0.01,
        lr = 0.00025,
        rbf_on = True,
        n_neurons_per_input = 5,
        ranges = [-1.0, 1.0],
        sigma = None,
        is_trainable = True,
    )

def run(config = None, **kwargs):
    # Create environment
    env = gym.make('LunarLander-v2')
    config = eu.combine_dicts(kwargs, config, default_config())
    log.activate_tensorboard()

    # Instantiate the agent
    model = DQN('MlpPolicy', env, verbose=1, learning_rate= config.lr, gamma= config.gamma, config = config)
    # Train the agent
    model.learn(total_timesteps=int(1e6))
    # Save the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    log.add_scalar('mean_reward', mean_reward)
    log.add_scalar('std_reward', std_reward)
    log.save()


run()