from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import exputils as eu
import torch
import collections
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor
from zmq import device
import torch.nn.functional as F

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
import time

class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, config = None, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.max = -1e6
        self.min = 1e6
        self.prev_max = -1e6
        self.prev_min = 1e6

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        if config != None and config.mrbf_on == True:
            self.linear = nn.Sequential(nn.Linear(n_flatten, 32), nn.ReLU(),
                                        MRBF(32, config.mrbf_units),
                                        nn.Linear(config.mrbf_units, features_dim), nn.ReLU())
        else:
            self.linear = nn.Sequential(nn.Linear(n_flatten, config.latent_dim), nn.ReLU(),
                                        nn.Linear(config.latent_dim, 128), nn.ReLU(),
                                        nn.Linear(128, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = self.cnn(observations)
        #print the max and min according to the observations from cnn
        self.max = max(self.max, observations.max().item())
        self.min = min(self.min, observations.min().item())
        if self.max != self.prev_max or self.min != self.prev_min:
            print(f"max: {self.max}, min: {self.min}")
            self.prev_max = self.max
            self.prev_min = self.min

        observations = self.linear(observations)
        return observations

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    config = None,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """
    if len(net_arch) > 0:
        if config != None and config.rbf_on == True:
            modules = [nn.Linear(input_dim*config.n_neurons_per_input, net_arch[0]), activation_fn()]
        elif config != None and config.rbf_mlp == True and config.sutton_maze == False:
            modules = [RBFLayer(input_dim, config = config)]
            modules.append(nn.Linear(input_dim* config.n_neurons_per_input, net_arch[0]))
            modules.append(activation_fn())
        elif config != None and config.mrbf_on == True:
            modules = [MRBF(input_dim, config.mrbf_units)]
            modules.append(nn.Linear(config.mrbf_units, net_arch[0]))
            modules.append(activation_fn())
        elif config != None and config.rbf_on == False and config.rbf_mlp == False and config.mrbf_on == False:
            modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
        else:
            modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]

    else:
        modules = []    

    for idx in range(len(net_arch) - 1):
        if config != None and config.sutton_maze == True and net_arch[idx] == config.latent_dim and config.rbf_mlp == True:
            modules.append(RBFLayer(net_arch[idx], config = config))
            modules.append(nn.Linear(net_arch[idx] * config.n_neurons_per_input, net_arch[idx + 1]))
            modules.append(activation_fn())
        else:
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(activation_fn())
    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules



class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous feature extractor (i.e. a CNN) or directly
    the observations (if no feature extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer))  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip_longest(policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.shared_net(features))


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch


class RBFLayer(torch.nn.Module):
    """RBF layer that has for each input dimension n RBF neurons.

    Each RBF neuron y_i,j for input x_i has a Gaussian activation function:
        y_i,j = exp(-0.5 * ((x_i - mu_i,j) / sigma_i,j)**2)

    Config:
        n_neurons_per_input (int): Number of RBF neurons for each input dimension.
            Also defines the output dimensions (n_out = n_input * n_neurons_per_input).
            (default = 5)
        ranges (list): Defines the value range ([min_value, max_value]) of each input dimension used to
            define the initial peaks of the RBF layers. The RBF peaks are equally distributed within the
            range. For example, having RBF neurons (n_neurons_per_input=5) with in a range=[-1.0, 1.0] yields
            the following peaks: [-1.0, -0.5, 0.0, 0.5, 1.0]
            Can be a single range that is used for each input dimension (ranges=[0.0, 1.0]) or a range
            for each dimension (ranges=[[0.0, 1.0],[-2.0, 2.0]]).
            (default = [-1.0, 1.0])
        sigma (float or list): Defines the spread (sigma) of each RBF neuron.
            Can be a single sigma (sigma=0.5) used for each input dimension or a list of sigmas for each
            individual input dimension (sigma=[0.5, 0.3]).
            If no sigma is given (sigma=None), then a sigma is chosen such that for an input value that is
            in the middle of the peaks of 2 neurons, both neurons have an equal activation of 0.5.
            Formula: (-dist**2/(8*np.log(0.5)))**0.5 where dist is the distance between the 2 peaks.
            (default = None)
        is_trainable (bool): True if the parameters of the RBF neurons (peak position and spread) are trainable.
            False if not.
            (default = False)

    Properties:
        n_in (int): Number input dimensions.
        n_neurons_per_input (int): Number of RBF neurons per input dimension.
        n_out (int): Number of output dimensions.

    Example:
        # input a batch of 2 inputs
        x = torch.tensor([
            [0.2, 0.4, 0.3],
            [-0.1, 0.2, 0.0]
        ])
        y = rbf_layer(x)
        print(y)
    """


    @staticmethod
    def default_config():
        return eu.AttrDict(
            n_neurons_per_input=None,
            ranges=[-3.0, 3.0],
            sigma=None,
            is_trainable=True,
        )


    @property
    def dtype(self):
        # torch.nn.Linear uses torch.float32 as dtype for parameters
        return torch.float32


    @property
    def n_in(self):
        return self._n_in


    @property
    def n_neurons_per_input(self):
        return self._n_neurons_per_input


    @property
    def n_out(self):
        return self._n_out


    def __init__(self, n_in, n_out=None, config=None, **argv):
        """Creates a RBF Layer.

        Args:
            n_in (int): Size of the input dimension.
        """
        super().__init__()
        self.config = eu.combine_dicts(argv, config, self.default_config())

        self._n_in = n_in
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.min = 1e+6
        self.max = -1e+6

        # identify self._n_neurons_per_input
        n_neurons_per_input_according_to_n_out = None
        n_neurons_per_input_according_to_config = None
        if n_out is None and self.config.n_neurons_per_input is None:
            raise ValueError('Either n_out or config.n_neurons_per_input must be set!')
        if n_out is not None:
            if n_out % n_in != 0:
                ValueError('n_in must be a divisible multitude of n_out!')
            n_neurons_per_input_according_to_n_out = int(n_out / n_in)
            self._n_neurons_per_input = n_neurons_per_input_according_to_n_out
        if self.config.n_neurons_per_input is not None:
            n_neurons_per_input_according_to_config = self.config.n_neurons_per_input
            self._n_neurons_per_input = n_neurons_per_input_according_to_config
        if n_neurons_per_input_according_to_n_out is not None and n_neurons_per_input_according_to_config is not None:
            if n_neurons_per_input_according_to_n_out != n_neurons_per_input_according_to_config:
                raise ValueError('Number of RBF neurons must be consistent in between config.n_neurons_per_input and n_out!')


        self._n_out = self._n_in * self.n_neurons_per_input

        if self.config.ranges is None:
            self.ranges = np.array([[-1.0, 1.0]] * self._n_in)
        elif np.ndim(self.config.ranges) == 1:
            self.ranges = np.array([self.config.ranges] * self._n_in)
        else:
            self.ranges = np.array(self.config.ranges)

        self.peaks = torch.Tensor(self.n_out)
        for input_idx in range(self._n_in):
            start_idx = input_idx * self._n_neurons_per_input
            end_idx = start_idx + self._n_neurons_per_input
            self.peaks[start_idx:end_idx] = torch.linspace(self.ranges[input_idx][0], self.ranges[input_idx][1], self._n_neurons_per_input)

        # handle different types of sigma parameters and convert them to a list with one sigma per input
        if self.config.sigma is None:
            self.sigma = np.zeros(self._n_in)
            for input_idx in range(self._n_in):
                dist = self.peaks[input_idx * self._n_neurons_per_input + 1] - self.peaks[input_idx * self._n_neurons_per_input]
                self.sigma[input_idx] = (-dist ** 2 / (8 * np.log(0.5))) ** 0.5
        elif not isinstance(self.config.sigma, collections.Sequence):
            self.sigma = np.ones(self._n_in) * self.config.sigma
        else:
            self.sigma = self.config.sigma

        self.sigmas = torch.Tensor(self.n_out)
        for input_idx in range(self._n_in):
            start_idx = input_idx * self._n_neurons_per_input
            end_idx = start_idx + self._n_neurons_per_input
            self.sigmas[start_idx:end_idx] = self.sigma[input_idx]

        # if the layer should be trainable, then add the peaks and sigmas as parameters
        if self.config.is_trainable:
            self.peaks = torch.nn.Parameter(self.peaks)
            self.sigmas = torch.nn.Parameter(self.sigmas)


    def forward(self, x):
        """Calculate the RBF layer output.

        Args:
            x (torch.Tensor): Torch tensor with a batch of inputs.
                The tensor has 2 dimensions, where each row vector is a single input.
        """

        # reapeat input vector so that every map-neuron gets its accordingly input
        # example: n_neuron_per_inpu = 3 then [[1,2,3]] --> [[1,1,1,2,2,2,3,3,3]]
        #if x.min() < self.min:
        #    self.min = x.min()
        #    print(self.min, self.max)

        #if x.max() > self.max:
        #    self.max = x.max()
        #    print(self.min, self.max)
        x = x.repeat_interleave(repeats=self.n_neurons_per_input, dim=1)
        # calculate gauss activation per map-neuron
        output =  torch.exp(-0.5 * ((x - self.peaks) / self.sigmas) ** 2)
        return output


class My_RBF(nn.Module):
  def __init__(self, input_features:int, output_features: int) -> None:
    super(My_RBF, self).__init__()
    self.input_features = input_features
    self.output_features = output_features
    print(self.input_features)
    print(self.output_features)
    self.weights = torch.ones(self.output_features, self.input_features)
    self.peaks = torch.ones(self.output_features, self.input_features)
    self.sigmas = torch.eye(self.output_features, self.input_features)


  def forward(self, input: Tensor) -> Tensor:
    distribution = MultivariateNormal(self.peaks, self.sigmas)
    prob = torch.exp(distribution.log_prob(input))
    return torch.sum(self.weights * prob, dim = 1)



class NatureCNNRBF(BaseFeaturesExtractor):
    """
    A CNN with a RBF layer network architecture
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box,config, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        self.config = config
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.fc1 = nn.Linear(n_flatten, config.latent_dim)
        self.rbf = RBFLayer(config.latent_dim, config = self.config)
        self.fc2 = nn.Linear(config.latent_dim*self.config.n_neurons_per_input, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = self.cnn(observations)
        observations = F.relu(self.fc1(observations))
        observations = self.rbf(observations)
        observations = F.relu(self.fc2(observations))
        return observations
        
        
class MRBF(torch.nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
            """

    def __init__(self, in_features, out_features):
        super(MRBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        #print(input.shape, size)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        #log_sigmas = self.log_sigmas.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        #distances = ((x - c).pow(2) / torch.exp(log_sigmas).pow(2)).sum(-1).pow(0.5)
        #
        output = torch.exp(-1*distances.pow(2))
        return output





