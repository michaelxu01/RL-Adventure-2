import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from IPython.display import clear_output
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done

	def __len__(self):
		return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
	def action(self, action):
		low = self.action_space.low
		high = self.action_space.high

		action = low + (action + 1.0) * 0.5 * (high - low)
		action = np.clip(action, low, high)

		return action

	def reverse_action(self, action):
		low = self.action_space.low
		high = self.action_space.high

		action = 2 * (action - low) / (high - low) - 1
		action = np.clip(action, low, high)

		return actions

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


class ValueNetwork(nn.Module):
	def __init__(self, state_dim, hidden_dim, init_w=3e-3):
		super(ValueNetwork, self).__init__()

		self.linear1 = nn.Linear(state_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)

		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

	def forward(self, state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x


class SoftQNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
		super(SoftQNetwork, self).__init__()

		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x


class PolicyNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
		super(PolicyNetwork, self).__init__()

		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)

		self.mean_linear = nn.Linear(hidden_size, num_actions)
		self.mean_linear.weight.data.uniform_(-init_w, init_w)
		self.mean_linear.bias.data.uniform_(-init_w, init_w)

		self.log_std_linear = nn.Linear(hidden_size, num_actions)
		self.log_std_linear.weight.data.uniform_(-init_w, init_w)
		self.log_std_linear.bias.data.uniform_(-init_w, init_w)

	def forward(self, state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))

		mean = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

		return mean, log_std

	def evaluate(self, state, epsilon=1e-6):
		mean, log_std = self.forward(state)
		std = log_std.exp()

		normal = Normal(mean, std)
		z = normal.sample()
		action = torch.tanh(z)

		log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
		log_prob = log_prob.sum(-1, keepdim=True)

		return action, log_prob, z, mean, log_std

	def get_action(self, state):
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		mean, log_std = self.forward(state)
		std = log_std.exp()

		normal = Normal(mean, std)
		z = normal.sample()
		action = torch.tanh(z)

		action = action.detach().cpu().numpy()
		return action[0]




class SAC():
	def __init__(self, num_inputs, num_actions, hidden_size,
	             lr_policy=3e-4, lr_soft_q=3e-4, lr_value=3e-4):
		self.device = torch.device("cuda" if args.cuda else "cpu")
		self.policy = PolicyNetwork(num_inputs, num_actions, hidden_size).to(self.device)
		self.soft_q = SoftQNetwork(num_inputs, num_actions, hidden_size).to(self.device)
		self.value = ValueNetwork(num_inputs, hidden_size).to(self.device)
		self.value_target = ValueNetwork(num_inputs, hidden_size).to(self.device)

		for param_target, param in zip(self.value_target.parameters(), self.value.parameters()):
			param_target.data.copy_(param.data)

		value_criterion = nn.MSELoss()
		soft_q_criterion = nn.MSELoss()

		self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_value)
		self.soft_q_optimizer = optim.Adam(self.soft_q.parameters(), lr=lr_soft_q)
		self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)

	def soft_q_update(self, batch_size, replay,
	                  gamma=0.95,
	                  mean_lambda=1e-3,
	                  std_lambda=1e-3,
	                  z_lambda=0.0,
	                  soft_tau=1e-2,
	                  ):
		state, action, reward, next_state, done = replay.sample(batch_size)

		state = torch.FloatTensor(state).to(device)
		next_state = torch.FloatTensor(next_state).to(device)
		action = torch.FloatTensor(action).to(device)
		reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
		done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

		expected_q_value = self.soft_q(state, action)
		expected_value = self.value(state)
		new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

		target_value = self.value_target(next_state)
		next_q_value = reward + (1 - done) * gamma * target_value
		q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

		expected_new_q_value = self.soft_q(state, new_action)
		next_value = expected_new_q_value - log_prob
		value_loss = value_criterion(expected_value, next_value.detach())

		log_prob_target = expected_new_q_value - expected_value
		policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

		mean_loss = mean_lambda * mean.pow(2).mean()
		std_loss = std_lambda * log_std.pow(2).mean()
		z_loss = z_lambda * z.pow(2).sum(1).mean()

		policy_loss += mean_loss + std_loss + z_loss

		self.soft_q_optimizer.zero_grad()
		q_value_loss.backward()
		self.soft_q_optimizer.step()

		self.value_optimizer.zero_grad()
		value_loss.backward()
		self.value_optimizer.step()

		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		self.policy_optimizer.step()

		for param_target, param in zip(self.value_target.parameters(), self.value.parameters()):
			param_target.data.copy_(
				param_target.data * (1.0 - soft_tau) + param.data * soft_tau
			)

	# Save model parameters
	def save_model(self, env_name, suffix="", actor_path=None, soft_q_path=None, target=None):
		if not os.path.exists('models/'):
			os.makedirs('models/')

		if actor_path is None:
			actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
		if critic_path is None:
			critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
		print('Saving models to {} and {}'.format(actor_path, soft_q_path))
		torch.save(self.policy.state_dict(), actor_path)
		torch.save(self.soft_q.state_dict(), soft_q_path)

	# Load model parameters
	def load_model(self, actor_path, soft_q_path):
		print('Loading models from {} and {}'.format(actor_path, soft_q_path))
		if actor_path is not None:
			self.policy.load_state_dict(torch.load(actor_path))
		if soft_q_path is not None:
			self.soft_q.load_state_dict(torch.load(soft_q_path))

env = NormalizedActions(gym.make("gym_STEMsim:STEMsim-beamcenter-v5"))
env._max_episode_steps = 200

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
hidden_dim = 128





replay_buffer_size = 1000
replay_buffer = ReplayBuffer(replay_buffer_size)

max_frames  = 40000
max_steps   = 201
frame_idx   = 0
rewards     = []
batch_size  = 128
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

max_frames  = 40000

PATH = "./SAC-PT_bc-v5/"

num_success = 0
epochs = 0
while frame_idx < max_frames:
	state = env.reset()
	episode_reward = 0

	for step in range(max_steps):
		if frame_idx > 1000:
			action = policy_net.get_action(state).detach()
			next_state, reward, done, _ = env.step(action.numpy())
		else:
			action = env.action_space.sample()
			next_state, reward, done, _ = env.step(action)
		# action = policy_net.get_action(state)
		# next_state, reward, done, _ = env.step(action)

		replay_buffer.push(state, action, reward, next_state, done)
		if len(replay_buffer) > batch_size:
			soft_q_update(batch_size)
			epochs += 1

		state = next_state
		episode_reward += reward
		frame_idx += 1

		if frame_idx % 1000 == 0:
			plot(frame_idx, rewards)
		if step > env._max_episode_steps:
			done = True
		if done:
			break

	rewards.append(episode_reward)

	# save model if consecutive rewards for 50 frames > 90
	if len(rewards) > 51:
		avg_50 = sum(rewards[-50:]) / 50
		print(avg_50)
		if avg_50 > 65:
			print("model saved, successful training")
			break