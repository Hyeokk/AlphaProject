import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import deque
from torch.distributions import Normal

# TensorBoard 선택적 import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class RunningMeanStd:
    """보상 정규화를 위한 실행 평균/표준편차 계산"""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class PPOActor(nn.Module):
    """PPO Actor 네트워크"""
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # 초기화
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0)

    def forward(self, x):
        features = self.network(x)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

    def get_action_and_log_prob(self, x):
        mean, std = self.forward(x)
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_log_prob(self, x, action):
        mean, std = self.forward(x)
        dist = Normal(mean, std)
        raw_action = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        return log_prob


class PPOCritic(nn.Module):
    """PPO Critic 네트워크"""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        for layer in self.network[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.constant_(self.network[-1].bias, 0)

    def forward(self, x):
        return self.network(x)


class PPOBuffer:
    """PPO 경험 버퍼"""
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.clear()

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones, self.next_states = [], [], [], []
        self.current_episode_length = 0

    def add(self, state, action, reward, value, log_prob, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.current_episode_length += 1

        if done and self.current_episode_length <= 3:
            print(f"[BUFFER] 극단적으로 짧은 에피소드 ({self.current_episode_length}스텝) 제거")
            self.remove_last_episode()
            return True
        return False

    def remove_last_episode(self):
        for _ in range(self.current_episode_length):
            if self.states:
                self.states.pop()
                self.actions.pop()
                self.rewards.pop()
                self.values.pop()
                self.log_probs.pop()
                self.dones.pop()
                self.next_states.pop()
        self.current_episode_length = 0

    def episode_reset(self):
        self.current_episode_length = 0

    def get(self):
        return (self.states, self.actions, self.rewards,
                self.values, self.log_probs, self.dones, self.next_states)

    def size(self):
        return len(self.states)


class PPOAgentVector:
    """CSV 기반 벡터 관측을 위한 PPO 에이전트"""
    def __init__(self, obs_dim, action_dim, action_bounds, log_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.gamma, self.lam = 0.995, 0.95
        self.clip_epsilon, self.c1, self.c2 = 0.2, 0.5, 0.01
        self.ppo_epochs, self.mini_batch_size = 10, 64
        self.max_grad_norm = 0.5

        self.actor = PPOActor(obs_dim, action_dim).to(self.device)
        self.critic = PPOCritic(obs_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.buffer = PPOBuffer()
        self.action_bounds = action_bounds

        self.reward_rms = RunningMeanStd()
        self.normalize_rewards = True

        if log_dir and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=log_dir)
            self.log_enabled = True
        else:
            self.writer, self.log_enabled = None, False

    def preprocess_obs(self, obs, is_batch=False):
        if is_batch:
            return torch.FloatTensor(np.array(obs)).to(self.device)
        else:
            return torch.FloatTensor(obs).unsqueeze(0).to(self.device)

    def get_action(self, obs, training=True):
        feature = self.preprocess_obs(obs)
        with torch.no_grad():
            if training:
                action, log_prob = self.actor.get_action_and_log_prob(feature)
                value = self.critic(feature)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]
                value = value.cpu().numpy()[0]
            else:
                mean, _ = self.actor(feature)
                action = torch.tanh(mean).cpu().numpy()[0]
                log_prob, value = None, None
        scaled_action = self._scale_action(action)
        return (scaled_action, log_prob, value) if training else scaled_action

    def _scale_action(self, action):
        scaled = []
        for i in range(len(action)):
            low, high = self.action_bounds[i]
            scaled_val = (action[i] + 1) / 2 * (high - low) + low
            scaled.append(np.clip(scaled_val, low, high))
        return np.array(scaled, dtype=np.float32)

    def _unscale_action(self, scaled_action):
        unscaled = []
        for i in range(len(scaled_action)):
            low, high = self.action_bounds[i]
            val = 2 * (scaled_action[i] - low) / (high - low) - 1
            unscaled.append(np.clip(val, -1, 1))
        return np.array(unscaled, dtype=np.float32)

    def store(self, obs, action, reward, value, log_prob, done, next_obs):
        if self.normalize_rewards:
            self.reward_rms.update([reward])
            reward = reward / np.sqrt(self.reward_rms.var + 1e-8)
            reward = np.clip(reward, -10, 10)
        unscaled_action = self._unscale_action(action)
        self.buffer.add(obs, unscaled_action, reward, value, log_prob, done, next_obs)

    def compute_gae(self, rewards, values, dones, next_values):
        advs, gae = [], 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * (next_values[i]) * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advs.insert(0, gae)
        returns = [a + v for a, v in zip(advs, values)]
        return advs, returns

    def train(self):
        if self.buffer.size() < self.mini_batch_size:
            return {}
        states, actions, rewards, values, old_log_probs, dones, _ = self.buffer.get()

        next_values = [0.0] * len(states)
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)
        advantages = (np.array(advantages) - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        states_tensor = self.preprocess_obs(states, is_batch=True)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device).unsqueeze(1)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        returns_tensor = torch.FloatTensor(returns).to(self.device).unsqueeze(1)

        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.mini_batch_size):
                end = min(start+self.mini_batch_size, len(states))
                mb_idx = indices[start:end]

                mb_states = states_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_log_probs = old_log_probs_tensor[mb_idx]
                mb_adv = advantages_tensor[mb_idx]
                mb_ret = returns_tensor[mb_idx]

                new_log_probs = self.actor.get_log_prob(mb_states, mb_actions)
                new_values = self.critic(mb_states)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, mb_ret)

                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

        self.buffer.clear()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def log_episode_metrics(self, episode, reward, length, total_steps):
        if self.log_enabled:
            self.writer.add_scalar('Episode/Reward', reward, episode)
            self.writer.add_scalar('Episode/Length', length, episode)
            self.writer.add_scalar('Episode/Total_Steps', total_steps, episode)

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(save_path, "PPO_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(save_path, "PPO_critic.pt"))
        print(f"모델 저장 완료: {save_path}")

    def load_model(self, load_path):
        self.actor.load_state_dict(torch.load(os.path.join(load_path, "PPO_actor.pt"), weights_only=True))
        self.critic.load_state_dict(torch.load(os.path.join(load_path, "PPO_critic.pt"), weights_only=True))
        print(f"모델 로드 완료: {load_path}")

    def close(self):
        if self.writer:
            self.writer.close()
