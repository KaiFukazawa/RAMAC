import torch

class Data_Sampler(object):
    """Lightweight tensor-based sampler for offline RL datasets.

    Wraps numpy-based replay data into torch tensors on a given device and
    provides IID sampling compatible with training loops.
    """
    def __init__(self, data, device, reward_tune='no'):
        """Initialize tensors from dataset dict and optional reward tuning.

        Args:
            data (dict): Keys include 'observations', 'actions',
                'next_observations', 'rewards', 'terminals'.
            device (str | torch.device): Target device for sampled batches.
            reward_tune (str): Optional reward normalization scheme.
        """
        # Basic columns
        self.state = torch.from_numpy(data['observations']).float()
        self.action = torch.from_numpy(data['actions']).float()
        self.next_state = torch.from_numpy(data['next_observations']).float()
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
        self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

        # optional reward tuning
        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0

        self.reward = reward

    def sample(self, batch_size):
        """Sample a minibatch of transitions as torch tensors on device.

        Args:
            batch_size (int): Number of IID samples to draw.

        Returns:
            tuple[torch.Tensor, ...]: (state, action, next_state, reward, not_done)
        """
        ind = torch.randint(0, self.size, (batch_size,))
        return (
            self.state[ind].to(self.device),        # s
            self.action[ind].to(self.device),       # a
            self.next_state[ind].to(self.device),   # s′   
            self.reward[ind].to(self.device),       # r　
            self.not_done[ind].to(self.device)      # d
        )


def iql_normalize(reward, not_done):
    """Normalize rewards per-trajectory as in IQL for locomotion tasks.

    Args:
        reward (torch.Tensor): Rewards column vector.
        not_done (torch.Tensor): 1 - done flags to delimit trajectories.

    Returns:
        torch.Tensor: Scaled reward tensor.
    """
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    if len(trajs_rt) < 2:
        return reward
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
    denom = (rt_max - rt_min).clamp(min=1e-7)
    reward /= denom
    reward *= 1000.
    return reward
