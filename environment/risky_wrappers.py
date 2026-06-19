import random

import gym


class RewardHighVelocity(gym.Wrapper):
    def __init__(self, env, max_vel=3.0, prob_vel_penal=0.05, cost_vel=-20.0):
        super().__init__(env)
        self.max_vel = max_vel
        self.prob_vel_penal = prob_vel_penal
        self.cost_vel = cost_vel

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        info = dict(info)
        info["risky_state"] = False

        env_id = self.spec.id.lower()
        if "hopper" in env_id:
            velocity = state[5]
        elif "walker2d" in env_id:
            velocity = state[8]
        elif "halfcheetah" in env_id:
            velocity = state[8]
        else:
            raise ValueError(f"Unsupported env for velocity wrapper: {self.spec.id}")

        if abs(velocity) > self.max_vel:
            info["risky_state"] = True
            if random.random() < self.prob_vel_penal:
                reward += self.cost_vel

        return state, reward, done, info


class RewardUnhealthyPose(gym.Wrapper):
    def __init__(
        self,
        env,
        prob_pose_penal=0.1,
        cost_pose=-50.0,
        healthy_angle_range=(-0.1, 0.1),
        done_if_exceed_factor=2.0,
    ):
        super().__init__(env)
        self.prob_pose_penal = prob_pose_penal
        self.cost_pose = cost_pose
        self.healthy_angle_range = healthy_angle_range
        self.done_if_exceed_factor = done_if_exceed_factor

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        info = dict(info)
        info["risky_state"] = False

        pitch_angle = state[1]
        low, high = self.healthy_angle_range
        if pitch_angle < low or pitch_angle > high:
            info["risky_state"] = True
            if random.random() < self.prob_pose_penal:
                reward += self.cost_pose

        if abs(pitch_angle) > self.done_if_exceed_factor * abs(high):
            done = True

        return state, reward, done, info


def make_risky_env(
    env,
    max_vel=3.0,
    prob_vel_penal=0.05,
    cost_vel=-20.0,
    prob_pose_penal=0.1,
    cost_pose=-50.0,
    healthy_angle_range=(-0.1, 0.1),
    done_if_exceed_factor=2.0,
):
    env_id = env.spec.id.lower()

    if "halfcheetah" in env_id:
        return RewardHighVelocity(
            env,
            max_vel=max_vel,
            prob_vel_penal=prob_vel_penal,
            cost_vel=cost_vel,
        )

    if "hopper" in env_id or "walker2d" in env_id:
        return RewardUnhealthyPose(
            env,
            prob_pose_penal=prob_pose_penal,
            cost_pose=cost_pose,
            healthy_angle_range=tuple(healthy_angle_range),
            done_if_exceed_factor=done_if_exceed_factor,
        )

    raise ValueError(f"Unsupported env for risky wrapper: {env.spec.id}")
