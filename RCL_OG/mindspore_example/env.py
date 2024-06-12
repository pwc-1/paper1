import copy

import numpy as np
import torch

from models.rewards import get_reward
from scores.bge_score import BGeScore


class causalenv:
    def __init__(self, config, data, prune_graph):
        self.data = data
        self.config = config
        self.prune_graph = prune_graph
        if self.config.score_type == 'BIC_different_var' or self.config.score_type == 'BIC':
            self.reward = get_reward(config.num_variables, data, config.score_type, config.reg_type)
        elif self.config.score_type == 'BGe':
            self.reward = BGeScore(data)
    #choice action for some state
    def choice_action(self, state, q_value, greedy):
        state = np.squeeze(state)
        e = np.random.binomial(1, greedy)
        zero = -9e15 * np.ones_like(state)
        if e == 0:
            p = q_value.numpy()
            max_p = np.max(p, -1).reshape(state.shape[0], 1)
            #p = p - max_p
            p = np.where(p == -9e15, p, p-max_p)
        else:
            one = np.ones_like(state)
            p = np.where(state == 1, zero, one)
        p = (np.exp(p)/np.sum(np.exp(p), -1, keepdims=True))
        actions = np.arange(0, self.config.num_variables)
        action = [np.random.choice(actions, size=1, replace=True, p=p[i]) for i in range(state.shape[0])]
        return np.eye(self.config.num_variables, dtype=np.float32)[action]
    #choice action with the biggest probability
    def choice_max_action(self, q_value):
        q_value = q_value.numpy()
        actions = np.argmax(q_value, -1)
        return np.eye(self.config.num_variables, dtype=np.float32)[actions]
    #get next_state by state and action
    def next_step(self, states, actions):
        next_states = states + actions
        if self.config.score_type == 'BIC_different_var' or self.config.score_type == 'BIC':
            rewards = self.reward.cal_local_rewards(actions, copy.copy(states), self.prune_graph)
        elif self.config.score_type == 'BGe':
            rewards = self.reward.cal_local_rewards(actions, copy.copy(states), self.prune_graph)
        return rewards, next_states

        