import math

import numpy as np
import copy
from models.rewards import get_reward
from utils.analyze_utils import from_order_to_graph


class probability:
    def __init__(self, data, config, prune_graph):
        self.data = data
        self.config = config
        self.prune_graph = prune_graph
        self.lists = []
        self.reward = get_reward(config.num_variables, data, config.score_type, config.reg_type)
        self.get_all_score()
    #get all score for all orderings
    def get_all_score(self):
        list = []
        for i in range(self.config.num_variables):
            list.append(i)
        lists = []
        self.perm(0, self.config.num_variables, list, lists)
        _matrixs = []
        for i in range(len(lists)):
            _matrix = from_order_to_graph(lists[i])
            _matrixs.append(_matrix * self.prune_graph)
        graphs = np.stack(_matrixs)
        self.reward.cal_rewards(graphs, lists)
    #get all orderings by given variables
    def perm(self, cur, right, list, lists):
        if cur == right:
            list = copy.deepcopy(list)
            lists.append(list)
            return
        else:
            for i in range(cur, right):
                list = self.swap(i, cur, list)
                self.perm(cur+1, right, list, lists)
                list = self.swap(i, cur, list)
    # swap variables
    def swap(self, i, j, list):
        a = list[i]
        list[i] = list[j]
        list[j] = a
        return list
    #get exact probability of state-action
    def get_probability(self, state, action):
        start_variables = [i for i, x in enumerate(state) if x == 1]
        start_lists = []
        self.perm(0, len(start_variables), start_variables, start_lists)

        middle_variable = action.index(1)
        end_variables = [i for i, x in enumerate(state) if x == 0]
        end_variables.remove(middle_variable)
        end_lists = []
        self.perm(0, len(end_variables), end_variables, end_lists)
        rest_variables = [i for i, x in enumerate(state) if x == 0]
        rest_list = []
        self.perm(0, len(rest_variables), rest_variables, rest_list)

        all_lists = []
        for i in start_lists:
            for j in rest_list:
                all_lists.append(i + j)

        lists = []
        for i in start_lists:
            i.append(middle_variable)
            for j in end_lists:
                lists.append(i+j)

        reward = 0
        for i in lists:
            graph_to_int = list(np.int32(i))
            graph_batch_to_tuple = tuple(graph_to_int)
            reward += math.exp(self.reward.d[graph_batch_to_tuple])

        all_reward = 0
        for i in all_lists:
            graph_to_int = list(np.int32(i))
            graph_batch_to_tuple = tuple(graph_to_int)
            all_reward += math.exp(self.reward.d[graph_batch_to_tuple])
        return reward/all_reward
    #get partial reward of state
    def get_reward(self, state, prune_graph):
        end_variables = [i for i, x in enumerate(state) if x == 0]
        reward = 0
        if len(end_variables) == 1:
            source = state * prune_graph[:, end_variables[0]]
            RSSi = self.reward.d_RSS[end_variables[0]][str(source)]
            R = -np.log(np.array(RSSi, dtype=np.float32) / self.config.num_samples + 1e-8)
            return R
        for i in end_variables:
            target_node = i
            source = str(state * prune_graph[:, i])
            RSSi = self.reward.d_RSS[target_node][source]
            R =  -np.log(np.array(RSSi, dtype=np.float32)/self.config.num_samples+1e-8)
            state[i] = 1
            reward += np.exp(R + self.get_reward(state, prune_graph))
            state[i] = 0
        return np.log(reward)
