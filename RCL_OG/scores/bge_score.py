import math
import numpy as np

from scipy.special import gammaln

from scores.base import BaseScore


def logdet(array):
    _, logdet = np.linalg.slogdet(array)
    return logdet


class BGeScore(BaseScore):
    def __init__(
            self,
            data,
            mean_obs=None,
            alpha_mu=1.,
            alpha_w=None
        ):
        num_variables = data.shape[1]
        self.d = {}
        self.d_RSS = [{} for _ in range(num_variables)]
        if mean_obs is None:
            mean_obs = np.zeros((num_variables,))
        if alpha_w is None:
            alpha_w = num_variables + 2.

        super().__init__(data)
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w

        self.num_samples = self.data.shape[0]
        self.t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)

        T = self.t * np.eye(self.num_variables)
        data = np.asarray(self.data)
        data_mean = np.mean(data, axis=0, keepdims=True)
        data_centered = data - data_mean

        self.R = (T + np.dot(data_centered.T, data_centered)
            + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
            * np.dot((data_mean - self.mean_obs).T, data_mean - self.mean_obs)
        )
        all_parents = np.arange(self.num_variables)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + gammaln(0.5 * (self.num_samples + self.alpha_w - self.num_variables + all_parents + 1))
            - gammaln(0.5 * (self.alpha_w - self.num_variables + all_parents + 1))
            - 0.5 * self.num_samples * math.log(math.pi)
            + 0.5 * (self.alpha_w - self.num_variables + 2 * all_parents + 1) * math.log(self.t)
        )

    def local_score(self, target, indices):
        num_parents = len(indices)

        if indices:
            variables = [target] + list(indices)

            log_term_r = (
                0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents)
                * logdet(self.R[np.ix_(indices, indices)])
                - 0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents + 1)
                * logdet(self.R[np.ix_(variables, variables)])
            )
        else:
            log_term_r = (-0.5 * (self.num_samples + self.alpha_w - self.num_variables + 1)
                * np.log(np.abs(self.R[target, target])))

        return self.log_gamma_term[num_parents] + log_term_r

    def cal_local_rewards(self, targets, sources, prune_graph):
        local_score_ls = []
        targets = np.where(targets == 1)
        for i in range(len(targets[2])):
            sources[i][0] = sources[i][0] * prune_graph[:, targets[2][i]]
            source = [i for i, x in enumerate(sources[i][0]) if x == 1]
            print(source)
            local_score = self.local_score(targets[2][i], source)
            local_score_ls.append(local_score)
        rewards = np.array(local_score_ls)
        return rewards.reshape(rewards.shape[0], 1)

    def cal_rewards(self, graphs, orders):
        rewards = []
        for graph, order in zip(graphs, orders):
            reward = self.cal_single_graph_reward(graph, order)
            rewards.append(reward)
        return rewards

    def cal_single_graph_reward(self, graph, order):
        graph_to_int = list(np.int32(order))
        graph_batch_to_tuple = tuple(graph_to_int)
        if graph_batch_to_tuple in self.d:
            score = self.d[graph_batch_to_tuple]
            return score
        local_score_ls = []
        for i in range(self.num_variables):
            source = [i for i, x in enumerate(graph[:, i]) if x == 1]
            local_score = self.local_score(i, source)
            print(i, source, local_score)
            local_score_ls.append(local_score)
        RSS_ls = np.array(local_score_ls)
        score = np.sum(RSS_ls)
        return score

    def get_local_scores(self, target, indices):
        local_score = self.local_score(target, indices)
        return local_score
