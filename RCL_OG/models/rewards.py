import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, cho_solve

class get_reward(object):
    def __init__(self, num_variables, data, score_type='BIC', reg_type='LR', alpha=1.0, med_w=1.0, median_flag=False):
        self.num_variables = num_variables
        self.alpha = alpha
        self.med_w = med_w
        self.med_w_flag = median_flag
        self.d = {}
        self.d_RSS = [{} for _ in range(num_variables)]
        self.data = data.astype(np.float32)
        self.num_samples = data.shape[0]
        if score_type not in ('BIC', 'BIC_different_var'):
            raise ValueError('Reward type not supported.')
        if reg_type not in ('LR', 'QR', 'GPR', 'GPR_learnable'):
            raise ValueError('Reg type not supported')
        self.score_type = score_type
        self.reg_type = reg_type
        self.bic_penalty = np.log(data.shape[0])/data.shape[0]

        if self.reg_type == 'LR':
            self.ones = np.ones((data.shape[0], 1), dtype=np.float32)
            X = np.hstack((self.data, self.ones))
            self.X = X
            self.XtX = X.T.dot(X)
    #calculate rewards for graphs
    def cal_rewards(self, graphs, orders):
        rewards = []
        for graph, order in zip(graphs, orders):
            reward = self.cal_single_graph_reward(graph, order)
            rewards.append(reward)
        return rewards
    #calculate reward for one graph
    def cal_single_graph_reward(self, graph, order):
        graph_to_int = list(np.int32(order))
        graph_batch_to_tuple = tuple(graph_to_int)
        if graph_batch_to_tuple in self.d:
            score = self.d[graph_batch_to_tuple]
            return score
        RSS_ls = []
        for i in range(self.num_variables):
            RSSi = self.cal_RSSi(i, graph[:, i])
            RSS_ls.append(RSSi)
        RSS_ls = np.array(RSS_ls)
        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls) / self.num_samples + 1e-8) + np.sum(graph) * self.bic_penalty / self.maxlen
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls) / self.num_samples + 1e-8)) + np.sum(graph)*self.bic_penalty
        self.d[graph_batch_to_tuple] = -BIC
        return -BIC
    #calculate the partial score of graph
    def cal_local_rewards(self, targets, sources, prune_graph):
        RSSi_ls = []
        targets = np.where(targets == 1)
        for i in range(len(targets[2])):
            sources[i][0] = sources[i][0] * prune_graph[:, targets[2][i]]
            RSSi = self.cal_RSSi(targets[2][i], sources[i][0])
            RSSi_ls.append(RSSi)
        rewards = -np.log(np.array(RSSi_ls, dtype=np.float32)/self.num_samples+1e-8) - np.sum(sources[i][0])*self.bic_penalty
        return rewards.reshape(rewards.shape[0], 1)
    # calculate RSSi to calculate the partial score of graph
    def cal_RSSi(self, target_node, source_col):
        str_col = str(source_col)
        if str_col in self.d_RSS[target_node]:
            RSSi = self.d_RSS[target_node][str_col]
            return RSSi
        if np.sum(source_col) < 0.1:
            y_err = self.data[:, target_node]
            y_err = y_err - np.mean(y_err)
        else:
            cols_TrueFalse = source_col > 0.5
            if self.reg_type == 'LR':
                cols_TrueFalse = np.append(cols_TrueFalse, True)
                X_train = self.X[:, cols_TrueFalse]
                y_train = self.X[:, target_node]
                XtX = self.XtX[:, cols_TrueFalse][cols_TrueFalse, :]
                Xty = self.XtX[:, target_node][cols_TrueFalse]
                y_err = self.calculate_yerr(X_train, y_train, XtX, Xty)
            elif self.reg_type == 'GPR':
                X_train = self.data[:, cols_TrueFalse]
                y_train = self.data[:, target_node]
                p_eu = pdist(X_train, 'sqeuclidean')
                if self.med_w_flag:
                    self.med_w = np.median(p_eu)
                train_y = np.asarray(y_train)
                p_eu_nor = p_eu / self.med_w
                K = np.exp(-0.5 * p_eu_nor)
                K = squareform(K)
                np.fill_diagonal(K, 1)
                K_trans = K.copy()
                K[np.diag_indices_from(K)] += self.alpha
                L_ = cholesky(K, lower=True)
                alpha_ = cho_solve((L_, True),train_y)
                y_mean = K_trans.dot(alpha_)
                y_err = y_train - y_mean

        RSSi = np.sum(np.square(y_err))
        self.d_RSS[target_node][str_col] = RSSi
        return RSSi

    def calculate_yerr(self, X_train, y_train, XtX, Xty):
        if self.reg_type == 'LR':
            return self.calculate_LR(X_train, y_train, XtX, Xty)

    def calculate_LR(self, X_train, y_train, XtX, Xty):
        theta = np.linalg.solve(XtX, Xty)
        y_pre = X_train.dot(theta)
        y_err = y_pre - y_train
        return y_err