import numpy as np
import mindspore as ms
from mindspore import nn, ops
from tqdm import tqdm
from mindspore_model.q_net import q_net
from models.rewards import get_reward
from utils.analyze_utils import from_order_to_graph, graph_prunned_by_coef, pruning_cam, count_sid, count_shdcp
from mindspore_example.env import causalenv
from utils.plot_dag import GraphDAG
from utils.replaybuffer import replaybuffer
from gendata.evaluation import MetricsDAG

class DAG_RL:
    def __init__(self, config, data, prune_graph):
        super(DAG_RL, self).__init__()
        self.config = config
        self.greedy = config.greedy
        self.initial_greedy = config.greedy
        self.final_greedy = 0.01
        self.init_state = np.zeros(config.num_variables)
        self.init_mask = np.ones(config.num_variables)
        self.q_nets = [q_net(config.num_variables, config.num_samples, config.embed_dim, config.hidden_dim, config.heads,
                           config.dropout_rate) for _ in range(config.num_variables)]
        self.optimizers = [nn.Adam(self.q_nets[i].trainable_params(), learning_rate=self.config.learning_rate) for i in range(config.num_variables)]
        self.data = data
        self.prune_graph = prune_graph
        self.env = causalenv(self.config, data, prune_graph)
        self.replays = [replaybuffer(self.config.num_variables, self.config.capacity) for _ in range(config.num_variables)]
    #run algorithm
    def learn(self):

        def loss_fn(states, actions, rewards, next_states, data_batch, layer):
            q_value = self.q_nets[layer](data_batch, states)
            source = ops.sum(ops.mul(q_value, ops.squeeze(actions)), -1)
            rewards = np.array(rewards.reshape(self.config.batch_size).copy(), dtype=np.float32)
            if layer == self.config.num_variables - 1:
                target_rewards = rewards
            else:
                next_q_values = self.q_nets[layer + 1](data_batch, next_states)
                target_rewards = rewards + np.log(np.sum(np.exp(next_q_values.numpy().astype(np.float64)), -1) + 1e-8)
            targets = ms.Tensor(target_rewards, dtype=ms.float32)
            loss = ms.nn.MSELoss(reduction='mean')(targets, source)
            return loss
        grad_fns = [ms.value_and_grad(loss_fn, None, self.optimizers[i].parameters) for i in range(self.config.num_variables)]
        def train_step(samples, data_batch, layer):
            loss, grads = grad_fns[layer](ms.Tensor(samples['state']), ms.Tensor(samples['action']), ms.Tensor(samples['reward']), ms.Tensor(samples['next_state']), ms.Tensor(data_batch), layer)
            self.optimizers[layer](grads)
            return loss

        data_batch = self.get_batch(self.data.transpose(), self.config.batch_size)
        states = self.get_batch([self.init_state], self.config.batch_size)
        layer = 0
        for i in tqdm(range(self.config.epoch)):
            q_values = self.q_nets[layer](ms.Tensor(data_batch), ms.Tensor(states))
            actions = self.env.choice_action(states, q_values, self.greedy)
            self.greedy -= (self.initial_greedy-self.final_greedy)/self.config.epoch
            rewards, next_states = self.env.next_step(states, actions)
            self.replays[layer].add(states, actions, rewards, next_states)
            if self.replays[layer].length() >= self.config.batch_size:
                samples = self.replays[layer].sample(self.config.batch_size)

                loss = train_step(samples, data_batch, layer)

            if (i+1) % self.config.num_variables == 0:
                layer = 0
                states = self.get_batch([self.init_state], self.config.batch_size)
            else:
                layer += 1
                states = next_states
    #calculate loss to updata
    def cal_loss(self, samples, data_batch, layer):
        q_value = self.q_nets[layer](ms.Tensor(data_batch), ms.Tensor(samples['state']))
        source = ops.sum(ops.mul(q_value, ms.Tensor(np.squeeze(samples['action']))), -1)
        rewards = np.array(samples['reward'].reshape(self.config.batch_size).copy(), dtype=np.float32)
        if layer == self.config.num_variables-1:
            target_rewards = rewards
        else:
            next_q_values = self.q_nets[layer+1](ms.Tensor(data_batch), ms.Tensor(samples['next_state']))
            target_rewards = rewards + np.log(np.sum(np.exp(next_q_values.numpy().astype(np.float64)), -1) + 1e-8)
        targets = ms.Tensor(target_rewards, dtype=ms.float32)
        loss = ms.nn.MSELoss(reduction='mean')(targets, source)
        return loss
    #sample orderings by probability model
    def sample(self, sample_size):
        data_batch = self.get_batch(self.data.transpose(), sample_size)
        states = self.get_batch([self.init_state], sample_size)
        for i in range(self.config.num_variables):
            q_values = self.q_nets[i](ms.Tensor(data_batch), ms.Tensor(states))
            actions = self.env.choice_action(states, q_values, 0)
            if i == 0:
                positions = np.where(actions == 1)[2].reshape(sample_size, 1)
                p = q_values.numpy()

                max_p = np.max(p, -1).reshape(states.shape[0], 1)
                p = np.where(p == -9e15, p, p - max_p)

                p = np.exp(p) / np.sum(np.exp(p), -1, keepdims=True)
                p = np.log(np.sum(np.multiply(p, np.squeeze(actions)), -1) + 1e-8)
            else:
                positions = np.concatenate([positions, np.where(actions == 1)[2].reshape(sample_size, 1)], axis=1)
                pi = q_values.numpy()

                max_p = np.max(pi, -1).reshape(states.shape[0], 1)
                pi = np.where(pi == -9e15, pi, pi - max_p)

                pi = np.exp(pi) / np.sum(np.exp(pi), -1, keepdims=True)
                pi = np.log(np.sum(np.multiply(pi, np.squeeze(actions)), -1) + 1e-8)
                p = p + pi
            states = states + actions
        return positions, p
    #sample states and actions by probability model
    def sample_state_action(self, sample_size):
        samples = {
            'states': [],
            'actions': []
        }
        data_batch = self.get_batch(self.data.transpose(), sample_size)
        states = self.get_batch([self.init_state], sample_size)
        for i in range(self.config.num_variables):
            q_values = self.q_nets[i](ms.Tensor(data_batch), ms.Tensor(states))
            actions = self.env.choice_action(states, q_values, 0)
            if i == 0:
                positions = np.where(actions == 1)[2].reshape(sample_size, 1)
            else:
                positions = np.concatenate([positions, np.where(actions == 1)[2].reshape(sample_size, 1)], axis=1)
            if i>0:
                samples['states'] += states.tolist()
                samples['actions'] += actions.tolist()
            states = states + actions
        index = np.random.choice(sample_size*(self.config.num_variables - 1), sample_size, replace=False)
        return {'states': np.array(samples['states'])[index], 'actions': np.array(samples['actions'])[index]}
    #get result with biggest probability
    def max_sample_matrix(self, true_dag):
        data_batch = self.get_batch(self.data.transpose(), 1)
        states = self.get_batch([self.init_state], 1)
        for i in range(self.config.num_variables):
            q_values = self.q_nets[i](ms.Tensor(data_batch), ms.Tensor(states))
            action = self.env.choice_max_action(q_values)
            if i == 0:
                positions = np.where(action == 1)[1].reshape(1, 1)
            else:
                positions = np.concatenate([positions, np.where(action == 1)[1].reshape(1, 1)], axis=1)
            states = states + action
        mets = {
            'fdr': 0,
            'tpr': 0,
            'fpr': 0,
            'shd': 0,
            'precision': 0,
            'F1': 0,
        }
        if self.config.reg_type == 'LR':
            graph_pruned = np.array(graph_prunned_by_coef(from_order_to_graph(positions[0]), self.data, th=0.3, if_normal=False)).T
        elif self.config.reg_type == 'GPR':
            graph_pruned = pruning_cam(self.data, from_order_to_graph(positions[0]))
        # total_edge = float(np.sum(np.reshape(graph_pruned, (graph_pruned.size,))))
        # correct_edge = float(np.sum(np.reshape(np.multiply(graph_pruned, true_dag), (np.multiply(graph_pruned, true_dag).size,))))
        # sid = count_sid(true_dag, graph_pruned)
        # cpshd = count_shdcp(true_dag, graph_pruned)
        met = MetricsDAG(graph_pruned, true_dag)
        GraphDAG(graph_pruned, true_dag)
        for name in mets.keys():
            mets[name] = met.metrics[name]
        # mets['total_edge'] = total_edge
        # mets['correct_edge'] = correct_edge
        # mets['sid'] = sid
        # mets['cpshd'] = cpshd
        return mets
    #get transition probability of state-action by probability model
    def get_probability(self, state, action):
        layer = int(np.sum(state))
        q_value = self.q_nets[layer](ms.Tensor(self.get_batch(self.data.transpose(), 1)), ms.Tensor(self.get_batch([state], 1)))
        p = q_value.detach().numpy()
        # max_p = np.max(p, -1).reshape(state.shape[0], 1)
        # # p = p - max_p
        # p = np.where(p == -9e15, p, p - max_p)
        p = (np.exp(p) / np.sum(np.exp(p), -1, keepdims=True))
        p = np.sum(np.multiply(p, action))
        return p
    #process data to a batch data
    def get_batch(self, data, batch_size):
        batch = []
        for _ in range(batch_size):
            batch.append(data)
        return np.array(batch, dtype=np.float32)
    #get mean results
    def get_e_matrix(self, samples, logp, true_dag):
        mets = {
            'fdr': 0,
            'tpr': 0,
            'fpr': 0,
            'shd': 0,
            'precision': 0,
            'F1': 0,
        }
        # total_edges = 0
        # correct_edges = 0
        # sids = 0
        # cpshds = 0
        for i in range(samples.shape[0]):
            for j in range(i, samples.shape[0]):
                if logp[i]<logp[j]:
                    a = logp[i]
                    b = samples[i]
                    logp[i] = logp[j]
                    samples[i] = samples[j]
                    logp[j] = a
                    samples[j] = b
        for i in range(20):
            if self.config.reg_type == 'LR':
                graph_pruned = np.array(graph_prunned_by_coef(from_order_to_graph(samples[i]), self.data, th=0.3, if_normal=False)).T
            elif self.config.reg_type == 'GPR':
                graph_pruned = pruning_cam(self.data, from_order_to_graph(samples[i]))
            # total_edge = float(np.sum(np.reshape(graph_pruned, (graph_pruned.size,))))
            # correct_edge = float(np.sum(np.reshape(np.multiply(graph_pruned, true_dag), (np.multiply(graph_pruned, true_dag).size,))))
            # sid = count_sid(true_dag, graph_pruned)
            # cpshd = count_shdcp(true_dag, graph_pruned)
            met = MetricsDAG(graph_pruned, true_dag)
            for name in mets.keys():
                mets[name] += met.metrics[name]
            # total_edges += total_edge
            # correct_edges += correct_edge
            # sids += sid
            # cpshds += cpshd
        for name in mets.keys():
            mets[name] /= 20
        # mets['total_edge'] = total_edges/20
        # mets['correct_edge'] = correct_edges/20
        # mets['sid'] = sids/20
        # mets['cpshd'] = cpshds/20
        return mets
    #get result with the best score
    def get_max_matrix(self, samples, true_dag):
        max_reward = float('-inf')
        max_index = 0
        mets = {
            'fdr': 0,
            'tpr': 0,
            'fpr': 0,
            'shd': 0,
            'precision': 0,
            'F1': 0,
        }
        _matrixs = []
        for i in range(samples.shape[0]):
            _matrix = from_order_to_graph(samples[i]) * self.prune_graph
            _matrixs.append(_matrix)
        graphs = np.stack(_matrixs)
        rewards = self.env.reward.cal_rewards(graphs, samples)
        rewards = np.array(rewards)
        for i in range(samples.shape[0]):
            if rewards[i] > max_reward:
                max_reward = rewards[i]
                max_index = i
        if self.config.reg_type == 'LR':
            graph = np.array(graph_prunned_by_coef(from_order_to_graph(samples[max_index]), self.data, th=0.3, if_normal=False)).T
        elif self.config.reg_type == 'GPR':
            graph = pruning_cam(self.data, from_order_to_graph(samples[max_index]))
        # total_edge = float(np.sum(np.reshape(graph, (graph.size,))))
        # correct_edge = float(np.sum(np.reshape(np.multiply(graph, true_dag), (np.multiply(graph, true_dag).size,))))
        # sid = count_sid(true_dag, graph)
        # cpshd = count_shdcp(true_dag, graph)
        met = MetricsDAG(graph, true_dag)
        for name in mets.keys():
            mets[name] = met.metrics[name]
        # mets['total_edge'] = total_edge
        # mets['correct_edge'] = correct_edge
        # mets['sid'] = sid
        # mets['cpshd'] = cpshd
        return mets
    #sample results
    def sample_num(self, true_dag):
        if_end = True
        max_tpr = 0
        max_tpr_samples = 0
        sample_num = 0
        while if_end and sample_num <= 20000:
            positions, logp = self.sample(200)
            for i in range(200):
                sample_num += 1
                if self.config.reg_type == 'LR':
                    graph_pruned = np.array(graph_prunned_by_coef(from_order_to_graph(positions[i]), self.data, th=0.3, if_normal=False)).T
                elif self.config.reg_type == 'GPR':
                    graph_pruned = pruning_cam(self.data, from_order_to_graph(positions[i]))
                met = MetricsDAG(graph_pruned, true_dag)
                if met.metrics['tpr'] > max_tpr:
                    max_tpr = met.metrics['tpr']
                    max_tpr_samples = sample_num
                if met.metrics['tpr'] == 1:
                    max_tpr = met.metrics['tpr']
                    max_tpr_samples = sample_num
                    if_end = False
                    break
            print('finish 200 sampling')
        return sample_num, max_tpr, max_tpr_samples




