#!/usr/bin/python3
# dont use seed swap,directly margin-loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer,Uniform
from helper import *
import json
import logging
import os
from tqdm import tqdm

def hinge_loss(positive_score, negative_score, gamma):
    err = positive_score - negative_score + gamma
    max_err = err.clamp(0)
    return max_err


def entlist2emb(Model, entids, cuda_num, entembed):
    """
    return basic bert unit output embedding of entities
    """

    batch_emb = []
    for eid in entids:
        embed = entembed[eid]
        batch_emb.append(embed)

    return batch_emb





class KGEModel(nn.Cell):
    def __init__(self, model_name, dict_local, init, E_init, R_init, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.embed_loc = dict_local
        self.E_init = E_init
        self.R_init = R_init
        self.init = init

        self.gamma = mindspore.Parameter(
            mindspore.Tensor([gamma]),
            requires_grad=False
        )
        self.embedding_range = mindspore.Parameter(
            mindspore.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )


        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        ''' Intialize embeddings '''
        if self.init == 'crawl':
            self.entity_embedding = mindspore.Parameter(mindspore.Tensor(self.E_init))
            self.relation_embedding = mindspore.Parameter(mindspore.Tensor(self.R_init))
        else:
            self.entity_embedding = mindspore.Parameter(mindspore.ops.zeros(nentity, self.entity_dim))
            self.entity_embedding = initializer(Uniform(scale=self.embedding_range.item()),self.entity_embedding,mindspore.float32)
            self.relation_embedding = mindspore.Parameter(mindspore.ops.zeros(nrelation, self.relation_dim))
            self.relation_embedding = initializer(Uniform(scale=self.relation_embedding.item()), self.relation_embedding,mindspore.float32)

        if model_name == 'pRotatE' or model_name == 'new_rotate':
            self.modulus = mindspore.Parameter(mindspore.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE']:
            raise ValueError('model %s not supported' % model_name)

    def construct(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        if mode == 'single':
            batch_size, negative_sample_size = sample.shape[0], 1

            head = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = mindspore.ops.index_select(
                self.relation_embedding,
                axis=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.shape[0], head_part.shape[1]

            head = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = mindspore.ops.index_select(
                self.relation_embedding,
                axis=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.shape[0], tail_part.shape[1]

            head = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = mindspore.ops.index_select(
                self.relation_embedding,
                axis=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        elif mode == 'align':
            head = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            tail = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            score = head - tail
            score = mindspore.ops.norm(score, ord=1, dim=2)

            return score
        elif mode == 'head-align':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.shape[0], head_part.shape[1]

            head = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            tail = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            tail = tail.tile((1, negative_sample_size, 1))
            score = head - tail
            score = mindspore.ops.norm(score, ord=1, dim=2)

            return score
        elif mode == 'tail-align':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.shape[0], tail_part.shape[1]
            print(batch_size)
            print(negative_sample_size)
            head = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            tail = mindspore.ops.index_select(
                self.entity_embedding,
                axis=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            head = head.tile((1, negative_sample_size, 1))
            score = head - tail
            score = mindspore.ops.norm(score, ord=1, dim=2)
            return score
        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
            score = mindspore.ops.norm(score, ord=1, dim=2)
            return score

        else:
            score = (head + relation) - tail
            score = mindspore.ops.norm(score, ord=1, dim=2)
            return score

    @staticmethod
    def train_step(args, model, optimizer, train_iterator,seed_iterator1,seed_iterator2):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        negative_sample_size1 = int(args.single_negative_sample_size)
        negative_sample_size2 = int(args.cross_negative_sample_size)
        gamma1 = mindspore.ops.full((1, negative_sample_size1), float(args.single_gamma))
        positive_sample1, negative_sample1, subsampling_weight, mode1 = next(train_iterator).values()
        positive_sample2, negative_sample2, mode2 = next(seed_iterator1).values()
        mode2=str(mode2.asnumpy()[0])
        positive_sample3, negative_sample3, mode3 = next(seed_iterator2).values()
        mode3 = str(mode3.asnumpy()[0])
        gamma2 = mindspore.ops.full((1, negative_sample_size2), float(20))
        def forward_fn():
            negative_score1 = model((positive_sample1, negative_sample1), mode=mode1)
            positive_score1 = model(positive_sample1)
            positive_score1 = positive_score1.repeat(1, negative_sample_size1)
            loss1 = hinge_loss(positive_score1, negative_score1, gamma1)

            negative_score2 = model((positive_sample2, negative_sample2), mode=mode2)
            positive_score2 = model(positive_sample2, mode='align')
            positive_score2 = positive_score2.tile((1, negative_sample_size2))
            loss2pos = positive_score2
            loss2neg = hinge_loss(0, negative_score2, gamma2)

            negative_score3 = model((positive_sample3, negative_sample3), mode=mode3)
            positive_score3 = model(positive_sample3, mode='align')
            positive_score3 = positive_score3.tile((1, negative_sample_size2))
            loss3pos = positive_score3
            loss3neg = hinge_loss(0, negative_score3, gamma2)

            loss = loss1.sum()+loss2pos.sum()+0.5*loss2neg.sum()+loss3pos.sum()+0.5*loss3neg.sum()
            #loss =  loss2pos.sum() + 0.5 * loss2neg.sum() + loss3pos.sum() + 0.5 * loss3neg.sum()

            # if args.regularization != 0.0:
            #     regularization = args.regularization * (
            #             model.entity_embedding.norm(ord=2) ** 2 +
            #             model.relation_embedding.norm(ord=2) ** 2
            #     )
            #     loss = loss + regularization
            #     regularization_log = {}
            #
            # else:
            regularization_log = {}

            return loss,regularization_log

        grad_fn=mindspore.value_and_grad(forward_fn,None,optimizer.parameters,has_aux=True)
        (loss,regularization_log),grads=grad_fn()
        loss = mindspore.ops.depend(loss,optimizer(grads))
        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return log

    def get_seeds(self, p, side_info, logging):
        self.p = p
        self.side_info = side_info
        self.logging = logging
        self.id2ent, self.id2rel = self.side_info.id2ent, self.side_info.id2rel
        self.ent2id, self.rel2id = self.side_info.ent2id, self.side_info.rel2id
        self.ent2triple_id_list, self.rel2triple_id_list = self.side_info.ent2triple_id_list, self.side_info.rel2triple_id_list
        self.trpIds = self.side_info.trpIds
        entity_embedding, relation_embedding = self.entity_embedding.data, self.relation_embedding.data
        self.seed_trpIds, self.seed_sim = [], []
        for i in tqdm(range(len(entity_embedding))):
            for j in range(i + 1, len(entity_embedding)):
                e1_embed, e2_embed = entity_embedding[i], entity_embedding[j]
                sim = mindspore.ops.cosine_similarity(e1_embed, e2_embed, dim=0)
                if sim > self.p.entity_threshold:
                    ent1, ent2 = self.id2ent[i], self.id2ent[j]
                    for ent in [ent1, ent2]:
                        triple_list = self.ent2triple_id_list[ent]
                        for triple_id in triple_list:
                            triple = self.trpIds[triple_id]
                            if str(self.id2ent[triple[0]]) == str(ent1):
                                trp = (self.ent2id[str(ent2)], triple[1], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2ent[triple[0]]) == str(ent2):
                                trp = (self.ent2id[str(ent1)], triple[1], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2ent[triple[2]]) == str(ent1):
                                trp = (triple[0], triple[1], self.ent2id[str(ent2)])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2ent[triple[2]]) == str(ent2):
                                trp = (triple[0], triple[1], self.ent2id[str(ent1)])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)

        for i in tqdm(range(len(relation_embedding))):
            for j in range(i + 1, len(relation_embedding)):
                r1_embed, r2_embed = relation_embedding[i], relation_embedding[j]
                sim = mindspore.ops.cosine_similarity(r1_embed, r2_embed, dim=0)
                if sim > self.p.relation_threshold:
                    rel1, rel2 = self.id2rel[i], self.id2rel[j]
                    for rel in [rel1, rel2]:
                        triple_list = self.rel2triple_id_list[rel]
                        for triple_id in triple_list:
                            triple = self.trpIds[triple_id]
                            if str(self.id2rel[triple[1]]) == str(rel1):
                                trp = (triple[0], self.rel2id[str(rel2)], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2rel[triple[1]]) == str(rel2):
                                trp = (triple[0], self.rel2id[str(rel1)], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
        return self.seed_trpIds, self.seed_sim

    def set_logger(self):
        '''
        Write logs to checkpoint and console
        '''

        if self.p.do_train:
            log_file = os.path.join(self.p.out_path or self.p.init_checkpoint, 'train.log')
        else:
            log_file = os.path.join(self.p.out_path or self.p.init_checkpoint, 'test.log')

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        log = logging.getLogger('')

        log.addHandler(console)

    def log_metrics(self, mode, step, metrics):
        '''
        Print the evaluation logs
        '''
        for metric in metrics:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
    def remove_logger(self):

        log = logging.getLogger('')
        log.removeHandler(log.handlers[-1])