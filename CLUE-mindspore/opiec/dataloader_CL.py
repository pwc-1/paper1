#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import mindspore
from mindspore.dataset import GeneratorDataset

resdict = dict()
fname1 = 'data/myexp/numbertrueDict.txt'
file1 = open(fname1, 'r', encoding='utf-8')
for line in file1:
    dict1 = eval(line)
    resdict.update(dict1)


fname2='self.ent2id'
fname3='self.id2ent'
ent2id = pickle.load(open(fname2, 'rb'))
id2ent = pickle.load(open(fname3, 'rb'))

class TrainDataset():
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = mindspore.ops.sqrt(1 / mindspore.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = mindspore.Tensor(negative_sample)

        positive_sample = mindspore.Tensor(positive_sample).long()
        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(positive_sample, negative_sample, subsampling_weight,mode):
        #positive_sample = mindspore.ops.stack([_[0] for _ in data], axis=0)
        #negative_sample = mindspore.ops.stack([_[1] for _ in data], axis=0)
        #subsample_weight = mindspore.ops.concat([_[2] for _ in data], axis=0)
        #mode = data[0][3]
        #print(mode)
        return positive_sample, negative_sample, subsampling_weight, mode
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

class CanSeedDataset():
    def __init__(self, pairs, nentity, nrelation, negative_sample_size, mode,ents):
        self.len = len(pairs)
        self.pairs = pairs
        self.pair_set = set(pairs)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.ents=ents
        self.mode = mode
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.pairs)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.pairs[idx]


        head, tail = positive_sample


        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            if self.mode == 'head-batch':

                randlist = np.random.randint(low=0,high=21855, size=self.negative_sample_size * 2)
                randlist = randlist.tolist()
                negative_sample=np.array([self.ents[x] for x in randlist])
                mask = np.in1d(
                    negative_sample,
                    self.true_head[tail],
                    assume_unique=True,
                    invert=True
                )
                mode='head-align'
            elif self.mode == 'tail-batch':
                randlist = np.random.randint(low=0,high=21855, size=self.negative_sample_size * 2)
                randlist = randlist.tolist()
                negative_sample=np.array([self.ents[x] for x in randlist])
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[head],
                    assume_unique=True,
                    invert=True
                )
                mode = 'tail-align'
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = mindspore.Tensor(negative_sample)

        positive_sample = mindspore.Tensor(positive_sample).long()


        return positive_sample, negative_sample, mode

    @staticmethod
    def collate_fn(positive_sample, negative_sample, mode):
        # positive_sample = mindspore.ops.stack([_[0] for _ in data], axis=0)
        # negative_sample = mindspore.ops.stack([_[1] for _ in data], axis=0)

        # mode = data[0][2]
        print(mode)
        return positive_sample, negative_sample, mode

    @staticmethod
    def get_true_head_and_tail(pairs):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, tail in pairs:
            if head not in true_tail:
                true_tail[head] = []
            true_tail[head].append(tail)
            if tail not in true_head:
                true_head[tail] = []
            true_head[tail].append(head)

        for tail in true_head:
            true_head[tail] = np.array(list(set(true_head[tail])))
        for head in true_tail:
            true_tail[head] = np.array(list(set(true_tail[head])))

        return true_head, true_tail


class ELSeedDataset():
    def __init__(self, pairs, nentity, nrelation, negative_sample_size, mode,ents):
        self.len = len(pairs)
        self.pairs = pairs
        self.pair_set = set(pairs)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.ents=ents
        self.mode = mode
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.pairs)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.pairs[idx]


        head, tail = positive_sample
        negative_sample_list = []
        negative_sample_size = 0
        '''
        if self.mode == 'tail-batch' and id2ent[head] in resdict.keys():
            for candidate in resdict[id2ent[head]].keys():
                if ent2id[candidate]!=tail:
                    negative_sample_list.append(np.array([ent2id[candidate]]))
                    negative_sample_size+=1
        '''

        while negative_sample_size < self.negative_sample_size:
            if self.mode == 'head-batch':

                randlist = np.random.randint(low=0,high=21855, size=self.negative_sample_size * 2)
                randlist = randlist.tolist()
                negative_sample=np.array([self.ents[x] for x in randlist])
                mask = np.in1d(
                    negative_sample,
                    self.true_head[tail],
                    assume_unique=True,
                    invert=True
                )
                mode='head-align'
            elif self.mode == 'tail-batch':
                randlist = np.random.randint(low=0,high=1023417, size=self.negative_sample_size * 2)
                randlist = randlist.tolist()
                negative_sample=np.array([self.ents[x] for x in randlist])
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[head],
                    assume_unique=True,
                    invert=True
                )
                mode = 'tail-align'
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = mindspore.Tensor(negative_sample)

        positive_sample = mindspore.Tensor(positive_sample).long()

        return positive_sample, negative_sample, mode

    @staticmethod
    def collate_fn(positive_sample,negative_sample, mode):
        #positive_sample = mindspore.ops.stack([_[0] for _ in data], axis=0)
        #negative_sample = mindspore.ops.stack([_[1] for _ in data], axis=0)

        #mode = data[0][2]
        print(mode)
        return positive_sample, negative_sample,  mode




    @staticmethod
    def get_true_head_and_tail(pairs):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, tail in pairs:
            if head not in true_tail:
                true_tail[head] = []
            true_tail[head].append(tail)
            if tail not in true_head:
                true_head[tail] = []
            true_head[tail].append(head)

        for tail in true_head:
            true_head[tail] = np.array(list(set(true_head[tail])))
        for head in true_tail:
            true_tail[head] = np.array(list(set(true_tail[head])))

        return true_head, true_tail


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

