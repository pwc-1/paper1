import gensim, itertools, pickle, time
from helper import *
from utils import cos_sim
from test_performance import cluster_test, HAC_getClusters
from weighted_train_embedding_model import Train_Embedding_Model, pair2triples


class DisjointSet(object):
    def __init__(self):
        self.leader = {}  # maps a member to the group's leader
        self.group = {}  # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return  # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])


def amieInfo(triples, ent2id, rel2id):
    uf = DisjointSet()
    min_supp = 2
    min_conf = 0.5  # cesi=0.2
    amie_cluster = []
    rel_so = {}

    for trp in triples:
        sub, rel, obj = trp['triple']
        if sub in ent2id and rel in rel2id and obj in ent2id:
            sub_id, rel_id, obj_id = ent2id[sub], rel2id[rel], ent2id[obj]
            rel_so[rel_id] = rel_so.get(rel_id, set())
            rel_so[rel_id].add((sub_id, obj_id))

    for r1, r2 in itertools.combinations(rel_so.keys(), 2):
        supp = len(rel_so[r1].intersection(rel_so[r2]))
        if supp < min_supp: continue

        s1, _ = zip(*list(rel_so[r1]))
        s2, _ = zip(*list(rel_so[r2]))

        z_conf_12, z_conf_21 = 0, 0
        for ele in s1:
            if ele in s2: z_conf_12 += 1
        for ele in s2:
            if ele in s1: z_conf_21 += 1

        conf_12 = supp / z_conf_12
        conf_21 = supp / z_conf_21

        if conf_12 >= min_conf and conf_21 >= min_conf:
            amie_cluster.append((r1, r2))  # Replace with union find DS
            uf.add(r1, r2)

    rel2amie = uf.leader
    return rel2amie


def seed_pair2cluster(seed_pair_list, ent_list):
    pair_dict = dict()
    for seed_pair in seed_pair_list:
        a, b = seed_pair
        if a != b:
            if a < b:
                rep, ent_id = a, b
            else:
                ent_id, rep = b, a
            if ent_id not in pair_dict:
                if rep not in pair_dict:
                    pair_dict.update({ent_id: rep})
                else:
                    new_rep = pair_dict[rep]
                    j = 0
                    while rep in pair_dict:
                        new_rep = pair_dict[rep]
                        rep = new_rep
                        j += 1
                        if j > 1000000:
                            break
                    pair_dict.update({ent_id: new_rep})
            else:
                if rep not in pair_dict:
                    new_rep = pair_dict[ent_id]
                    if rep > new_rep:
                        pair_dict.update({rep: new_rep})
                    else:
                        pair_dict.update({new_rep: rep})
                else:
                    old_rep = rep
                    new_rep = pair_dict[rep]
                    j = 0
                    while rep in pair_dict:
                        new_rep = pair_dict[rep]
                        rep = new_rep
                        j += 1
                        if j > 1000000:
                            break
                    if old_rep > new_rep:
                        pair_dict.update({ent_id: new_rep})
                    else:
                        pair_dict.update({ent_id: old_rep})

    cluster_list = []
    for i in range(len(ent_list)):
        cluster_list.append(i)
    for ent_id in pair_dict:
        rep = pair_dict[ent_id]
        if ent_id < len(cluster_list):
            cluster_list[ent_id] = rep
    return cluster_list


def get_entseed_pair(ent_list, ent2id):
    seed_pair = []
    scoredict={}
    alreadynp=set()
    with open('data/myexp/originallinkseed.txt', "r", encoding='utf-8') as f:
        for line in f:
            res = line.strip('\n').split('#####')
            i=ent2id[res[0]]
            j=ent2id[res[1]]
            id_tuple = (i, j)
            alreadynp.add(i)
            scoredict[(i, j)]=1
            seed_pair.append(id_tuple)
    with open('data/myexp/softlinkseed.txt', "r", encoding='utf-8') as f:
        for line in f:
            res = line.strip('\n').split('#####')
            i=ent2id[res[0]]
            j=ent2id[res[1]]
            score=float(res[2])
            if i not in alreadynp:
                id_tuple = (i, j)
                scoredict[(i, j)] = score
                seed_pair.append(id_tuple)
    return seed_pair,scoredict

def get_canseed_pair(ent_list, ent2id):
    seed_pair = []
    scoredict = {}

    with open('data/myexp/softcanseed.txt', "r", encoding='utf-8') as f:
        for line in f:
            res = line.strip('\n').split('#####')
            i=ent2id[res[0]]
            j=ent2id[res[1]]
            score=float(res[2])
            id_tuple = (i, j)
            scoredict[(i, j)] = score
            seed_pair.append(id_tuple)
    with open('data/myexp/originalcanseed.txt', "r", encoding='utf-8') as f:
        for line in f:
            res = line.strip('\n').split('#####')
            i=ent2id[res[0]]
            j=ent2id[res[1]]
            id_tuple = (i, j)
            scoredict[(i, j)] = 1
            seed_pair.append(id_tuple)
    return seed_pair,scoredict




def difference_cluster2pair(cluster_list_1, cluster_list_2, EL_seed):
    new_seed_pair_list = []
    for i in range(len(cluster_list_1)):
        id_1, id_2 = cluster_list_1[i], cluster_list_2[i]
        if id_1 == id_2:
            continue
        else:
            index_list_1 = [i for i, x in enumerate(cluster_list_1) if x == id_1]
            index_list_2 = [i for i, x in enumerate(cluster_list_2) if x == id_2]
            if len(index_list_2) == 1:
                continue
            else:
                iter_list_1 = list(itertools.combinations(index_list_1, 2))
                iter_list_2 = list(itertools.combinations(index_list_2, 2))
                if len(iter_list_1) > 0:
                    for iter_pair in iter_list_1:
                        if iter_pair in iter_list_2: iter_list_2.remove(iter_pair)
                for iter in iter_list_2:
                    if iter not in EL_seed:
                        new_seed_pair_list.append(iter)
    return new_seed_pair_list


def totol_cluster2pair(cluster_list):
    seed_pair_list, id_list = [], []
    for i in range(len(cluster_list)):
        id = cluster_list[i]
        if id not in id_list:
            id_list.append(id)
            index_list = [i for i, x in enumerate(cluster_list) if x == id]
            if len(index_list) > 1:
                iter_list = list(itertools.combinations(index_list, 2))
                seed_pair_list += iter_list
    return seed_pair_list





class W_Embeddings(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, ite, params, side_info, true_ent2clust, true_clust2ent, sub_uni2triple_dict=None,
                 triple_list=None):
        self.ite=ite
        self.p = params

        self.side_info = side_info
        self.ent2embed = {}  # Stores final embeddings learned for noun phrases
        self.rel2embed = {}  # Stores final embeddings learned for relation phrases
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.sub_uni2triple_dict = sub_uni2triple_dict
        self.triples_list = triple_list
        self.NPname={}
        with open('data/myexp/NPid.txt', "r", encoding='utf-8') as f:
            for line in f:
                res = line.strip('\n').split('#####')
                self.NPname[res[1]]=res[0]

        self.RPname={}
        with open('data/myexp/RPid.txt', "r", encoding='utf-8') as f:
            for line in f:
                res = line.strip('\n').split('#####')
                self.RPname[res[1]]=res[0]
        self.entname={}
        with open('data/myexp/ckbname2id.txt', "r", encoding='utf-8') as f:
            for line in f:
                if '######' in line:
                    res = line.strip('\n').split('######')
                    self.entname[res[1]]=res[0]
                else:
                    res = line.strip('\n').split('#####')
                    self.entname[res[1]]=res[0]

        self.relname={}
        with open('data/myexp/relation2id.txt', "r", encoding='utf-8') as f:
            for line in f:
                res = line.strip('\n').split('#####')
                self.relname[res[0]]=res[1]

    def getrelname(self,rel):
        if 'P' in rel:
            return self.relname[rel]
        else:
            return self.RPname[rel]

    def getentname(self,ent):
        num = int(ent)
        if num <= 21855:
            return self.NPname[ent]
        else:
            return self.entname[ent]
    def fit(self):

        show_memory = False
        if show_memory:
            print('show_memory:', show_memory)
            import tracemalloc
            tracemalloc.start(25)  # 默认25个片段，这个本质还是多次采样

        clean_ent_list, clean_rel_list = [], []
        entname_list,relname_list=[],[]
        for ent in self.side_info.ent_list: clean_ent_list.append(ent)
        for rel in self.side_info.rel_list: clean_rel_list.append(rel)

        for ent in self.side_info.ent_list: entname_list.append(self.getentname(ent))
        for rel in self.side_info.rel_list: relname_list.append(self.getrelname(rel))
        print('clean_ent_list:', type(clean_ent_list), len(clean_ent_list))
        print('clean_rel_list:', type(clean_rel_list), len(clean_rel_list))
        ''' Intialize embeddings '''
        if self.p.embed_init == 'crawl':
            fname1, fname2 = './file/' + self.p.dataset + '_' + self.p.split + '/1E_init', './file/' + self.p.dataset + '_' + self.p.split + '/1R_init'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate pre-trained embeddings')

                model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
                self.E_init = getEmbeddings(model, entname_list, self.p.embed_dims)
                self.R_init = getEmbeddings(model, relname_list, self.p.embed_dims)

                pickle.dump(self.E_init, open(fname1, 'wb'))
                pickle.dump(self.R_init, open(fname2, 'wb'))
            else:
                print('load init embeddings')
                self.E_init = pickle.load(open(fname1, 'rb'))
                self.R_init = pickle.load(open(fname2, 'rb'))

        else:
            print('generate init random embeddings')
            self.E_init = np.random.rand(len(clean_ent_list), self.p.embed_dims)
            self.R_init = np.random.rand(len(clean_rel_list), self.p.embed_dims)

        folder_to_make = './file/' + self.p.dataset + '_' + 'iter'+str(self.ite+1)+ '/'
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)

        fname_EL = './file/' + self.p.dataset + '_' + self.p.split + '/EL_seed'
        fname_ELscore = './file/' + self.p.dataset + '_' + self.p.split + '/EL_seedscore'
        if not checkFile(fname_EL):
            self.EL_seed,elscoredict = get_entseed_pair(self.side_info.ent_list, self.side_info.ent2id)
            pickle.dump(self.EL_seed, open(fname_EL, 'wb'))
            pickle.dump(elscoredict, open(fname_ELscore, 'wb'))
        else:
            self.EL_seed = pickle.load(open(fname_EL, 'rb'))
            elscoredict=pickle.load(open(fname_ELscore, 'rb'))
        #print('self.EL_seed:', type(self.EL_seed), len(self.EL_seed))

        fname_Can = './file/' + self.p.dataset + '_' + self.p.split + '/Can_seed'
        fname_Canscore = './file/' + self.p.dataset + '_' + self.p.split + '/Can_seedscore'
        if not checkFile(fname_Can):
            self.Can_seed,canscoredict = get_canseed_pair(self.side_info.ent_list, self.side_info.ent2id)
            pickle.dump(self.Can_seed, open(fname_Can, 'wb'))
            pickle.dump(canscoredict, open(fname_Canscore, 'wb'))
        else:
            self.Can_seed = pickle.load(open(fname_Can, 'rb'))
            canscoredict=pickle.load(open(fname_Canscore, 'rb'))




        self.all_elseed_pair_list = []

        for pair in self.EL_seed:
            if pair not in self.all_elseed_pair_list:
                self.all_elseed_pair_list.append(pair)

        self.all_canseed_pair_list = []
        for pair in self.Can_seed:
            if pair not in self.all_canseed_pair_list:
                self.all_canseed_pair_list.append(pair)




        relation_seed_pair_list = self.all_canseed_pair_list
        relation_seed_cluster_list = seed_pair2cluster(relation_seed_pair_list, clean_ent_list)

        self.seed_trpIds, self.seed_sim = pair2triples(relation_seed_pair_list, clean_ent_list, self.side_info.ent2id,
                                                       self.side_info.id2ent, self.side_info.ent2triple_id_list,
                                                       self.side_info.trpIds, self.E_init, cos_sim, is_cuda=False,
                                                       high_confidence=False)
        if self.p.use_Embedding_model:
            fname1, fname2 = folder_to_make+ '1E_init',  folder_to_make+ '1R_init'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate TransE embeddings', fname1)
                self.new_seed_trpIds, self.new_seed_sim = self.seed_trpIds, self.seed_sim
                entity_embedding, relation_embedding = self.E_init, self.R_init
                print('self.training_time', 'use pre-trained crawl embeddings ... ')

                TEM = Train_Embedding_Model(self.p, self.side_info, entity_embedding, relation_embedding,
                                            self.all_elseed_pair_list,self.all_canseed_pair_list, self.new_seed_trpIds, self.new_seed_sim,elscoredict,canscoredict)
                self.entity_embedding, self.relation_embedding = TEM.train()

                pickle.dump(self.entity_embedding, open(fname1, 'wb'))
                pickle.dump(self.relation_embedding, open(fname2, 'wb'))
            else:
                print('load TransE embeddings')
                self.entity_embedding = pickle.load(open(fname1, 'rb'))
                self.relation_embedding = pickle.load(open(fname2, 'rb'))

            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.entity_embedding[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.relation_embedding[id]

        else:  # do not use embedding model
            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.E_init[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.R_init[id]




        # exit()

