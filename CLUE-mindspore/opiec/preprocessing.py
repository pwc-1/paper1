'''
Side Information Acquisition module
'''

from helper import *
import pdb, itertools
from nltk.corpus import stopwords
from utils import *
import pickle

'''*************************************** INPUT CLASS ********************************************'''

class SideInfo(object):
    def __init__(self, args, triples_list):
        self.p = args
        self.file = open(self.p.out_path + '/side_info.txt', 'w')
        self.triples = triples_list

        self.initVariables()
        self.process()

    def process(self):
        self.folder_to_make = './file/' + self.p.dataset + '_' + self.p.split + '/'
        self.common='./file/common/'
        if not os.path.exists(self.folder_to_make):
            print('folder_to_make:', self.folder_to_make)
            os.makedirs(self.folder_to_make)
        fname1, fname2, fname3 = self.common + '/self.rel_list', self.common + '/self.ent_list', self.common + '/self.sub_list'
        fname4, fname5, fname6 = self.common + '/self.obj_list', self.common + '/self.ent2id', self.common + '/self.rel2id'
        fname7, fname8, fname9 = self.common + '/self.isSub', self.common + '/self.ent_freq', self.common + '/self.rel_freq'
        fname10, fname11, fname12 = self.common + '/self.id2ent', self.common + '/self.id2rel', self.common + '/self.trpIds'
        fname13, fname14, fname15 = self.common + '/self.sub2id', self.common + '/self.id2sub', self.common + '/self.obj2id'
        fname16, fname17, fname18 = self.common + '/self.id2obj', self.common + '/self.ent_id2sentence_list', self.common + '/self.sentence_list'
        fname19, fname20 = self.common + '/self.ent2triple_id_list', self.common + '/self.rel2triple_id_list'

        if not checkFile(fname1) or not checkFile(fname2):
            print('generate side_info')
            ent1List, relList, ent2List = [], [], []  # temp variables
            self.sentence_List = []
            self.ent2triple_id_list, self.rel2triple_id_list = dict(), dict()
            if self.p.use_assume:#是否使用Unique形式表示NP
                self.triple_str = str('triple')
                print('use assume...')
            else:
                self.triple_str = str('triple_unique')
                print('do not use assume...')
            #triple_num, sentence_num = 0, 0
            triple_num = 0
            for triple in self.triples:  # Get all subject, objects and relations
                sub, rel, obj = triple[self.triple_str][0], triple[self.triple_str][1], triple[self.triple_str][2]
                ent1List.append(sub)
                relList.append(rel)
                ent2List.append(obj)

                if sub not in self.ent2triple_id_list:
                    self.ent2triple_id_list.update({sub: [triple_num]})
                else:
                    self.ent2triple_id_list[sub].append(triple_num)

                if rel not in self.rel2triple_id_list:
                    self.rel2triple_id_list.update({rel: [triple_num]})
                else:
                    self.rel2triple_id_list[rel].append(triple_num)
                
                if obj not in self.ent2triple_id_list:
                    self.ent2triple_id_list.update({obj: [triple_num]})
                else:
                    self.ent2triple_id_list[obj].append(triple_num)


                triple_num += 1

            #print('relList:', len(relList))  # 35812
            #print('ent1List:', len(ent1List))  # 35812
            #print('ent2List:', len(ent2List))  # 35812


            assume_rel, assume_sub, assume_obj = dict(), dict(), dict()
            for i in range(len(relList)):
                rel = relList[i]
                if rel in assume_rel.keys():
                    assume_rel[rel].append(i)
                else:
                    assume_rel[rel] = [i]

            for i in range(len(ent1List)):
                sub = ent1List[i]
                if sub in assume_sub.keys():
                    assume_sub[sub].append(i)
                else:
                    assume_sub[sub] = [i]

            for i in range(len(ent2List)):
                obj = ent2List[i]
                if obj in assume_obj.keys():
                    assume_obj[obj].append(i)
                else:
                    assume_obj[obj] = [i]

            #print('assume_rel, assume_sub, assume_obj:', len(assume_rel), len(assume_sub),len(assume_obj))  # 18288 12295 14935
            self.rel_list = list(assume_rel.keys())
            self.sub_list = list(assume_sub.keys())
            self.obj_list = list(assume_obj.keys())
            self.ent_list = []  # self.ent_list 's order is self.sub_list + self.obj_list +no knowledge ent
            self.ent_id2sentence_list = []
            # print('assume_sub:', assume_sub)  # {'The Guardian': [0, 1], 'Guardian': [2], 'Franz Kafka': [3, 4], 'Kafka': [5],

            for i in range(len(self.sub_list)):
                ent = self.sub_list[i]

                self.ent_list.append(ent)


            for i in range(len(self.obj_list)):
                obj = self.obj_list[i]

                if obj in self.sub_list:
                    continue
                else:
                    self.ent_list.append(obj)
            #这里是新加的
            for i in range(21856,1045274):
                if str(i) not in self.ent_list:
                    self.ent_list.append(str(i))


            #print('self.ent_list:', len(self.ent_list))  # 23735
            #print('self.sub_list:', len(self.sub_list))  # 12295
            #print('self.obj_list:', len(self.obj_list))  # 14935
            #print('self.rel_list:', len(self.rel_list))  # 18288

            # Generate a unique id for each entity and relations
            self.ent2id = dict([(v, k) for k, v in enumerate(self.ent_list)])
            self.rel2id = dict([(v, k) for k, v in enumerate(self.rel_list)])
            self.sub2id = dict([(v, k) for k, v in enumerate(self.sub_list)])
            self.obj2id = dict([(v, k) for k, v in enumerate(self.obj_list)])
            #print('self.sub2id:', len(self.sub2id))  # 12295
            #print('self.obj2id:', len(self.obj2id))  # 14935
            #print('self.ent2id:', len(self.ent2id))  # 23735
            #print('self.rel2id:', len(self.rel2id))  # 18288

            self.isSub = {}
            for sub in self.sub_list:
                self.isSub[self.ent2id[sub]] = 1
            #print('self.isSub:', len(self.isSub))  # 12295

            # Get frequency of occurence of entities and relations
            for ele in ent1List:
                ent = self.ent2id[ele]
                self.ent_freq[ent] = self.ent_freq.get(ent, 0)
                self.ent_freq[ent] += 1

            for ele in ent2List:
                ent = self.ent2id[ele]
                self.ent_freq[ent] = self.ent_freq.get(ent, 0)
                self.ent_freq[ent] += 1

            for ele in relList:
                rel = self.rel2id[ele]
                self.rel_freq[rel] = self.rel_freq.get(rel, 0)
                self.rel_freq[rel] += 1

            # Creating inverse mapping as well
            self.id2ent = invertDic(self.ent2id)
            self.id2rel = invertDic(self.rel2id)
            self.id2sub = invertDic(self.sub2id)
            self.id2obj = invertDic(self.obj2id)
            # self.id2text = invertDic(self.text2id)

            #print('self.ent_freq:', len(self.ent_freq))  # 23735
            #print('self.rel_freq:', len(self.rel_freq))  # 18288
            #print('self.id2ent:', len(self.id2ent))  # 23735
            #print('self.id2rel:', len(self.id2rel))  # 18288
            #print('self.id2sub:', len(self.id2sub))  # 12295
            #print('self.id2obj:', len(self.id2obj))  # 14935

            for triple in self.triples:
                trp = (
                    self.ent2id[triple[self.triple_str][0]], self.rel2id[triple[self.triple_str][1]],
                    self.ent2id[triple[self.triple_str][2]])
                self.trpIds.append(trp)
            #print('self.trpIds:', len(self.trpIds))  # 35812

            pickle.dump(self.rel_list, open(fname1, 'wb'))
            pickle.dump(self.ent_list, open(fname2, 'wb'))
            pickle.dump(self.sub_list, open(fname3, 'wb'))
            pickle.dump(self.obj_list, open(fname4, 'wb'))
            pickle.dump(self.ent2id, open(fname5, 'wb'))
            pickle.dump(self.rel2id, open(fname6, 'wb'))
            pickle.dump(self.isSub, open(fname7, 'wb'))
            pickle.dump(self.ent_freq, open(fname8, 'wb'))
            pickle.dump(self.rel_freq, open(fname9, 'wb'))
            pickle.dump(self.id2ent, open(fname10, 'wb'))
            pickle.dump(self.id2rel, open(fname11, 'wb'))
            pickle.dump(self.trpIds, open(fname12, 'wb'))
            pickle.dump(self.sub2id, open(fname13, 'wb'))
            pickle.dump(self.id2sub, open(fname14, 'wb'))
            pickle.dump(self.obj2id, open(fname15, 'wb'))
            pickle.dump(self.id2obj, open(fname16, 'wb'))
            pickle.dump(self.ent2triple_id_list, open(fname19, 'wb'))
            pickle.dump(self.rel2triple_id_list, open(fname20, 'wb'))

        else:
            #print('load side_info')
            self.rel_list = pickle.load(open(fname1, 'rb'))
            self.ent_list = pickle.load(open(fname2, 'rb'))
            self.sub_list = pickle.load(open(fname3, 'rb'))
            self.obj_list = pickle.load(open(fname4, 'rb'))
            self.ent2id = pickle.load(open(fname5, 'rb'))
            self.rel2id = pickle.load(open(fname6, 'rb'))
            self.isSub = pickle.load(open(fname7, 'rb'))
            self.ent_freq = pickle.load(open(fname8, 'rb'))
            self.rel_freq = pickle.load(open(fname9, 'rb'))
            self.id2ent = pickle.load(open(fname10, 'rb'))
            self.id2rel = pickle.load(open(fname11, 'rb'))
            self.trpIds = pickle.load(open(fname12, 'rb'))
            self.sub2id = pickle.load(open(fname13, 'rb'))
            self.id2sub = pickle.load(open(fname14, 'rb'))
            self.obj2id = pickle.load(open(fname15, 'rb'))
            self.id2obj = pickle.load(open(fname16, 'rb'))
            self.ent2triple_id_list = pickle.load(open(fname19, 'rb'))
            self.rel2triple_id_list = pickle.load(open(fname20, 'rb'))
            '''
        print('self.rel_list:', type(self.rel_list), len(self.rel_list))
        print('self.ent_list:', type(self.ent_list), len(self.ent_list))
        print('self.sub_list:', type(self.sub_list), len(self.sub_list))
        print('self.obj_list:', type(self.obj_list), len(self.obj_list))
        print('self.ent2id:', type(self.ent2id), len(self.ent2id))
        print('self.rel2id:', type(self.rel2id), len(self.rel2id))
        print('self.isSub:', type(self.isSub), len(self.isSub))
        print('self.ent_freq:', type(self.ent_freq), len(self.ent_freq))
        print('self.rel_freq:', type(self.rel_freq), len(self.rel_freq))
        print('self.id2ent:', type(self.id2ent), len(self.id2ent))
        print('self.id2rel:', type(self.id2rel), len(self.id2rel))
        print('self.trpIds:', type(self.trpIds), len(self.trpIds))
        print('self.sub2id:', type(self.sub2id), len(self.sub2id))
        print('self.id2sub:', type(self.id2sub), len(self.id2sub))
        print('self.obj2id:', type(self.obj2id), len(self.obj2id))
        print('self.id2obj:', type(self.id2obj), len(self.id2obj))
        print('self.ent2triple_id_list:', type(self.ent2triple_id_list), len(self.ent2triple_id_list))
        print('self.rel2triple_id_list:', type(self.rel2triple_id_list), len(self.rel2triple_id_list))
        print()
'''



    def get_EL_seed(self):
        seed_sim, seed_trpIds = [], []

        fname1, fname2 = self.folder_to_make + '/1E_init', self.folder_to_make + '/1R_init'
        if not checkFile(fname1) or not checkFile(fname2):
            print('generate pre-trained embeddings')
            import gensim
            model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
            self.E_init = getEmbeddings(model, self.ent_list, self.p.embed_dims)
            self.R_init = getEmbeddings(model, self.rel_list, self.p.embed_dims)

            pickle.dump(self.E_init, open(fname1, 'wb'))
            pickle.dump(self.R_init, open(fname2, 'wb'))
        else:
            print('load pre-trained embeddings')
            self.E_init = pickle.load(open(fname1, 'rb'))
            self.R_init = pickle.load(open(fname2, 'rb'))


    ''' ATTRIBUTES DECLARATION '''

    def initVariables(self):
        self.ent_list = None  # List of all entities
        self.clean_ent_list = []
        self.rel_list = None  # List of all relations
        self.trpIds = []  # List of all triples in id format
        self.node = []
        self.seed_trpIds = []
        self.new_trpIds = []

        self.ent2id = None  # Maps entity to its id (o2o)
        self.rel2id = None  # Maps relation to its id (o2o)
        self.id2ent = None  # Maps id to entity (o2o)
        self.id2rel = None  # Maps id to relation (o2o)

        self.ent_freq = {}  # Entity to its frequency
        self.rel_freq = {}  # Relation to its frequency

        self.ent2name_seed = {}
        self.rel2name_seed = {}