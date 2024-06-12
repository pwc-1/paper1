from helper import *
from preprocessing import SideInfo  # For processing data and side information
from embeddings_multi_task import Embeddings
from embeddings_weighted import W_Embeddings
from utils import *
import os, argparse, pickle, codecs
from test_performance import cluster_test, HAC_getClusters
from collections import defaultdict as ddict
from findkbyDB import findkbydb
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
''' *************************************** DATASET PREPROCESSING **************************************** '''



def link(ite:int):
    import random
    import math
    import itertools
    random.seed(1999)
    fname1 = './output/reverb_iter'+str(ite)+'_1/embed_ent.pkl'
    fname2 = './file/common/self.ent2id'
    fname3 = './file/common/self.id2ent'

    fname4 = 'data/myexp/numbertrueDict.txt'

    res1, res2 =  'iter'+str(ite),  'iter'+str(ite)+'confidence'
    if not checkFile(res1) or not checkFile(res2):
        def sigmoid(num):
            return 1 / (1 + math.exp(-num))

        k = 1

        def cos_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
            cos_theta = 0.5 + 0.5 * cos_theta
            return cos_theta

        ent2embed = pickle.load(open(fname1, 'rb'))
        ent2id = pickle.load(open(fname2, 'rb'))
        id2ent = pickle.load(open(fname3, 'rb'))

        resdict = dict()
        file4 = open(fname4, 'r', encoding='utf-8')
        for line in file4:
            dict1 = eval(line)
            resdict.update(dict1)

        result = dict()
        totalcandi = [str(i) for i in range(21856, 1045274)]

        confidence = dict()
        errcount = 0

        for i in tqdm(range(28797)):
            queryid = ent2id[str(i)]
            queryemb = ent2embed[queryid]
            if str(i) in resdict.keys():
                candidates = resdict[str(i)]
                scoredict = dict()
                for ent in candidates:
                    emb = ent2embed[ent2id[ent]]
                    sim = cos_sim(queryemb, emb)
                    scoredict[ent] = sim
                mostcloset = max(scoredict, key=scoredict.get)
                result[str(i)] = str(mostcloset)
                if len(scoredict) == 1:
                    confidence[str(i)] = 1
                else:
                    x1 = scoredict[sorted(scoredict, key=scoredict.get, reverse=True)[0]]
                    x2 = scoredict[sorted(scoredict, key=scoredict.get, reverse=True)[1]]
                    score = sigmoid(k * (x1 - x2) / x1)
                    confidence[str(i)] = score
            else:
                errcount += 1
                scoredict = dict()
                for ent in totalcandi:
                    emb = ent2embed[ent2id[ent]]
                    sim = cos_sim(queryemb, emb)
                    scoredict[ent] = sim
                mostcloset = max(scoredict, key=scoredict.get)
                result[str(i)] = str(mostcloset)

                x1 = scoredict[sorted(scoredict, key=scoredict.get, reverse=True)[0]]
                x2 = scoredict[sorted(scoredict, key=scoredict.get, reverse=True)[1]]
                score = sigmoid(k * (x1 - x2) / x1)
                confidence[str(i)] = score


        pickle.dump(result, open('iter'+str(ite), 'wb'))

        pickle.dump(confidence, open('iter'+str(ite)+'confidence', 'wb'))

    # #generate soft cano seed for next iteration
    print('generate soft canonicalization seed for iter' + str(ite + 1))
    result = pickle.load(open('iter'+str(ite), 'rb'))
    confidence = pickle.load(open('iter'+str(ite)+'confidence', 'rb'))

    inv_map = {}
    for k, v in result.items():
         if math.log(1+math.exp(confidence[k]),math.e)>0.95+0.05*math.exp(-(ite+1)/10):#threshold

            inv_map[v] = inv_map.get(v, [])
            inv_map[v].append(k)

    dst = open('data/myexp/softcanseed.txt', "w", encoding='utf-8')


    for ent in inv_map:
        if len(inv_map[ent]) > 1:
            cc = list(itertools.combinations(inv_map[ent], 2))
            for tuple in cc:
                a = tuple[0]
                b = tuple[1]
                confid = (confidence[a] + confidence[b]) / 2
                dst.writelines(a + '#####' + b + '#####' + str(confid) + '\n')


    #test link accuracy
    truecount = 0
    totalcount = 0
    result1 = pickle.load(open('iter'+str(ite), 'rb'))



    ckbdict = dict()
    with open('data/myexp/ckbentid.txt', "r", encoding='utf-8') as f:
        for line in f:
            res = line.strip('\n').split('#####')
            qid = res[0]
            entnumber = res[1]
            ckbdict[qid] = entnumber


    with open('data/myexp/cleantesttriple.txt', "r", encoding='utf-8') as f:
        for line in f:
            res = line.strip('\n').split('#####')
            num = res[0]
            query1 = res[1]
            query2 = res[3]
            subent = res[4]
            objent = res[5]
            totalcount += 1
            predict1 = result1[query1]
            answer1 = ckbdict[subent]
            if answer1 == predict1:
                truecount += 1
    print("Link result:"+'iter'+str(ite))
    print("right:"+str(truecount))
    print("total:"+str(totalcount))
    print("Accuracy:"+str(truecount / totalcount))


def cano(ite:int,args):
    triples_list = []
    true_ent2clust = ddict(set)
    ckbdict = dict()
    with open('data/myexp/ckbentid.txt', "r", encoding='utf-8') as f:
        for line in f:
            res = line.strip('\n').split('#####')
            qid = res[0]
            entnumber = res[1]
            ckbdict[qid] = entnumber
    with open('data/myexp/cleantesttriple.txt', "r", encoding='utf-8') as f:
        for line in f:
            trp = {}
            res = line.strip('\n').split('#####')
            num = res[0]
            subnp = res[1]
            objnp = res[3]
            rel = res[2]
            subent = res[4]
            objent = res[5]
            trp['triple'] = [subnp, rel, objnp]
            trp['triple_unique'] = [subnp + '|' + num, rel + '|' + num,
                                    objnp + '|' + num]

            trp['true_sub_link'] = ckbdict[subent]
            trp['true_obj_link'] = ckbdict[objent]
            triples_list.append(trp)
            true_ent2clust[subnp + '|' + num].add(ckbdict[subent])

    true_clust2ent = invertDic(true_ent2clust, 'm2os')


    fname2 = './file/common/self.ent2id'
    fname3 = './output/reverb_iter'+str(ite)+'_1/embed_ent.pkl'
    fname4 = 'nplist'


    ent2id = pickle.load(open(fname2, 'rb'))
    ent2embed = pickle.load(open(fname3, 'rb'))
    newid2ent = pickle.load(open(fname4, 'rb'))
    embed_list = []
    for ent in newid2ent:
        embed_list.append(ent2embed[ent2id[ent]])
    npemb = np.array(embed_list)
    pickle.dump(npemb, open('npembeddingiter'+str(ite), 'wb'))



    np2embed = pickle.load(open('npembeddingiter'+str(ite), 'rb'))

    np2id= pickle.load(open('self.np2id', 'rb'))
    id2np= pickle.load(open('self.id2np', 'rb'))
    isSub= pickle.load(open('self.isSub', 'rb'))
    relation_view_embed = []
    for ent in newid2ent:
        id = np2id[ent]
        if id in isSub:
            relation_view_embed.append(np2embed[id])
    print("Start predicting k for clustering!")
    k=findkbydb(relation_view_embed)

    threshold_or_cluster = 'cluster'
    if threshold_or_cluster == 'threshold':
        cluster_threshold_real = 0.02
    else:
        cluster_threshold_real = k
    print('cluster_k:', cluster_threshold_real)


    labels, clusters_center = HAC_getClusters(args, relation_view_embed, cluster_threshold_real, threshold_or_cluster)
    cluster_predict_list = list(labels)
    ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
        = cluster_test(args, ite, isSub, triples_list, np2id, id2np, cluster_predict_list, true_ent2clust,
                       true_clust2ent)
    print("Canonicalization result:"+'iter'+str(ite))
    print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
          'pair_prec=', pair_prec)
    print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
          'pair_recall=', pair_recall)
    print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
    print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
    print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
    print()





class CLUE_Main(object):

    def __init__(self, args):
        self.p = args
        self.read_triples()

    def read_triples(self):
        fname = self.p.out_path + self.p.file_triples  # File for storing processed triples
        self.triples_list = []  # List of all triples in the dataset
        self.amb_ent = ddict(int)  # Contains ambiguous entities in the dataset
        self.amb_mentions = {}  # Contains all ambiguous mentions
        self.isAcronym = {}  # Contains all mentions which can be acronyms

        print('dataset:', args.dataset)
        if args.dataset == 'reverb':
            print('load Reverb45k_dataset ... ')
            self.true_ent2clust = ddict(set)
            #self.triples_list = pickle.load(open(args.data_path, 'rb'))
            ckbdict=dict()
            with open('data/myexp/ckbentid.txt', "r", encoding='utf-8') as f:
                for line in f:
                    res = line.strip('\n').split('#####')
                    qid=res[0]
                    entnumber=res[1]
                    ckbdict[qid]=entnumber
            with open('data/myexp/cleanOKBtriple.txt', "r", encoding='utf-8') as f:
                for line in f:
                    trp={}
                    res = line.strip('\n').split('#####')
                    num=res[0]
                    subnp=res[1]
                    objnp=res[3]
                    rel=res[2]
                    subent=res[4]
                    objent=res[5]
                    trp['triple'] = [subnp, rel, objnp]
                    trp['triple_unique'] = [subnp + '|' + num, rel + '|' + num,
                                            objnp + '|' + num]

                    trp['true_sub_link'] = ckbdict[subent]
                    trp['true_obj_link'] = ckbdict[objent]
                    self.triples_list.append(trp)
                    self.true_ent2clust[subnp+'|'+num].add(ckbdict[subent])
            count=45032
            with open('data/myexp/cleansonset.txt', "r", encoding='utf-8') as f:
                for line in f:
                    trp={}
                    res = line.strip('\n').split('#####')
                    num=str(count)
                    subent=res[0]
                    objent=res[2]
                    rel=res[1]
                    trp['triple'] = [subent, rel, objent]
                    trp['triple_unique'] = [subent + '|' + num, rel + '|' + num,
                                            objent + '|' + num]

                    trp['true_sub_link'] = subent
                    trp['true_obj_link'] = objent
                    self.triples_list.append(trp)
                    self.true_ent2clust[subnp+'|'+num].add(subent)
                    count+=1
            self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')



        amb_clust = {}
        for trp in self.triples_list:
            sub = trp['triple'][0]
            for tok in sub.split():
                amb_clust[tok] = amb_clust.get(tok, set())
                amb_clust[tok].add(sub)

        for rep, clust in amb_clust.items():
            if rep in clust and len(clust) >= 3:
                self.amb_ent[rep] = len(clust)
                for ele in clust: self.amb_mentions[ele] = 1


    def get_sideInfo(self):
        fname = self.p.out_path + self.p.file_sideinfo_pkl

        if not checkFile(fname):
            self.side_info = SideInfo(self.p, self.triples_list)

            del self.side_info.file
            pickle.dump(self.side_info, open(fname, 'wb'))
        else:
            self.side_info = pickle.load(open(fname, 'rb'))

    def embedKG(self):
        fname1 = self.p.out_path + self.p.file_entEmbed
        fname2 = self.p.out_path + self.p.file_relEmbed

        if not checkFile(fname1) or not checkFile(fname2):
            embed = Embeddings(self.p, self.side_info, true_ent2clust=self.true_ent2clust,
                               true_clust2ent=self.true_clust2ent, triple_list=self.triples_list)
            embed.fit()

            self.ent2embed = embed.ent2embed  # Get the learned NP embeddings
            self.rel2embed = embed.rel2embed  # Get the learned RP embeddings

            pickle.dump(self.ent2embed, open(fname1, 'wb'))
            pickle.dump(self.rel2embed, open(fname2, 'wb'))
        else:
            self.ent2embed = pickle.load(open(fname1, 'rb'))
            self.rel2embed = pickle.load(open(fname2, 'rb'))


    def iterembedKG(self,ite:int):
        fname1 = self.p.out_path + self.p.file_entEmbed
        fname2 = self.p.out_path + self.p.file_relEmbed

        if not checkFile(fname1) or not checkFile(fname2):
            embed = W_Embeddings(ite,self.p, self.side_info, true_ent2clust=self.true_ent2clust,
                               true_clust2ent=self.true_clust2ent, triple_list=self.triples_list)
            embed.fit()

            self.ent2embed = embed.ent2embed  # Get the learned NP embeddings
            self.rel2embed = embed.rel2embed  # Get the learned RP embeddings

            pickle.dump(self.ent2embed, open(fname1, 'wb'))
            pickle.dump(self.rel2embed, open(fname2, 'wb'))
        else:
            self.ent2embed = pickle.load(open(fname1, 'rb'))
            self.rel2embed = pickle.load(open(fname2, 'rb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information')
    parser.add_argument('-data', dest='dataset', default='reverb', help='Dataset to run CESI on')
    parser.add_argument('-split', dest='split', default='iter0', help='Dataset split for evaluation')
    parser.add_argument('-data_dir', dest='data_dir', default='data', help='Data directory')
    parser.add_argument('-out_dir', dest='out_dir', default='output', help='Directory to store CESI output')
    parser.add_argument('-reset', dest="reset", action='store_true', default=True,
                        help='Clear the cached files (Start a fresh run)')
    parser.add_argument('-name', dest='name', default=None, help='Assign a name to the run')
    parser.add_argument('-word2vec_path', dest='word2vec_path', default='data/crawl-300d-2M.vec', help='word2vec_path')
    parser.add_argument('-alignment_module', dest='alignment_module', default='swapping', help='alignment_module')

    # system settings
    parser.add_argument('-embed_init', dest='embed_init', default='crawl', choices=['crawl', 'random'],
                        help='Method for Initializing NP and Relation embeddings')
    parser.add_argument('-embed_loc', dest='embed_loc', default='data/crawl-300d-2M.vec',
                        help='Location of embeddings to be loaded')

    parser.add_argument('--use_assume', default=True)
    parser.add_argument('--use_Entity_linking_dict', default=True)
    parser.add_argument('--input', default='entity', choices=['entity', 'relation'])

    parser.add_argument('--use_Embedding_model', default=True)
    parser.add_argument('--relation_view_seed_is_web', default=True)
    parser.add_argument('--view_version', default=1.2)
    parser.add_argument('--use_cluster_learning', default=False)
    parser.add_argument('--use_cross_seed', default=True)
    parser.add_argument('--combine_seed_and_train_data', default=False)
    parser.add_argument('--use_soft_learning', default=False)

    parser.add_argument('--update_seed', default=False)
    parser.add_argument('--only_update_sim', default=True)

    parser.add_argument('--use_bert_update_seeds', default=False)
    parser.add_argument('--use_new_embedding', default=False)

    parser.add_argument('--max_steps', default=50000, type=int)
    parser.add_argument('--turn_to_seed', default=1000, type=int)
    parser.add_argument('--seed_max_steps', default=2000, type=int)
    parser.add_argument('--update_seed_steps', default=6000, type=int)

    parser.add_argument('--get_new_cross_seed', default=False)
    parser.add_argument('--entity_threshold', dest='entity_threshold', default=0.9, help='entity_threshold')
    parser.add_argument('--relation_threshold', dest='relation_threshold', default=0.95, help='relation_threshold')


    parser.add_argument('--step_0_use_hac', default=False)

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data', default=False)
    parser.add_argument('--save_path', default='../output', type=str)

    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true', default=False)
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true', default=False)


    parser.add_argument('-n1', '--single_negative_sample_size', default=32, type=int)
    parser.add_argument('-n2', '--cross_negative_sample_size', default=40, type=int)
    parser.add_argument('-d', '--hidden_dim', default=300, type=int)
    parser.add_argument('-g1', '--single_gamma', default=12.0, type=float)
    parser.add_argument('-g2', '--cross_gamma', default=0.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b1', '--single_batch_size', default=2048, type=int)
    parser.add_argument('-b2', '--cross_batch_size', default=2048, type=int)
    parser.add_argument('-r', '--regularization', default=0.1, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec', default=True)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=12, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=1000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('-embed_dims', dest='embed_dims', default=300, type=int, help='Embedding dimension')

    # word2vec and iteration hyper-parameters
    parser.add_argument('-retrain_literal_embeds', dest='retrain_literal_embeds', default=True,
                        help='retrain_literal_embeds')

    # Clustering hyper-parameters
    parser.add_argument('-linkage', dest='linkage', default='complete', choices=['complete', 'single', 'average'],
                        help='HAC linkage criterion')
    parser.add_argument('-metric', dest='metric', default='cosine',
                        help='Metric for calculating distance between embeddings')
    parser.add_argument('-num_canopy', dest='num_canopy', default=1, type=int,
                        help='Number of caponies while clustering')
    parser.add_argument('-true_seed_num', dest='true_seed_num', default=2361, type=int)
    args = parser.parse_args()

    # if args.name == None: args.name = args.dataset + '_' + args.split + '_' + time.strftime(
    #     "%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
    if args.name == None: args.name = args.dataset + '_' + args.split + '_' + '1'

    args.file_triples = '/triples.txt'  # Location for caching triples
    args.file_entEmbed = '/embed_ent.pkl'  # Location for caching learned embeddings for noun phrases
    args.file_relEmbed = '/embed_rel.pkl'  # Location for caching learned embeddings for relation phrases
    args.file_entClust = '/cluster_ent.txt'  # Location for caching Entity clustering results
    args.file_relClust = '/cluster_rel.txt'  # Location for caching Relation clustering results
    args.file_sideinfo = '/side_info.txt'  # Location for caching side information extracted for the KG (for display)
    args.file_sideinfo_pkl = '/side_info.pkl'  # Location for caching side information extracted for the KG (binary)
    args.file_results = '/results.json'  # Location for loading hyperparameters

    args.out_path = args.out_dir + '/' + args.name  # Directory for storing output
    print('args.out_path:', args.out_path)
    print('args.reset:', args.reset)
    args.data_path = args.data_dir + '/' + args.dataset + '/' + args.dataset + '_' + args.split  # Path to the dataset
    if args.reset: os.system('rm -r {}'.format(args.out_path))  # Clear cached files if requeste
    if not os.path.isdir(args.out_path): os.system(
        'mkdir -p ' + args.out_path)  # Create the output directory if doesn't exist

    clue = CLUE_Main(args)  # Loading KG triples
    clue.get_sideInfo()  # Side Information Acquisition
    clue.embedKG()  # Learning embedding for Noun and relation phrases
    link(0)
    cano(0,args)
    for ite in range(1,11):
        args.split='iter'+str(ite)
        args.max_steps=5000
        args.name = args.dataset + '_' + args.split + '_' + '1'
        args.out_path = args.out_dir + '/' + args.name  # Directory for storing output
        print('args.out_path:', args.out_path)
        args.data_path = args.data_dir + '/' + args.dataset + '/' + args.dataset + '_' + args.split  # Path to the dataset
        if args.reset: os.system('rm -r {}'.format(args.out_path))  # Clear cached files if requeste
        if not os.path.isdir(args.out_path): os.system(
            'mkdir -p ' + args.out_path)  # Create the output directory if doesn't exist
        clue = CLUE_Main(args)  # Loading KG triples
        clue.get_sideInfo()  # Side Information Acquisition
        clue.iterembedKG(ite)  # Learning embedding for Noun and relation phrases
        link(ite)
        cano(ite,args)
