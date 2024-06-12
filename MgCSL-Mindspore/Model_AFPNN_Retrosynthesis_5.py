# import torch

#from torch_geometric.graphgym import GATConv
from Model3_Reactant_recommendation_5 import *
# torch.autograd.set_detect_anomaly(True)
from mindspore import Tensor, SparseTensor
import mindspore.ops.operations as P
import numpy as np
from mindspore import Tensor, dtype as mstype
from mindspore import value_and_grad



def get_BondtypeOneHot(bond):
    BondType = bond.GetBondType()
    if str(BondType) == 'SINGLE':
        return Tensor(np.array([1, 0, 0, 0]).astype(np.float32))
    elif str(BondType) == 'DOUBLE':
        return Tensor(np.array([0, 1, 0, 0]).astype(np.float32))
    elif str(BondType) == 'TRIPLE':
        return Tensor(np.array([0, 0, 1, 0]).astype(np.float32))
    elif str(BondType) == 'AROMATIC':
        return Tensor(np.array([0, 0, 0, 1]).astype(np.float32))
    else:
        print('键类型错误')




def normalize_Adj(mx_list):    
    mx = np.matrix(mx_list)
    max_ = mx.max()
    sum_ = 0
    #for x in mx_list:
    #	sum_ += sum(x)
    if max_ == 0:
        print('###################################################################################')
    return (mx/max_).tolist()
    #return (mx/sum_).tolist()


def normalize_Features(F_mx):
    result_T = []
    mx = np.matrix(F_mx)
    mx_T_list = mx.T.tolist()
    for row in mx_T_list:
        max_ = max(row)
        min_ = min(row)
        if max_ == 0:
            result_T.append([x for x in row])
        else:
            result_T.append([x/max_ for x in row])
    result_T = np.matrix(result_T)
    result = result_T.T.tolist()
    return result


def getbondneighbortype(mol, i):
    atomnum = mol.GetNumAtoms()
    bondtype = []
    count = 0
    for j in range(atomnum):
        bond = mol.GetBondBetweenAtoms(i, j)
        if str(bond) == 'None':
            continue
        count += 1
        bondtype.append(get_BondtypeOneHot(bond))
    zero_tensor = np.zeros(4)
    for tensor in bondtype:
        zero_tensor += tensor.asnumpy()
    return zero_tensor / count


def accuracy(output, labels):
    output = output.tolist()
    labels = labels.tolist()
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels),pred

def sparse_mx_to_mindspore_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = Tensor(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = Tensor(sparse_mx.data)
    shape = sparse_mx.shape
    return SparseTensor(indices, values, shape)



def GetBondenergy_margin(Adj,i,j): 
    Neighbor_Bond_i = []
    Neighbor_Bond_j = []
    for num in range(len(Adj[i])):
        if num == j:
            continue
        if Adj[i][num] == 0:
            continue
        Neighbor_Bond_i.append(Adj[i][num])
    for num2 in range(len(Adj[j])):
        if num2 == i:
            continue
        if Adj[j][num2] == 0:
            continue
        Neighbor_Bond_j.append(Adj[j][num2])
    if len(Neighbor_Bond_i) == 0:
        Neighbor_Bond_i.append(Adj[i][j])
    if len(Neighbor_Bond_j) == 0:
        Neighbor_Bond_j.append(Adj[i][j])
    return max(Neighbor_Bond_i)-Adj[i][j]+0.001,max(Neighbor_Bond_j)-Adj[i][j]+0.001



def load_data(Datapath):
    Smiles_P_dict = {}
    Smiles_R1_dict = {}
    Smiles_R2_dict = {}

    BreakBondID_dict = {}
    Feature_Dict_P = {}
    Feature_Dict_R1 = {}
    Feature_Dict_R2 = {}

    Adj_Dic_P = {}
    Adj_Dic_R1 = {}
    Adj_Dic_R2 = {}
    Reaction_class_dict = {}

    count = 0
    with open(Datapath, 'r', encoding='UTF-8') as file_object:
        for line in file_object:
            line = line.rstrip()
            words = line.split('	')

            reaction_class = words[0]
            Smiles_P = words[1]
            Smiles_R1 = words[3]
            Smiles_R2 = words[4]

            BreakBondID = words[2]
            if BreakBondID == 'None':
                continue
            if words[-1] == 'None':
                continue

            Smiles_P_dict[str(count)] = Smiles_P
            Smiles_R1_dict[str(count)] = Smiles_R1
            Smiles_R2_dict[str(count)] = Smiles_R2
            BreakBondID_dict[str(count)] = BreakBondID
            Reaction_class_dict[str(count)] = reaction_class
            count += 1

    Adj_Dic_P_Normal = {}
    for num in range(count):
        Smi_P = Smiles_P_dict[str(num)]
        Smi_R1 = Smiles_R1_dict[str(num)]
        Smi_R2 = Smiles_R2_dict[str(num)]

        features_P = GetFeatureForMoleculeFromSmiles(Smi_P)
        features_R1 = GetFeatureForMoleculeFromSmiles(Smi_R1)
        features_R2 = GetFeatureForMoleculeFromSmiles(Smi_R2)

        Adj_P = GetAdjForMoleculeFromSmiles(Smi_P)
        Adj_R1 = GetAdjForMoleculeFromSmiles(Smi_R1)
        Adj_R2 = GetAdjForMoleculeFromSmiles(Smi_R2)

        features_P = Tensor(normalize_Features(features_P))
        features_R1 = Tensor(normalize_Features(features_R1))
        features_R2 = Tensor(normalize_Features(features_R2))

        Adj_P_Normal = Tensor(normalize_Adj(Adj_P))
        Adj_R1 = Tensor(normalize_Adj(Adj_R1))
        Adj_R2 = Tensor(normalize_Adj(Adj_R2))

        Feature_Dict_P[str(num)] = features_P
        Feature_Dict_R1[str(num)] = features_R1
        Feature_Dict_R2[str(num)] = features_R2

        Adj_Dic_P[str(num)] = Adj_P
        Adj_Dic_R1[str(num)] = Adj_R1
        Adj_Dic_R2[str(num)] = Adj_R2
        Adj_Dic_P_Normal[str(num)] = Tensor(Adj_P_Normal)

    Result = [Smiles_P_dict, Smiles_R1_dict, Smiles_R2_dict, BreakBondID_dict,
              Feature_Dict_P, Feature_Dict_R1, Feature_Dict_R2, Adj_Dic_P,
              Adj_Dic_R1, Adj_Dic_R2, Adj_Dic_P_Normal, Reaction_class_dict]
    return Result


'''
class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, out_feats, heads=8, concat=False)
        #self.conv2 = GATConv(h_feats, out_feats, heads=8, concat=False)
    
    def Adj_To_Edges(self, adj):
        Edges = []
        for i in range(len(adj)):
            for j in range(len(adj[i])):
                if adj[i][j] != 0:
                    Edges.append([i,j])
        return torch.IntTensor(Edges).t()
    
    def forward(self, x,adj):
        edge_index = self.Adj_To_Edges(adj)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        #print()
        #print(edge_index.shape)
        #print(edge_index)
        #x = self.conv2(x, edge_index)
        return x
'''

class GraphConvolution(nn.Cell):
    def __init__(self, in_features, out_features, device, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Dense(in_features, out_features)
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(Tensor(np.zeros(out_features).astype(np.float32)))
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_bias:
            self.bias.set_data(Tensor(np.zeros(self.out_features).astype(np.float32)))
        pass

    def construct(self, input_features, adj):
        support = self.weight(input_features)
        output = P.MatMul()(adj, support)
        output = P.Tanh()(output)
        output = output.clone()
        if self.use_bias:
            return output + self.bias
        else:
            return output

class GraphConvolution2(nn.Cell):
    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolution2, self).__init__()
        self.weight = nn.Dense(in_features, out_features)
        # eye_dd = np.eye(in_features).astype(np.float32)
        # eye = np.eye(in_features).astype(np.float32)
        # eye_tensor = mindspore.Tensor(eye)
        # fill_value = 1.0
        # self.eye = P.Fill()(mindspore.Tensor(fill_value), (in_features, in_features), mindspore.float32)
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(Tensor(np.zeros(out_features).astype(np.float32)))

    # def reset_parameters(self):
    #     if self.use_bias:
    #         self.bias.set_data(Tensor(np.zeros(self.out_features).astype(np.float32)))
    #     pass
    def construct(self, input_features):
        support = self.weight(input_features)
        # output = ops.matmul(support, self.eye)
        output = ops.tanh(support)
        if self.use_bias:
            return output + self.bias
        else:
            return output


def get_Atom_EP(M_Mol,x,EP_list,y):
    atomsnum = M_Mol.GetNumAtoms()
    ep = 0
    smiles_atomx = M_Mol.GetAtomWithIdx(x).GetSymbol()
    if smiles_atomx not in EP_list.keys():
        smiles_atomx = 'unknown'
    ep_atomx = EP_list[smiles_atomx]
    for i in range(atomsnum):
        if i == x:
            continue
        if i == y:
            continue
        if str(M_Mol.GetBondBetweenAtoms(i, x)) == 'None':
            continue
        smiles_atomi = M_Mol.GetAtomWithIdx(i).GetSymbol()
        if smiles_atomi not in EP_list.keys():
            smiles_atomi = 'unknown'
        ep_atomi = EP_list[smiles_atomi]
        ep += ep_atomx - ep_atomi
    return ep




class RedOut(nn.Cell):
    def __init__(self, device, input_dim=29):
        super(RedOut, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 64, device)
        self.gcn2 = GraphConvolution(64, input_dim, device)
        self.gcn3 = GraphConvolution(input_dim, 64, device)
        self.gcn4 = GraphConvolution(64, input_dim, device)
        self.gcn5 = GraphConvolution(input_dim, 64, device)
        self.gcn6 = GraphConvolution(64, input_dim, device)
        self.gcn7 = GraphConvolution2(input_dim, input_dim, device)
        self.weight_BondEnergy = nn.Dense(1, 1)
        self.weight_FC = nn.Dense(input_dim + 34, 1)
        self.weight_BondType = nn.Dense(4, 4)
        self.weight_BondType2 = nn.Dense(4, 4)
        self.liner = nn.Dense(1, 1)

    def reset_parameters(self):
        pass

    def Mpnn1(self, Feature, adj):
        # print(Feature.shape)
        # out = ops.relu(self.gcn1(Feature, adj))
        # out = self.gcn2(out, adj)
        s = (len(adj), 29)
        out = np.ones(s)
        tensor = mindspore.Tensor(out, mindspore.float32)
        return tensor

    def Mpnn2(self, Feature_R1, Feature_R2, adj_R1, adj_R2):
        # Feature = ops.Concat((Feature_R1, Feature_R2), axis=0)
        Feature  = P.Concat(axis=0)(Feature_R1, Feature_R2)
        # len_R1 = len(adj_R1)
        # len_R2 = len(adj_R2)
        length = len(adj_R1) + len(adj_R2)
        width = len(adj_R1) + len(adj_R2)
        # adj = []
        shape = (length,width)
        adj = np.ones(shape)
        # adj = ops.Stack(adj)
        # relu = nn.ReLU()
        # out = self.gcn3(Feature, adj)
        # out = self.gcn4(out, adj)
        s = (length, 29)
        out = np.ones(s)
        tensor = mindspore.Tensor(out, mindspore.float32)
        return tensor

    def Mpnn3(self, Feature, adj):
        s = (len(adj), 29)
        out = np.ones(s)
        tensor = mindspore.Tensor(out, mindspore.float32)
        return tensor

    def Redout_P_Bond_Test(self, output, Smiles, BondID, adj, AtomSymble_One_Hot, EP_list):
        output_RedOut = []
        Labels = []
        Bond_list = []
        M_Mol = Chem.MolFromSmiles(Smiles)
        print(M_Mol.GetBondBetweenAtoms(0, 1))
        Positive_bondfeature = {}
        Atom_symbol = []
        Bond_symbol = []
        # BE_list = []
        # for i in range(output.shape[0]):
        #     for j in range(i + 1, output.shape[0]):
        #         if str(M_Mol.GetBondBetweenAtoms(i, j)) == 'None':
        #             continue
        #         BE_list.append(GetBondenergy(M_Mol, i, j))
        # max_BE = max(BE_list)

        for i in range(output.shape[0]):
            for j in range(i + 1, output.shape[0]):
                # if str(M_Mol.GetBondBetweenAtoms(i, j)) == 'None':
                #     continue
                # bond_type = M_Mol.GetBondBetweenAtoms(i, j).IsInRing()
                # if bond_type == True:
                #     continue
                X = 1
                Mol_Bond_X_Structure = GetBondStructure(M_Mol, BondID, X)
                smi = Chem.MolToSmiles(Mol_Bond_X_Structure)
                Bond_symbol.append(smi)
                Mol_Bond_X_Structure1, Mol_Bond_X_Structure2 = GetAtomStructure(M_Mol, str(i) + ',' + str(j), X)
                smi1 = Chem.MolToSmiles(Mol_Bond_X_Structure1)
                smi2 = Chem.MolToSmiles(Mol_Bond_X_Structure2)
                Atom_symbol.append([smi1, smi2])

                atom1_symbol = M_Mol.GetAtomWithIdx(i).GetSymbol()
                if atom1_symbol not in AtomSymble_One_Hot.keys():
                    atom1_symbol = 'unknown'
                atom1_feature = Tensor(AtomSymble_One_Hot[atom1_symbol], dtype=mstype.float32)

                atom2_symbol = M_Mol.GetAtomWithIdx(j).GetSymbol()
                if atom2_symbol not in AtomSymble_One_Hot.keys():
                    atom2_symbol = 'unknown'
                atom2_feature = Tensor(AtomSymble_One_Hot[atom2_symbol], dtype=mstype.float32)
                # print(type(get_BondtypeOneHot(M_Mol.GetBondBetweenAtoms(i, j))))
                # bond_type_OneHot = self.weight_BondType(get_BondtypeOneHot(M_Mol.GetBondBetweenAtoms(i, j)))
                # Nighbors_BondType_Feature = self.weight_BondType2(getbondneighbortype(M_Mol, i) + getbondneighbortype(M_Mol, j))
                # EP_BondXY = self.liner(Tensor([abs(get_Atom_EP(M_Mol, i, EP_list, j) - get_Atom_EP(M_Mol, j, EP_list, i))], dtype=mstype.float32))
                add = P.Concat(axis=0)(atom1_feature, output[i]) + P.Concat(axis=0)(atom2_feature, output[j])
                # add = ops.Concat((bond_type_OneHot, add), axis=0)
                # add = ops.Concat((add, Nighbors_BondType_Feature), axis=0)
                # add = ops.Concat((add, EP_BondXY), axis=0)
                # BondEnergy = self.weight_BondEnergy(Tensor([GetBondenergy(M_Mol, i, j) / max_BE], dtype=mstype.float32))
                output_RedOut.append(add)
                Bond_list.append(str(i) + ',' + str(j))

                id1, id2 = BondID.split(',')
                if str(i) == id1 and str(j) == id2:
                    Labels.append(1)
                    Positive_bondfeature[id1] = mindspore.ops.stack([output[i]], 0)
                    Positive_bondfeature[id2] = mindspore.ops.stack([output[i]], 0)
                else:
                    Labels.append(0)

        # print('output_RedOut', type(output_RedOut[0]))
        output_RedOut = mindspore.ops.stack(output_RedOut, 0)
        # print('sssss',output_RedOut.shape)
        # ops.Stack(output_RedOut)
        # outputs_list = output_RedOut.asnumpy().tolist()
        # print('output_RedOut',output_RedOut)
        # print('output_RedOut',type(output_RedOut))
        # print('output_RedOut',output_RedOut.shape)
        output_RedOut = output_RedOut[:, 0]
        # print('output_RedOut_shape', output_RedOut)
        Labels = Tensor(Labels, dtype=mstype.float32)
        output_RedOut = P.Sigmoid()(output_RedOut)

        # return Positive_bondfeature, output_RedOut, Labels, Bond_list, Atom_symbol, Bond_symbol
        return Positive_bondfeature, output_RedOut, Labels, Bond_list, Atom_symbol, Bond_symbol

    def Pooling(self, Output):
        l = len(Output)
        result = Output[0]
        for i in range(1, l):
            result += Output[i]
        return P.Div()(result, l)


    def construct(self, EP_list, AtomSymble_One_Hot, Feature_P, adj_P, BreakBondID, Smiles_P, Feature_R1='None', Feature_R2='None', adj_R1='None', adj_R2='None', Neg_Feature_P='None', Neg_adj_P='None'):
        if Feature_R1 == 'None':
            Atom_output_P = self.Mpnn1(Feature_P, adj_P)
            Positive_bond_feature, output_Bond_RedOut_P, Label_P, Bond_list, Atom_symbol, Bond_symbol = self.Redout_P_Bond_Test(Atom_output_P, Smiles_P, BreakBondID, adj_P, AtomSymble_One_Hot, EP_list)
            Atom_feature_id1 = Positive_bond_feature[BreakBondID.split(',')[0]]
            Atom_feature_id2 = Positive_bond_feature[BreakBondID.split(',')[1]]

            return Atom_feature_id1, Atom_feature_id2, output_Bond_RedOut_P, Label_P, Bond_list, Atom_symbol, Bond_symbol
        else:
            Atom_output_R = self.Mpnn2(Feature_R1, Feature_R2, adj_R1, adj_R2)
            R_Feature = self.Pooling(Atom_output_R)
            Neg_Atom_output_P = self.Mpnn3(Neg_Feature_P, Neg_adj_P)
            Neg_P_Feature = self.Pooling(Neg_Atom_output_P)
            Atom_output_P = self.Mpnn1(Feature_P, adj_P)

            Positive_bond_feature, output_Bond_RedOut_P, Label_P, Bond_list, Atom_symbol, Bond_symbol = self.Redout_P_Bond_Test(Atom_output_P, Smiles_P, BreakBondID, adj_P, AtomSymble_One_Hot, EP_list)
            P_Feature = self.Pooling(Atom_output_P)
            Atom_feature_id1 = Positive_bond_feature[BreakBondID.split(',')[0]]
            Atom_feature_id2 = Positive_bond_feature[BreakBondID.split(',')[1]]

            return Atom_feature_id1, Atom_feature_id2, Label_P, Neg_P_Feature, Atom_symbol, Bond_symbol,output_Bond_RedOut_P,R_Feature,P_Feature

def GetTop5Result(Atom_feature_id1, Atom_feature_id2, Value_Bonds, Labels, Bonds_list, atom1_label, atom2_label):  # 输出单个分子中Top5的键及所对应的额外添加基团
    Value_Bonds_list = Value_Bonds.tolist()
    Labels_list = Labels.tolist()
    idx_value = {}
    for m in range(len(Bonds_list)):
        idx_value[str(m)] = Value_Bonds_list[m]
    Result = sorted(idx_value.items(), key=lambda x: x[1], reverse=True)
    Result_label = {}
    Result_Substructure = {}
    for i in range(len(Result)):
        indx = int(Result[i][0])
        label = Labels_list[indx]

        Result_label[i] = label

    Atom_id1 = Atom_feature_id1.tolist()
    Atom_id2 = Atom_feature_id2.tolist()

    indx_id1 = {}
    indx_id2 = {}
    for i in range(len(Atom_id1)):
        indx_id1[str(i)] = Atom_id1[i]
        indx_id2[str(i)] = Atom_id2[i]
    resu_id1 = sorted(indx_id1.items(), key=lambda x: x[1], reverse=True)
    resu_id2 = sorted(indx_id2.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(atom1_label)):
        #output = Positive_bondfeature.tolist()
        id1 = int(resu_id1[j][0])
        id2 = int(resu_id2[j][0])
        label1 = atom1_label[id1]
        label2 = atom2_label[id2]
        Result_Substructure[j] = [label1,label2]
    return Result_label, Result_Substructure
def GetLabelsTensor(Labels_dict, Labels):
    result = []
    for label in Labels:
        result.append(Labels_dict[label])
    return Tensor(result, dtype= mstype.float32)
def GetTopXIndx(list1,x):
    result = []
    idx_value = {}
    for m in range(len(list1)):
        idx_value[str(m)] = list1[m]
    Result = sorted(idx_value.items(), key=lambda x: x[1], reverse=True)
    for i in range(x):
        result.append(int(Result[i][0]))
    return result

import random
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops as P
from mindspore.ops import operations as ops
from mindspore.nn.metrics import Precision
from mindspore.nn.metrics import Recall
from sklearn.metrics import precision_recall_curve, auc, roc_curve






def train(epochs, Datalist, Train_list, Test_list, model, model2, optimizer1, optimizer2, loss_fn, loss_fn_3,
          Labels_dict, Labels_Features_GCN, model3, AtomSymble_One_Hot, EP_list):
    Smiles_P_dict = Datalist[0]
    Smiles_R1_dict = Datalist[1]
    Smiles_R2_dict = Datalist[2]
    BreakBondID_dict = Datalist[3]
    Feature_Dict_P = Datalist[4]
    Feature_Dict_R1 = Datalist[5]
    Feature_Dict_R2 = Datalist[6]
    # Adj_Dic_P = Datalist[7]
    Adj_Dic_R1 = Datalist[8]
    Adj_Dic_R2 = Datalist[9]
    Adj_Dic_P_Normal = Datalist[10]
    # Reaction_class_dict = Datalist[11]
    Result = []

    def net_forward(input, target):
        logit = model(input)
        loss = loss_fn(logit,target)
        return loss
    net_backward = value_and_grad(net_forward, optimizer1.parameters, has_aux=False)
    def train_step(input,target):
        loss, grad = net_backward(input,target)
        optimizer1(grad)
        return loss

    for epoch in range(epochs):
        # optimizer1.clear_gradients()
        # optimizer2.clear_grad()
        outputs_train = []
        Labels_train = []
        P_Feature_list = []
        Neg_P_Feature_list = []
        R_Feature_list = []
        Featuer_atom_id1 = []
        Label_atom_id1 = []

        for i_train in Train_list:
            print(i_train)
            features_P = Feature_Dict_P[str(i_train)]
            features_R1 = Feature_Dict_R1[str(i_train)]
            features_R2 = Feature_Dict_R2[str(i_train)]
            adj_P = Adj_Dic_P_Normal[str(i_train)]
            adj_R1 = Adj_Dic_R1[str(i_train)]
            adj_R2 = Adj_Dic_R2[str(i_train)]
            Smiles_P = Smiles_P_dict[str(i_train)]
            Smiles_R1 = Smiles_R1_dict[str(i_train)]
            Smiles_R2 = Smiles_R2_dict[str(i_train)]
            BreakBondID = BreakBondID_dict[str(i_train)]
            a = random.randint(1, 2)
            Neg_features_P = Feature_Dict_P[str(i_train + a)]
            Neg_adj_P = Adj_Dic_P_Normal[str(i_train + a)]
            Atom_feature_id1, Atom_feature_id2, Label_P, Neg_P_Feature, Atom_symbol, Bond_symbol, output_Bond_RedOut_P, R_Feature,P_Feature = model(
                EP_list, AtomSymble_One_Hot, features_P, adj_P, BreakBondID, Smiles_P, features_R1, features_R2, adj_R1,
                adj_R2, Neg_features_P, Neg_adj_P)

            Atom_feature_id1 = model2(Atom_feature_id1, Labels_Features_GCN)
            Atom_feature_id2 = model2(Atom_feature_id2, Labels_Features_GCN)
            Neg_P_Feature_list.append(Neg_P_Feature)

            outputs_train.append(output_Bond_RedOut_P)
            Labels_train.append(Label_P)
            P_Feature_list.append(P_Feature)
            R_Feature_list.append(R_Feature)
            Featuer_atom_id1.append(Atom_feature_id1.reshape(-1))
            Featuer_atom_id1.append(Atom_feature_id2.reshape(-1))
            data = [Smiles_P, BreakBondID, Smiles_R1, Smiles_R2]
            label_id1, label_id2 = Data_Processing(data, Labels_dict)
            Label_atom_id1.append(Tensor(label_id1, dtype=mindspore.float32))
            Label_atom_id1.append(Tensor(label_id2, dtype=mindspore.float32))
        Featuer_atom_id1 = ops.Concat(axis=0)(Featuer_atom_id1)
        Label_atom_id1 = ops.Concat(axis=0)(Label_atom_id1)

        L7 = loss_fn_3(Featuer_atom_id1, Label_atom_id1)
        outputs_train = ops.Concat(axis=0)(outputs_train)
        Labels_train = ops.Concat(axis=0)(Labels_train)
        P_FeatureS = ops.Concat(axis=0)(P_Feature_list)
        R_FeatureS = ops.Concat(axis=0)(R_Feature_list)
        Neg_P_Features = ops.Concat(axis=0)(Neg_P_Feature_list)

        loss = train_step(outputs_train, Labels_train)

        # L1 = loss_fn(outputs_train, Labels_train)
        # L2 = loss_fn_3(P_FeatureS, R_FeatureS)
        # L4 = loss_fn_3(P_FeatureS, Neg_P_Features)
        # Loss = L1
        # loss,grad = value_and_grad(Loss,)
        # optimizer1.grad.clear()
        # optimizer2.grad.clear()
        # loss,grad =
        # optimizer1.step()
        # optimizer2.step()

        outputs_test = []
        Labels_test = []
        count_top1 = 0
        count_top3 = 0
        count_top5 = 0
        count_all = 0
        count_bond_top1 = 0
        count_bond_top3 = 0
        count_bond_top5 = 0
        count_model2 = 0
        count_bond_no2 = 0

        for i_test in Test_list:
            features_P = Feature_Dict_P[str(i_test)]
            adj_P = Adj_Dic_P_Normal[str(i_test)]
            Smiles_P = Smiles_P_dict[str(i_test)]
            BreakBondID = BreakBondID_dict[str(i_test)]
            Smiles_R1 = Smiles_R1_dict[str(i_test)]
            Smiles_R2 = Smiles_R2_dict[str(i_test)]
            Atom_feature_id1, Atom_feature_id2, Output_P_Bond, label, Bond_list, outputs_list, Atom_symbol, Bond_symbol = model(
                EP_list, AtomSymble_One_Hot, features_P, adj_P, BreakBondID, Smiles_P)
            Atom_feature_id1 = model2(Atom_feature_id1, Labels_Features_GCN)
            Atom_feature_id2 = model2(Atom_feature_id2, Labels_Features_GCN)
            outputs_test.append(Output_P_Bond)
            Labels_test.append(label)
            data = [Smiles_P, BreakBondID, Smiles_R1, Smiles_R2]
            label_id1, label_id2 = Data_Processing(data, Labels_dict)
            Top5_label, Top5_Substructure = GetTop5Result(Atom_feature_id1.reshape(-1), Atom_feature_id2.reshape(-1),
                                                          Output_P_Bond, label, Bond_list, label_id1, label_id2)

            for num1 in range(1):
                if Top5_Substructure[num1][0] == 1:
                    for num2 in range(1):
                        if Top5_Substructure[num2][1] == 1:
                            count_model2 += 1
                            break

            for num in range(1):
                if Top5_label[num] == 1:
                    count_bond_top1 += 1
                    if Top5_Substructure[0][0] == 1 and Top5_Substructure[0][1] == 1:
                        count_top1 += 1
                        break
            if len(Top5_label) < 2:
                count_bond_no2 += 1
                continue
            else:
                for num in range(2):
                    if Top5_label[num] == 1:
                        count_bond_top3 += 1
                        for num1 in range(2):
                            if Top5_Substructure[num1][0] == 1:
                                for num2 in range(2):
                                    if Top5_Substructure[num2][1] == 1:
                                        count_top3 += 1
                                        break
            count_all += 1

        Labels_test = mindspore.ops.Concat(axis=0)(Labels_test)
        outputs_test = mindspore.ops.Concat(axis=0)(outputs_test)

        pred_list = outputs_test.asnumpy().tolist()
        y_true = []
        y_pred = []
        for i in range(len(Labels_test)):
            y_true.append(float(Labels_test[i]))
            y_pred.append(float(pred_list[i]))
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        aupr = auc(recall, precision)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        if (epoch % 1 == 0):
            print("Epoch: {}".format(epoch + 1),
                  "loss_train: {:.4f}".format(Loss.asnumpy()),
                  "aupr_test: {:.4f}".format(aupr),
                  "roc_auc: {:.4f}".format(roc_auc))
        mindspore.save_checkpoint('Model_Save/Class_all_model.ckpt', model)
        mindspore.save_checkpoint('Model_Save/Class_all_model2.ckpt', model2)
        mindspore.save_checkpoint('Model_Save/Class_all_model3.ckpt', model3)
        print('断裂位点识别准确率：',
              "Top1: {:.4f}".format(count_bond_top1 / count_all),
              "Top2: {:.4f}".format(count_bond_top3 / (count_all - count_bond_no2)),
              "Top5: {:.4f}".format(count_bond_top5 / count_all))

        print('总体准确率：',
              "Top1: {:.4f}".format(count_top1 / count_all),
              "Top2: {:.4f}".format(count_top3 / (count_all - count_bond_no2)),
              "Top5: {:.4f}".format(count_top5 / count_all))
        print('model2:', count_model2 / count_all)

        Result.append(
            str(epoch) + ': 断裂位点识别准确率：' + 'Top1:' + str(count_bond_top1 / count_all) + '  Top3:' + str(
                count_bond_top3 / count_all) + '   Top5:' + str(count_bond_top5 / count_all) + '\n')
        Result.append(str(epoch) + ': 总体准确率：' + '  Top1:' + str(count_top1 / count_all) + '  Top3:' + str(
            count_top3 / count_all) + '   Top5:' + str(count_top5 / count_all) + '\n')
        Result.append('\n')












