
import mindspore.nn as nn
from mindspore import Parameter, Tensor
import mindspore.ops.operations as P
from Model_AFPNN_Retrosynthesis_5 import *


def remove_zero_rows_and_columns(matrix):
    if not matrix:
        return matrix

    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # 寻找全零行和列
    zero_rows = set()
    zero_cols = set()

    for row in range(num_rows):
        for col in range(num_cols):
            if matrix[row][col] != 0:
                break  # 如果找到非零元素，跳出内层循环
        else:
            zero_rows.add(row)  # 如果内层循环完成而没有跳出，说明全是零


    for col in range(num_cols):
        for row in range(num_rows):
            if matrix[row][col] != 0:
                break  # 如果找到非零元素，跳出内层循环
        else:
            zero_cols.add(col)  # 如果内层循环完成而没有跳出，说明全是零

    # 删除全零行
    new_matrix = [matrix[i] for i in range(num_rows) if i not in zero_rows]

    # 删除全零列
    new_matrix = [[new_matrix[i][j] for j in range(num_cols) if j not in zero_cols] for i in range(len(new_matrix))]

    return new_matrix


def Listadd(list1, list2):
	result = []
	for i in range(len(list1)):
		result.append(list1[i] + list2[i])
	return result


def GetLabelsrelationship(Labels,Labels_dict_OneHot,Data):
	txt = open('output/离去基图.txt', 'w', encoding='UTF-8')
	LabelsDict_relasionship = {}
	for i in range(len(Labels)):
		LabelsDict_relasionship[Labels[i]] = [0.0]*len(Labels)
	for num in range(len(Data[0])):
		Smiles_P = Data[0][str(num)]
		Smiles_R1 = Data[1][str(num)]
		Smiles_R2 = Data[2][str(num)]
		BreakBondID = Data[3][str(num)]
		data = [Smiles_P, BreakBondID, Smiles_R1, Smiles_R2]
		label1, label2 = Data_Processing(data, Labels_dict_OneHot)
		x = 0
		y = 0
		for i in range(len(label1)):
			if label1[i] == 1:
				x = i
			if label2[i] == 1:
				y = i
		LabelsDict_relasionship[Labels[x]] = Listadd(LabelsDict_relasionship[Labels[x]],label2)
		LabelsDict_relasionship[Labels[y]] = Listadd(LabelsDict_relasionship[Labels[y]],label1)
	
	for x,y in LabelsDict_relasionship.items():
		print(x,y)
	Adj_Labels = []
	mini_labels = []
	for num in range(len(Labels)):
		if sum(LabelsDict_relasionship[Labels[num]]) != 0:
			mini_labels.append(Labels[num])
		Adj_Labels.append(LabelsDict_relasionship[Labels[num]])

	matrix = remove_zero_rows_and_columns(Adj_Labels)
	
	# 获取小邻接矩阵
	mini_adj = []
	mini_Labels_dict = {}
	print('mini_labels', len(mini_labels))
	# coun = 1
	for coun in range(len(mini_labels)):
		row = [0.0]*(coun) +  [1.0] + [0.0]*(len(mini_labels) - coun - 1)
		if mini_labels[coun] not in mini_Labels_dict.keys():
			mini_Labels_dict[mini_labels[coun]] = row
		mini_adj.append(row)
	v = 0
	result = []
	for i in matrix:
		normalized_list = [x / sum(i) for x in i]
		count = sum(i)
		result.append(normalized_list)
		for r in normalized_list:
			txt.write(str(r)+'	')
		txt.write('\n')
		v += count
		#print(i,count)
	print(v)
	return mini_Labels_dict, mini_labels,len(mini_labels),mini_adj,result




class GraphConvolution(nn.Cell):
	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.weight = nn.Dense(in_features, out_features)
		self.use_bias = bias
		if self.use_bias:
			self.bias = Parameter(Tensor(np.zeros(out_features).astype(np.float32)))
		# self.reset_parameters()

	# def reset_parameters(self):
	# 	if self.use_bias:
	# 		self.bias.set_data(Tensor(np.zeros(out_features).astype(np.float32)))
	# 	pass

	def construct(self, input_features, adj):
		support = self.weight(input_features)
		output = P.MatMul()(adj, support)
		output = P.Tanh()(output)
		if self.use_bias:
			return output + self.bias
		else:
			return output


class Model3(nn.Cell):
	def __init__(self, input_dim, output_dim):
		super(Model3, self).__init__()
		self.gcn1 = GraphConvolution(input_dim, input_dim * 2)
		self.gcn2 = GraphConvolution(input_dim * 2, output_dim)
		self.reset_parameters()
		pass

	def reset_parameters(self):
		pass

	def construct(self, Nodes_Feature, Adj):
		output = self.gcn1(Nodes_Feature, Adj)
		output = self.gcn2(output, Adj)
		return output


'''
Labels_dict = {}
Labels = []
with open('Data/Labels-957.txt', 'r',encoding='UTF-8') as file_object:   
	for line in file_object: 
		line = line.rstrip()
		if line not in Labels:
			Labels.append(line)
count = 1
for num1 in range(len(Labels)):
	if Labels[num1] not in Labels_dict.keys():
		Labels_dict[Labels[num1]] = [0.0]*(count-1) +  [1.0] + [0.0]*(len(Labels) - count)
		count += 1


Datapath = 'Data/Data(Smiles_BreakBond).txt'
Data = load_data(Datapath) # S_P_d = {}, S_R1_d = {}, S_R2_d = {}, Bond_dict = {},F_D_P = {},F_D_R1 = {} F_D_R2 = {},A_D_P = {}, A_D_R1 = {},A_D_R2 = {}
print()
adj=GetLabelsrelationship(Labels,Labels_dict,Data)
for i in adj:
	print(i)

'''



		
	
