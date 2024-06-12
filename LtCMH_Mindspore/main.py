from config import opt
from data_handler import *
import numpy as np
from tqdm import tqdm
from models import AE,img_module_nus,txt_module
from models.AE import AutoEncoder
from models.img_module_nus import ImgModuleNus
from models.txt_module import TxtModule
from utils import calc_map_k
import xlrd
import mindspore
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import nn
from mindspore import Tensor, CSRTensor, COOTensor
from mindspore import Parameter

def train(**kwargs):
    opt.parse(kwargs)

    Xtest, Xtrain, Ltest, Ltrain,Ytest, Ytrain= load_data(opt.data_path)

    y_dim = 5018
    y2_dim= 4096

    X,X1,Y,Y1,L,L1 = split_data(Xtest,Xtrain,Ytest,Ytrain,Ltest,Ltrain)

    print('...loading and splitting data finish')

    img_model_nus= ImgModuleNus(y2_dim,opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)
    en_decoder = AutoEncoder()

    #使用mindspore加载模型权重
    param_dict = mindspore.load_checkpoint("./data/encoder64nus.ckpt")
    param_not_load, _ = mindspore.load_param_into_net(en_decoder, param_dict)
    #这里param_not_load表示未被成功加载的参数，为空列表表示全部加载成功

    #使用mindspore的Tensor张量从numpy数组进行生成
    train_L = Tensor(L1['train'])
    train_x = Tensor(X1['train'])
    train_y = Tensor(Y1['train'])

    query_L = Tensor(L['query'])
    query_x = Tensor(X['query'])
    query_y = Tensor(Y['query'])

    retrieval_L = Tensor(L1['retrieval'])
    retrieval_x = Tensor(X1['retrieval'])
    retrieval_y = Tensor(Y1['retrieval'])

    #训练超参
    num_train = 13375
    batch_size = opt.batch_size
    lr = opt.lr

    #利用mindspore的Tensor张量根据数据直接生成随机初始化的值
    F_buffer = Tensor(np.random.randn(num_train, opt.bit).astype(np.float32))
    G_buffer = Tensor(np.random.randn(num_train, opt.bit).astype(np.float32))

    #利用mindspore生成数据
    B = Tensor(np.sign(F_buffer.asnumpy() + G_buffer.asnumpy()).astype(np.float32))

    
    #使用mindspore的nn.sgd优化器，通过model.trainable_params()方法获得模型的可训练参数，并传入学习率超参来初始化优化器
    optimizer_img = nn.SGD(img_model_nus.trainable_params(), learning_rate=lr)
    optimizer_txt = nn.SGD(txt_model.trainable_params(), learning_rate=lr)

    learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)

    result = {
        'loss': []
    }

    #利用mindspore生成全1向量
    ones = mnp.ones((batch_size, 1))
    ones_ = mnp.ones((num_train - batch_size, 1))
    # 创建mindspore算子
    matmul = ops.MatMul()
    exp = ops.Exp()
    sum_op = ops.ReduceSum()
    log = ops.Log()
    pow = ops.Pow()

    max_mapi2t = max_mapt2i = 0.

    def forward_fn(cur,FG,S,B):
        # 计算 theta
        theta = 0.5 * matmul(cur, FG.transpose())
        # 计算 logloss
        logloss = -sum_op(S * theta - log(1.0 + exp(theta)))
        # 计算 quantization
        quantization = sum_op(pow(B[ind, :] - cur, 2))
        # 计算 balance
        balance = sum_op(pow(cur.transpose() @ ones + FG[unupdated_ind].transpose() @ ones_, 2))
        # 计算 loss
        loss = logloss + opt.gamma * quantization + opt.eta * balance
        loss /= (batch_size * num_train)

        return loss

    for epoch in range(opt.max_epoch):
        # train image net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)
            #利用mindspore声明张量
            sample_L = Tensor(train_L[ind, :],dtype=mindspore.float32)
            image = Tensor(train_x[ind, :].reshape(batch_size, 1, 1, -1), dtype=mindspore.float32)
            
            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            cur_f = img_model_nus(image)  # cur_f: (batch_size, bit)
            imgindi, txtindi, common, imgout, txtout = en_decoder(cur_f, G_buffer[ind,:])
            cur_f=cur_f+common+imgindi
            F_buffer[ind, :] = cur_f.data
            # 创建参数F和G
            F = Parameter(F_buffer, name="F")
            G = Parameter(G_buffer, name="G")
            #利用mindspore的优化器更新梯度
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer_img.parameters, has_aux=True)
            (loss),grad = grad_fn(cur_f,G,S,B)
            optimizer_img(grad)
        # train txt net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)
            #利用mindspore声明张量
            sample_L = Tensor(train_L[ind, :],dtype=mindspore.float32)
            text = train_y[ind, :]
            text = ops.ExpandDims()(text, 1)  # 在第1维度添加维度
            text = ops.ExpandDims()(text, -1)  # 在最后一维度添加维度
            text = Tensor(mnp.array(text, dtype=mnp.float32))       

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            cur_g = txt_model(text)  # cur_f: (batch_size, bit)
            imgindi, txtindi, common, imgout, txtout = en_decoder(F_buffer[ind,:], cur_g)
            # hg, ghat = hashmodelg(cur_g+0.4*common+0.6*txtindi)
            cur_g=cur_g+common+txtindi
            G_buffer[ind, :] = cur_g.data
            # 创建参数F和G
            F = Parameter(F_buffer, name="F")
            G = Parameter(G_buffer, name="G")
            #利用mindspore的优化器更新梯度
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer_txt.parameters, has_aux=True)
            (loss),grad = grad_fn(cur_g,F,S,B)
            optimizer_txt(grad)
            

        if opt.valid:
            mapi2t, mapt2i = valid(img_model_nus, txt_model,query_x, retrieval_x, query_y, retrieval_y,
                                   query_L, retrieval_L,en_decoder)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                #利用mindspore的save方法保存训练好的模型参数
                mindspore.save_checkpoint(img_model_nus, "img_model.ckpt")
                mindspore.save_checkpoint(txt_model, "txt_model.ckpt")


        lr = learning_rate[epoch + 1]
        

        # set learning rate
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr


    print('...training procedure finish')


def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L,encoder):
    qBX,qBY = generate_code(img_model,txt_model,query_x,query_y,opt.bit,encoder)

    rBX,rBY = generate_code(img_model,txt_model,retrieval_x,retrieval_y,opt.bit,encoder)


    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def split_data(Xtest,Xtrain,Ytest,Ytrain,Ltest,Ltrain):
    X = {}
    X1={}
    X['query'] = Xtest[0: opt.query_size]
    X1['train'] = Xtrain[0: opt.training_size]
    X1['retrieval'] = Xtrain[0: opt.database_size]

    Y = {}
    Y1={}
    Y['query'] = Ytest[0: opt.query_size]
    Y1['train'] = Ytrain[0: opt.training_size]
    Y1['retrieval'] = Ytrain[0: opt.database_size]

    L = {}
    L1={}
    L['query'] = Ltest[0: opt.query_size]
    L1['train'] = Ltrain[0: opt.training_size]
    L1['retrieval'] = Ltrain[0:opt.database_size]

    return X,X1, Y,Y1, L,L1

def generate_code(img_model, txt_model, X, Y, bit, encoder):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = mnp.linspace(0, num_data - 1, num_data).astype(int)
    Bi = mnp.zeros((num_data, bit), dtype=mnp.float32)
    Bt = mnp.zeros((num_data, bit), dtype=mnp.float32)

    img_model.set_train(False)
    txt_model.set_train(False)
    encoder.set_train(False)

    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = mnp.array(X[ind, :].reshape(-1, 1, 1, X.shape[-1]), dtype=mnp.float32)
        text = mnp.array(Y[ind].reshape(-1, 1, 1, Y.shape[-1]), dtype=mnp.float32)

        cur_f = img_model(image)
        cur_g = txt_model(text)

        imgindi, txtindi, common, _, _ = encoder(cur_f, cur_g)

        cur_f = cur_f + common + imgindi
        cur_g = cur_g + common + txtindi

        Bi[ind, :] = cur_f.asnumpy()
        Bt[ind, :] = cur_g.asnumpy()

    Bi = mnp.sign(Bi)
    Bt = mnp.sign(Bt)
    return Bi, Bt

def calc_neighbor(label1, label2):
    # 创建mindspore算子
    matmul = ops.MatMul()
    transpose = ops.Transpose()
    greater = ops.Greater()
    cast = ops.Cast()

    # 计算相似矩阵
    Sim = greater(matmul(label1, transpose(label2)), 0).astype(mindspore.float32) 
    return Sim

if __name__ == '__main__':
    # import fire
    # fire.Fire()
    train()