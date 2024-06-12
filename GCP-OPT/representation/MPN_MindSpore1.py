import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import dtype as mstype
from mindspore import Tensor
import numpy as np
from mindspore import Tensor
import numpy as np
class MPN(nn.Cell):
    def __init__(self, iterNum=5, input_dim=256):
        super(MPN, self).__init__()
        self.iterNum = iterNum
        self.output_dim = int(input_dim*(input_dim+1)/2)
        self.sqrtm = Sqrtm(self.iterNum)
        self.triuvec = Triuvec()

    def construct(self, x):
        x = self.sqrtm(x)
        x = self.triuvec(x)
        # x = mindspore.ops.reshape(x, (x.shape[0], -1))
        return x

class Covpool(nn.Cell):
    def __init__(self):
        super(Covpool, self).__init__()
        self.matmul = P.BatchMatMul()
        self.I_hat = None

    def construct(self, input):
        num_elements = 64*128*14*14
        sequence = mindspore.numpy.arange(0.0001, 0.0001*num_elements + 0.0001, 0.0001,dtype=mindspore.float32)
        tensor = sequence.reshape(64, 128, 14, 14)
        x = mindspore.Tensor(tensor, mindspore.float32)   
        #x = input
        batchSize, dim, h, w = P.Shape()(x)
        M = h*w
        x = x.view(batchSize, dim, M)
        I_hat = (-1./M/M)*mindspore.ops.ones((M,M),mindspore.float32) + \
                (1./M)*mindspore.ops.eye(M,M,mindspore.float32)
        I_hat = mindspore.ops.tile(I_hat.view(1,M,M), (batchSize, 1, 1))

        x = x.asnumpy()
        print("x reshape")
        print(x)
        #y = x.bmm(I_hat).bmm(x.transpose(0,2,1))
        y=ops.matmul(ops.matmul(x,I_hat),x.transpose(0,2,1))
        print("bmm")
        print(ops.matmul(x,I_hat).shape)
        print(ops.matmul(x,I_hat))
        print("transpose")
        print(x.transpose(0,2,1))
        print(y )
        self.I_hat = I_hat
        return y

    def bprop(self, x, out, dout):
        I_hat = self.I_hat
        print("input" )
        print(input )
        input = x
        I_hat = I_hat.astype(x.dtype)
        dout = dout.astype(x.dtype)
        batchSize, dim, h, w = P.Shape()(input)
        M = h*w
        input = input.reshape(batchSize,dim,M)
        print(input)
        grad_input = (dout + dout.transpose(0,2,1)).bmm(input).bmm(I_hat)
        print("grad_input 2" )
        print(grad_input )
        grad_input = grad_input.reshape(batchSize, dim, h, w)
        print("grad_input 3" )
        print(grad_input )        
        return grad_input

class Sqrtm(nn.Cell):
    def __init__(self, iterN):
        super(Sqrtm, self).__init__()
        #self.iterN = mindspore.Parameter(Tensor(1, dtype=mindspore.int32))
        #print(iterN)
        #self.ind= mindspore.Parameter(Tensor(np.ones(8256),mindspore.float32))
        self.iterN = iterN
        self.A = None
        self.ZY = None
        self.normA = None
        self.Y = None
        self.Z = None
        self.matmul = P.BatchMatMul()
        self.print = mindspore.ops.Print()

    def construct(self, input):
        x = input
        x = x.astype(mindspore.float32)
        iterN = self.iterN
        batchSize, dim, _ = P.Shape()(input)
        dtype = x.dtype
        I3 = mindspore.ops.tile(3.0*mindspore.ops.eye(dim,dim,x.dtype).view(1, dim, dim),(batchSize,1,1)).astype(dtype)
        normA = (1.0/3.0)*x.bmm(I3).sum(axis=1).sum(axis=1)
        A = mindspore.ops.div(x,normA.view(batchSize,1,1).expand_as(x))
        Y = mindspore.ops.zeros((batchSize, iterN, dim, dim),x.dtype).astype(dtype)
        Z = mindspore.ops.tile(mindspore.ops.eye(dim,dim,dtype).view(1,dim,dim),(batchSize,iterN,1,1)).astype(dtype)
        if iterN < 2:
            ZY = 0.5*(I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
                ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
                Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
                Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            YZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
        y = YZY*mindspore.ops.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        y = y.astype(mindspore.float32)
        self.iterN = iterN
        self.A = A
        self.ZY = YZY
        self.normA = normA
        self.Y = Y
        self.Z = Z
        return y

    def bprop(self, x, out, dout):
        A = self.A
        ZY = self.ZY
        normA = self.normA
        Y = self.Y
        Z = self.Z
        iterN = self.iterN
        x = x.astype(mindspore.float32)
        grad_output = dout.astype(x.dtype)
        A = A.astype(x.dtype)
        ZY = ZY.astype(x.dtype)
        normA = normA.astype(x.dtype)
        Y = Y.astype(x.dtype)
        X = Z.astype(x.dtype)
        batchSize = x.shape[0]
        dim = x.shape[1]
        dtype = grad_output.dtype
        t = mindspore.ops.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postCom = grad_output*t
        der_postComAux = mindspore.ops.div((grad_output*ZY).sum(axis=1).sum(axis=1),2*mindspore.ops.sqrt(normA))
        I3 = mindspore.ops.tile(3.0*mindspore.ops.eye(dim,dim,x.dtype).view(1, dim, dim),(batchSize,1,1)).astype(dtype)

        if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                        Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
                YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
                ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
                dldY_ = 0.5*(dldY.bmm(YZ) -
                        Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) -
                            ZY.bmm(dldY))
                dldZ_ = 0.5*(YZ.bmm(dldZ) -
                        Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(0,2,1)
        grad_input = mindspore.ops.div(der_NSiter,normA.view(batchSize,1,1).expand_as(x))
        grad_aux = mindspore.ops.mul(der_NSiter,x)
        grad_aux = grad_aux.sum(axis=1).sum(axis=1)
        for i in range(batchSize):
            grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *mindspore.numpy.diag(mindspore.ops.ones(dim,dtype)).astype(dtype)
        self.print("sqrtm"+str(grad_input))
        return grad_input

class Triuvec(nn.Cell):
    def __init__(self):
        super(Triuvec, self).__init__()
        self.ind=mindspore.Tensor(np.ones(8256),mindspore.float32)
        #self.ind=mindspore.Parameter(Tensor(np.ones(8256),mindspore.float32))

    def construct(self,x):
        batchSize = x.shape[0]
        dim = x.shape[1]
        dtype=x.dtype
        x = x.reshape(batchSize, dim*dim)
        one=mindspore.ops.Ones()
        I = mindspore.numpy.triu(one((dim,dim),dtype)).reshape(dim*dim)
        index =mindspore.ops.nonzero(I)
        index=index.view(int(dim*(dim+1)/2))
        zero=mindspore.ops.Zeros()
        y = zero((batchSize,int(dim*(dim+1)/2)),dtype)
        y = x[:,index]
        self.ind=index
        return y

    def bprop(self,x,out,dout):
        index=self.ind
        print("dout"+str(dout))        
        batchSize = x.shape[0]
        dim = x.shape[1]
        dtype=x.dtype
        zer=mindspore.ops.Zeros()
        grad_input = zer((batchSize,dim*dim),dtype)
        grad_input[:,index] = dout
        output= grad_input.reshape(batchSize,dim,dim)
        print("output"+str(output))
        return output

if __name__ == '__main__':
    # x = Tensor(np.random.random(size=(64, 10 , 5, 5)), mindspore.float32)
    num_elements = 128*128*128
    sequence = mindspore.numpy.arange(1, num_elements + 1, dtype=mindspore.float32)

    tensor = sequence.reshape(128, 128, 128)
    x = Tensor(np.random.random(size=(64, 10 , 5, 5)), mindspore.float32)
    #x = Tensor(np.ones(size=(64, 10 , 5, 5)), mindspore.float32)
    print(x.size())
    # x = Covpool()(x)
    model = MPN()
    y = model(x)
    print(y)
