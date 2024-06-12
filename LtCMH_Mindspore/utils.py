import mindspore.ops as ops
import mindspore.numpy as mnp
from config import opt

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - ops.matmul(B1, B2.transpose()))
    return distH

def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (ops.matmul(q_L, retrieval_L.transpose()) > 0).squeeze().astype(mnp.float32)
        tsum = ops.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = ops.argsort(hamm)
        ind = ind.squeeze()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = mnp.arange(1, total + 1).astype(mnp.float32)
        tindex = mnp.squeeze(ops.nonzero(gnd)[:total]).astype(mnp.float32) + 1.0
        map = map + ops.mean(count / tindex)
    map = map / num_query
    return map
