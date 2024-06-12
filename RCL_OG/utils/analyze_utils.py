import cdt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri
from rpy2.robjects.vectors import ListVector
from tqdm import tqdm

#process ordering to graph
def from_order_to_graph(true_position):
    d = len(true_position)
    zero_matrix = np.zeros([d, d])
    for n in range(d - 1):
        row_index = true_position[n]
        col_index = true_position[n + 1:]
        zero_matrix[row_index, col_index] = 1
    return zero_matrix


def cover_rate(graph, graph_true):
    error = graph - graph_true
    return np.sum(np.float32(error > -0.1))

# linear pruning
def graph_prunned_by_coef(graph_batch, X, th=0.3, if_normal=True):  # if_normal=True,th=0.15,if_normal=False,th=0.3
    n, d = X.shape
    reg = LinearRegression()
    W = []

    loss = 0
    for i in range(d):
        col = np.abs(graph_batch[:, i]) > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue
        X_train = X[:, col]

        y = X[:, i]
        reg.fit(X_train, y)
        loss += 0.5 / n * np.sum(np.square(reg.predict(X_train) - y))
        reg_coeff = reg.coef_
        if if_normal:
            reg_coeff = abs(reg_coeff)
            reg_loss = max(reg_coeff) - min(reg_coeff)
            if reg_loss == 0:
                reg_coeff = reg_coeff / max(reg_coeff)
            else:
                reg_coeff = (reg_coeff - min(reg_coeff)) / reg_loss
        cj = 0
        new_reg_coeff = np.zeros(d, )
        for ci in range(d):
            if col[ci]:
                new_reg_coeff[ci] = reg_coeff[cj]
                cj += 1
        W.append(new_reg_coeff)
    return np.float32(np.abs(W) > th)


def graph_prunned_by_coef_2nd(graph_batch, X, th=0.3):
    d = len(graph_batch)
    reg = LinearRegression()
    poly = PolynomialFeatures()
    W = []

    for i in range(d):
        col = graph_batch[:, i] > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]
        X_train_expand = poly.fit_transform(X_train)[:, 1:]
        X_train_expand_names = poly.get_feature_names()[1:]

        y = X[:, i]
        reg.fit(X_train_expand, y)
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )

        for ci in range(d):
            if col[ci]:
                xxi = 'x{}'.format(cj)
                for iii, xxx in enumerate(X_train_expand_names):
                    if xxi in xxx:
                        if np.abs(reg_coeff[iii]) > th:
                            new_reg_coeff[ci] = 1.0
                            break
                cj += 1
        W.append(new_reg_coeff)

    return W

# preliminary neighborhood selectio
def _pns(model_adj, all_samples, num_neighbors, thresh=0.75):
    num_nodes = all_samples.shape[1]
    for node in tqdm(range(num_nodes), desc='Preliminary neighborhood selection'):
        x_other = np.copy(all_samples)
        x_other[:, node] = 0
        extraTree = ExtraTreesRegressor(n_estimators=500)
        extraTree.fit(x_other, all_samples[:, node])
        selected_reg = SelectFromModel(extraTree, threshold="{}*mean".format(thresh), prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False)
        model_adj[:, node] *= mask_selected
    return model_adj


base = rpackages.importr('base')
utils = rpackages.importr('utils')
cam = rpackages.importr('CAM')
mboost = rpackages.importr('mboost')

# CAM pruning
def _pruning(X, G, pruneMethod=robjects.r['selGam'],
             pruneMethodPars=ListVector({'cutOffPVal': 0.001, 'numBasisFcts': 10}), output=False):
    # X is a r matrix
    # G is a python numpy array adj matrix,
    d = G.shape[0]
    X = robjects.r.matrix(numpy2ri.py2rpy(X), ncol=d)
    G = robjects.r.matrix(numpy2ri.py2rpy(G), d, d)
    finalG = robjects.r.matrix(0, d, d)
    for i in range(d):
        parents = robjects.r.which(G.rx(True, i + 1).ro == 1)
        lenpa = robjects.r.length(parents)[0]
        if lenpa > 0:
            Xtmp = robjects.r.cbind(X.rx(True, parents), X.rx(True, i + 1))
            selectedPar = pruneMethod(Xtmp, k=lenpa + 1, pars=pruneMethodPars, output=output)
            finalParents = parents.rx(selectedPar)
            finalG.rx[finalParents, i + 1] = 1
    return np.array(finalG)
def _pns_cam(X,d,pruneMethod=robjects.r['selGamBoost'],
             pruneMethodPars=ListVector({'(atLeastThatMuchSelected ': 0.02, 'atMostThatManyNeighbors ': 10}), output=False):
    X = robjects.r.matrix(numpy2ri.py2rpy(X), ncol=d)
    conG = robjects.r.matrix(numpy2ri.py2rpy(np.ones([d])-np.eye(d)), ncol=d)
    finalG = robjects.r.matrix(0, d, d)
    for i in tqdm(range(d), desc='Preliminary neighborhood selection'):
        parents = robjects.r.which(conG.rx(True, i + 1).ro == 1)
        lenpa = robjects.r.length(parents)[0]
        Xtmp = robjects.r.cbind(X.rx(True, parents), X.rx(True, i + 1))
        selectedPar = pruneMethod(Xtmp, k=lenpa + 1, pars=pruneMethodPars, output=output)
        finalParents = parents.rx(selectedPar)
        finalG.rx[finalParents, i + 1] = 1
    return np.array(finalG)

def pruning_cam(XX, Adj,cutOffPVal=0.001,numBasisFcts=10):
    X2 = numpy2ri.py2rpy(XX)
    Adj = _pruning(X=X2, G=Adj, pruneMethod=robjects.r.selGam,
                   pruneMethodPars=ListVector({'cutOffPVal': cutOffPVal, 'numBasisFcts': numBasisFcts}), output=False)
    return Adj
def pns_gam(XX,atLeastThatMuchSelected=0.02,atMostThatManyNeighbors=10):
    d = XX.shape[1]
    X2 = numpy2ri.py2rpy(XX)
    Adj = _pns_cam(X=X2,d=d,pruneMethod=robjects.r.selGamBoost,
                   pruneMethodPars=ListVector({'atLeastThatMuchSelected': atLeastThatMuchSelected, 'atMostThatManyNeighbors': atMostThatManyNeighbors}), output=False)
    return Adj
# calculate SID
def count_sid(true_dag,pre_dag):
    if true_dag is not np.ndarray:
        true_dag = np.array(true_dag)
    if pre_dag is not np.ndarray:
        pre_dag = np.array(pre_dag)
    return cdt.metrics.SID(true_dag,pre_dag)
#calculate C-SHD
def count_shdcp(true_dag,pre_dag):
    if true_dag is not np.ndarray:
        true_dag = np.array(true_dag)
    if pre_dag is not np.ndarray:
        pre_dag = np.array(pre_dag)
    return cdt.metrics.SHD_CPDAG(true_dag,pre_dag)