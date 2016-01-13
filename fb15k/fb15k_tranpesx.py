import os
import pickle
import copy
import numpy as np
import scipy.sparse as sp
import sys
sys.path.append('/home/yuwu/code/tranpes/')       # root directory
from modelx import *
import math
import time


def create_nmat(shape, listidx):  # create the negative samples
    """
    This function create a random sparse index matrix with a given shape. It
    is useful to create negative triplets.

    :param shape: shape of the desired sparse matrix.
    :param listidx: list of index to sample from (default None: it samples from
                    all shape[0] indexes).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """
    if listidx is None:
        listidx = np.arange(shape[0])
    listidx = listidx[np.random.permutation(len(listidx))]
    randommat = sp.lil_matrix((shape[0], shape[1]),
            dtype=np.float32)
    idx_term = 0
    for idx_ex in range(shape[1]):
        if idx_term == len(listidx):
            idx_term = 0
        randommat[listidx[idx_term], idx_ex] = 1
        idx_term += 1
    return randommat.tocsr()

def create_nvec(shape, listidx):
    listidx = listidx[np.random.permutation(len(listidx))]
    randomvec = np.zeros(shape, dtype=np.int32)
    idx_term = 0
    for idx_ex in range(shape[0]):
        if idx_term == len(listidx):
            idx_term = 0
        randomvec[idx_ex] = listidx[idx_term]
        idx_term += 1
    return randomvec


class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z


def load_pkl(path):
    return sp.csr_matrix(pickle.load(open(path, 'rb')),
                         dtype=np.float32)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

def idxset(spmat):  # Nbent x batchsize
    rows, cols = spmat.nonzero()
    return np.unique(rows)


# ----------------------------------------------------------------------------------------------


# Experiment function --------------------------------------------------------------------------
def FB15kexp(state):
    # Show experiment parameters
    print(state)
    np.random.seed(state.seed)

    # Positives
    trainhmat = load_pkl(state.datapath + 'FB15k-train-hs.pkl')
    trainlmat = load_pkl(state.datapath + 'FB15k-train-ls.pkl')
    traintmat = load_pkl(state.datapath + 'FB15k-train-ts.pkl')
    if state.op == 'tranPES':
        trainhmat = trainhmat[:state.Nbsyn, :]
        trainlmat = trainlmat[-state.Nbrel:, :]
        traintmat = traintmat[:state.Nbsyn, :]

    # Valid set
    validhmat = load_pkl(state.datapath + 'FB15k-valid-hs.pkl')
    validlmat = load_pkl(state.datapath + 'FB15k-valid-ls.pkl')
    validtmat = load_pkl(state.datapath + 'FB15k-valid-ts.pkl')
    if state.op == 'tranPES':
        validhmat = validhmat[:state.Nbsyn, :]
        validlmat = validlmat[-state.Nbrel:, :]
        validtmat = validtmat[:state.Nbsyn, :]

    # Test set
    testhmat = load_pkl(state.datapath + 'FB15k-test-hs.pkl')
    testlmat = load_pkl(state.datapath + 'FB15k-test-ls.pkl')
    testtmat = load_pkl(state.datapath + 'FB15k-test-ts.pkl')
    if state.op == 'tranPES':
        testhmat = testhmat[:state.Nbsyn, :]
        testlmat = testlmat[-state.Nbrel:, :]
        testtmat = testtmat[:state.Nbsyn, :]

    # Index conversion
    trainhidx = convert2idx(trainhmat)[: state.neval]
    trainlidx = convert2idx(trainlmat)[: state.neval]
    traintidx = convert2idx(traintmat)[: state.neval]
    validhidx = convert2idx(validhmat)[: state.neval]
    validlidx = convert2idx(validlmat)[: state.neval]
    validtidx = convert2idx(validtmat)[: state.neval]
    testhidx = convert2idx(testhmat)[: state.neval]
    testlidx = convert2idx(testlmat)[: state.neval]
    testtidx = convert2idx(testtmat)[: state.neval]

    idxh = convert2idx(trainhmat)
    idxl = convert2idx(trainlmat)
    idxt = convert2idx(traintmat)
    idxvh = convert2idx(validhmat)
    idxvl = convert2idx(validlmat)
    idxvt = convert2idx(validtmat)
    idxth = convert2idx(testhmat)
    idxtl = convert2idx(testlmat)
    idxtt = convert2idx(testtmat)

    true_triples = np.concatenate([idxh, idxvh, idxth, idxl, idxvl, idxtl, idxt, idxvt, idxtt]).reshape(3,
                                                                                                        idxh.shape[0] +
                                                                                                        idxvh.shape[0] +
                                                                                                        idxth.shape[
                                                                                                            0]).T

    # Embeddings
    simfn = eval(state.simfn + 'sim')
    embedding = Embeddings(np.random, state.ndim, state.Nbsyn)
    lembedding = Embeddings(np.random, state.ndim, state.Nbrel)
    embeddings = [embedding, lembedding]


    #
    out = []
    outb = []
    state.bestvalid = -1
    batchsize = math.floor(trainhmat.shape[1]/state.nbatches)
    print('BEGIN TRAINING')
    timeref = time.time()
    for epoch_count in range(1, state.totepochs + 1):
        order = np.random.permutation(trainhmat.shape[1])

        idxh = idxh[order]
        idxl = idxl[order]
        idxt = idxt[order]


        for j in range(state.nbatches): # 1 x batchsize
            hbidx = idxh[j*batchsize : (j+1)*batchsize]
            lbidx = idxl[j*batchsize : (j+1)*batchsize]
            tbidx = idxt[j*batchsize : (j+1)*batchsize]

            hnidx = create_nvec(hbidx.shape, np.arange(state.Nbsyn))
            tnidx = create_nvec(tbidx.shape, np.arange(state.Nbsyn))

            subsetR = np.unique(lbidx)
            subsetE = np.unique(np.concatenate((hbidx, tbidx, hnidx,tnidx)))

            outtmp = TranPES(state, simfn, embeddings, hbidx, lbidx, tbidx, hnidx, tnidx, subsetE, subsetR)
            out += [outtmp[0]/batchsize]
            outb += [outtmp[1]]
            print(out)

            #embeddings[0].normalize()

        print('-- EPOCH %s (%s seconds per epoch):' % (epoch_count, (time.time() - timeref)))
        timeref = time.time()
        print('Cost mean: %s +/- %s      updates: %s%% ' % (np.mean(out), np.std(out), np.mean(outb)*100))
        out = []
        outb = []



        if (epoch_count % state.test_all) ==0 and epoch_count >= 400:
            # save current model
            state.nbepochs = epoch_count
            f = open(state.savepath + '/' + 'model' + str(state.nbepochs) + '.pkl', 'wb')
            pickle.dump(embeddings, f, -1)
            f.close()
            print('The saving took %s seconds' % (time.time() - timeref))
            timeref = time.time()

    f = open('state.pkl', 'wb')
    pickle.dump(state, f, -1)
    f.close()
    return





def launch(datapath='xdata/', dataset='FB15k', Nbent=16296, Nbsyn=14951, Nbrel=1345,
           op='tranPES', simfn='L2', ndim=50, marge=0.5, lremb=0.01, lrparam=0.01,
           nbatches=100, totepochs=500, test_all=10, neval=1000, savepath='FB15k_tranPESx',
           seed=123, alpha=1., beta=1.):
    state = DD()
    state.datapath = datapath
    state.dataset = dataset
    state.Nbent = Nbent
    state.Nbsyn = Nbsyn
    state.Nbrel = Nbrel
    state.op = op
    state.simfn = simfn
    state.ndim = ndim
    state.marge = marge
    state.lremb = lremb
    state.lrparam = lrparam
    state.nbatches = nbatches
    state.totepochs = totepochs
    state.test_all = test_all
    state.neval = neval
    state.savepath = savepath
    state.seed = seed
    state.alpha = alpha
    state.beta = beta

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    FB15kexp(state)


if __name__ == '__main__':
    launch(test_all=10, totepochs=500, neval=1, marge=0.3, nbatches=100, alpha=1., beta=0.01, lremb=0.002, lrparam=0.002)
