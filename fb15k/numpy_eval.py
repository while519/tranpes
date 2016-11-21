
# coding: utf-8

# In[3]:

import pickle
import sys
sys.path.append('/Users/yuwu/Documents/git-repo/tranpes/')
from model import *


# In[4]:

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


# In[71]:

from numpy.linalg import solve, inv

def load_pkl(path):
    f = open(path, 'rb')
    data = sp.csr_matrix(pickle.load(f), dtype=theano.config.floatX)
    return data

def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

def NProjecVec(A, vec):
    """
    This function calculate the projection of vec onto the
    column plane spanned by A
    """
    return  np.dot(np.dot(np.dot(A, inv(np.dot(A.T, A) + 1e-8*np.eye(2))), A.T), vec)

def NRankingScoreIdx(embedding_np, relation_np, idx_h, idx_l, idx_t, subtensorspec=None):
    errh = []
    errt = []


    embedding_sub = embedding_np[:, :subtensorspec]   # we only compute the score for a subset of entities
    for h, l, t in zip(idx_h, idx_l, idx_t):
        hscore = np.zeros(subtensorspec)
        tscore = np.zeros(subtensorspec)
        hvec = embedding_np[:,h]
        tvec = embedding_np[:,t]
        for i in range(subtensorspec):
            x = embedding_np[:,i] + NProjecVec(embedding_np[:,[i, t]], relation_np[:,l]) - tvec
            hscore[i] = -np.sqrt(x.dot(x))
            y = hvec + NProjecVec(embedding_np[:,[h, i]],relation_np[:,l]) - embedding_np[:,i]
            tscore[i] = -np.sqrt(y.dot(y))
        errh += [np.argsort(np.argsort(
                hscore.flatten())[::-1]).flatten() + 1]
        errt += [np.argsort(np.argsort(
                tscore.flatten())[::-1]).flatten() + 1]
    return errh, errt 
    

def FilteredNRankingScoreIdx(embedding_np, relation_np, idx_h, idx_l, idx_t, true_triples, subtensorspec=None):
    errh = []
    errt = []


    embedding_sub = embedding_np[:, :subtensorspec]   # we only compute the score for a subset of entities
    for h, l, t in zip(idx_h, idx_l, idx_t):
        hscore = np.zeros(subtensorspec)
        tscore = np.zeros(subtensorspec)
        hvec = embedding_np[:,h]
        tvec = embedding_np[:,t]
        for i in range(subtensorspec):
            x = embedding_np[:,i] + NProjecVec(embedding_np[:,[i, t]], relation_np[:,l]) - tvec
            hscore[i] = -np.sqrt(x.dot(x))
            y = hvec + NProjecVec(embedding_np[:,[h, i]],relation_np[:,l]) - embedding_np[:,i]
            tscore[i] = -np.sqrt(y.dot(y))
        errh += [np.argsort(np.argsort(
                hscore.flatten())[::-1]).flatten() + 1]
        errt += [np.argsort(np.argsort(
                tscore.flatten())[::-1]).flatten() + 1]
    return errh, errt 
    


# In[16]:

import numpy as np
import scipy.sparse as sp

## main snippet_1
f = open('state.pkl', 'rb')
state = pickle.load(f)
f.close()
state.neval = 10
state.bestvalid = -1
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
simfn = eval(state.simfn + 'sim')


# In[32]:

# main snippet_2
import time

f = open(state.savepath + '/' + 'model400' + '.pkl', 'rb')
embeddings = pickle.load(f)
f.close()
embedding_T = embeddings[0]
relation_T = embeddings[1]

embedding_np = embedding_T.E.eval()
relation_np = relation_T.E.eval()


# In[73]:

# main snippet_3 ----- numpywrap for evaluation
timeref = time.time()
# the argument:: simfn, embedding_np, relation_np, subtensorspec=state.Nbsyn, validhidx, validlidx, validtidx

get_ipython().magic('prun resvalid = NRankingScoreIdx(embedding_np, relation_np, validhidx, validlidx, validtidx, state.Nbsyn)')

print('the evaluation took %s' % (time.time() - timeref))
state.valid = np.mean(resvalid[0] + resvalid[1])
print(state)


# In[50]:

## testing windows
NProjecVec(embedding_np[:,[1, 2]], relation_np[:,1]).shape


# In[ ]:



