import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import *
import numpy as np
import scipy.sparse as sp

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
    f = open(path, 'rb')
    data = sp.csr_matrix(pickle.load(f), dtype=theano.config.floatX)
    return data

def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

# the main things
f = open('best_state.pkl', 'rb')
state = pickle.load(f)
f.close()
state.neval = 1000
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
                                                                                                  idxth.shape[0]).T

f = open(state.savepath + '/' +  state.bestmodel + '.pkl', 'rb')
embeddings = pickle.load(f)
f.close()

embedding_T = embeddings[0]
relation_T = embeddings[1]

embedding_np = embedding_T.E.eval()
relation_np = relation_T.E.eval()                                                                                                       


f = open(state.bestmodel + '_true_triplets.txt', 'wb')
np.savetxt(f,true_triples, '%d')
f.close()

f = open(state.bestmodel + '_embedding.txt', 'wb')
np.savetxt(f,embedding_np)
f.close()

f = open(state.bestmodel + '_relation.txt', 'wb')
np.savetxt(f,relation_np)
f.close()

f = open(state.bestmodel + '_idxth.txt', 'wb')
np.savetxt(f, idxth, '%d')
f.close()

f = open(state.bestmodel + '_idxtl.txt', 'wb')
np.savetxt(f, idxtl, '%d')
f.close()

f = open(state.bestmodel + '_idxtt.txt', 'wb')
np.savetxt(f, idxtt, '%d')
f.close()

print('saved sucessfully...')