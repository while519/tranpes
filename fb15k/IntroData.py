
# coding: utf-8

# In[113]:

import pickle
import scipy.sparse as sp
import copy
import theano
import numpy as np
import sys
sys.path.append('/Users/yuwu/Desktop/tranpes')
from model import *
import time


# In[102]:

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
                         dtype=theano.config.floatX)

def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

def convert2label(avghl, avgtl):
    if avghl <= 1.5:
        strt = '1'
    else:
        strt = 'Many'

    if avgtl <= 1.5:
        strh = '1'
    else: 
        strh = 'Many'
    return '-To-'.join((strh, strt))


# In[112]:

# load the state of the model
f = open('best_state.pkl', mode='rb')
state = pickle.load(f)
f.close()
print(state)

# load the triplets data
trainhmat = load_pkl(state.datapath + state.dataset + '-train-hs.pkl')
trainlmat = load_pkl(state.datapath + state.dataset + '-train-ls.pkl')
traintmat = load_pkl(state.datapath + state.dataset + '-train-ts.pkl')
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


# In[119]:

idx2cat = {}
cat2idx = {}
for lidx in range(0, state.Nbrel): 
    lo = np.argwhere(true_triples[:,1] == lidx).flatten()

    hl = true_triples[lo, 0]
    _, hlcounts = np.unique(hl, return_counts=True)

    lt = true_triples[lo, 2]
    _, tlcounts = np.unique(lt, return_counts=True)
    idx2cat[lidx] = convert2label(hlcounts.mean(), tlcounts.mean())
    cat2idx.setdefault(convert2label(hlcounts.mean(), tlcounts.mean()), []).append(lidx)

    
vlist = list(idx2cat.values())
[(s, vlist.count(s)) for s in set(vlist)]


# In[121]:

f = open(state.savepath + '/' + state.bestmodel + '.pkl', 'rb')
embeddings = pickle.load(f)
f.close()

rankhfunc = RankHeadFnIdx(simfn, embeddings, subtensorspec=state.Nbsyn)
ranktfunc = RankTailFnIdx(simfn, embeddings, subtensorspec=state.Nbsyn)



# In[ ]:

timeref = time.time()
hrank, trank = PRankingScoreIdx(rankhfunc, ranktfunc, idxth, idxtl, idxtt)
print('the evaluation took %s' % (time.time() - timeref))


# In[139]:

def hitn(lis):
    return np.mean(np.asarray(lis, dtype=np.int32) <=10)*100
        


# In[151]:

hcat2rank = {}
tcat2rank = {}

for hs, ts, linkid in zip(hrank, trank, idxtl):
    hcat2rank.setdefault(idx2cat[linkid], []).append(hs) 
    tcat2rank.setdefault(idx2cat[linkid], []).append(ts) 

print('#Predicting head:')
print('%12s  %12s  %12s  %12s' % ('1-To-1', '1-To-M', 'M-To-1', 'M-To-M'))
print('%12s%%  %12s%%  %12s%%  %12s%%' % (hitn(hcat2rank['1-To-1']), hitn(hcat2rank['1-To-Many']),
                                    hitn(hcat2rank['Many-To-1']), hitn(hcat2rank['Many-To-Many'])))

print('#Predicting tail:')
print('%12s  %12s  %12s  %12s' % ('1-To-1', '1-To-M', 'M-To-1', 'M-To-M'))
print('%12s%%  %12s%%  %12s%%  %12s%%' % (hitn(tcat2rank['1-To-1']), hitn(tcat2rank['1-To-Many']),
                                    hitn(tcat2rank['Many-To-1']), hitn(tcat2rank['Many-To-Many'])))


# In[150]:

hrank = np.random.randint(1,60, size=idxtl.shape)
trank = np.random.randint(1,900, size=idxtl.shape)


# In[ ]:



