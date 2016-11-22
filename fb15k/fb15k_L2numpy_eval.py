
# coding: utf-8

# In[15]:
import multiprocessing
from joblib import Parallel, delayed
from numpy.linalg import solve, inv
import numpy as np
import time


# In[25]:

bestmodel = 'model490'
n = 10
neval = 100
np.random.seed(123)

embedding_np = np.loadtxt(bestmodel + '_embedding.txt')
relation_np = np.loadtxt(bestmodel + '_relation.txt')
true_triples = np.loadtxt(bestmodel + '_true_triplets.txt', np.int32)
idxth = np.loadtxt(bestmodel + '_idxth.txt', np.int32)
idxtl = np.loadtxt(bestmodel + '_idxtl.txt', np.int32)
idxtt = np.loadtxt(bestmodel + '_idxtt.txt', np.int32)


# In[7]:

def FilteredPNerrht(embedding_np, relation_np, h, l, t, true_triples, subtensorspec=None):
    ih = np.argwhere(true_triples[:,0] == h).flatten()
    ir = np.argwhere(true_triples[:,1] == l).flatten()
    it = np.argwhere(true_triples[:,2] == t).flatten()
    
    inter_h = [i for i in ir if i in it]
    rmvhidx = [true_triples[i,0] for i in inter_h if true_triples[i,0]!=h]

    inter_t = [i for i in ih if i in ir]
    rmvtidx = [true_triples[i, 2] for i in inter_t if true_triples[i, 2] !=t]
    
    hscore = np.zeros(subtensorspec)
    tscore = np.zeros(subtensorspec)
    hvec = embedding_np[:,h]
    tvec = embedding_np[:,t]
    for i in range(subtensorspec):
        x = embedding_np[:,i] + NProjecVec(embedding_np[:,[i, t]], relation_np[:,l]) - tvec
        hscore[i] = -np.sqrt(x.dot(x))
        y = hvec + NProjecVec(embedding_np[:,[h, i]],relation_np[:,l]) - embedding_np[:,i]
        tscore[i] = -np.sqrt(y.dot(y))
    hscore[rmvhidx] = -np.inf
    tscore[rmvtidx] = -np.inf
    
    return np.argsort(np.argsort(hscore.flatten())[::-1]).flatten()[h] + 1, np.argsort(np.argsort(tscore.flatten())[::-1]).flatten()[t] + 1


def NProjecVec(A, vec):
    """
    This function calculate the projection of vec onto the
    column plane spanned by A
    """
    return  np.dot(np.dot(np.dot(A, inv(np.dot(A.T, A) + 1e-8*np.eye(2))), A.T), vec)

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


def output(model, res, n):
    dres = {}
    dres.update({'microlmean': np.mean(res[0])})
    dres.update({'microlmedian': np.median(res[0])})
    dres.update({'microlhits@n': np.mean(np.asarray(res[0]) <= n) * 100})
    dres.update({'micrormean': np.mean(res[1])})
    dres.update({'micrormedian': np.median(res[1])})
    dres.update({'microrhits@n': np.mean(np.asarray(res[1]) <= n) * 100})
    resg = res[0] + res[1]
    dres.update({'microgmean': np.mean(resg)})
    dres.update({'microgmedian': np.median(resg)})
    dres.update({'microghits@n': np.mean(np.asarray(resg) <= n) * 100})

    print("### " + model + " MICRO:")
    print("\t-- head   >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
        n, round(dres['microlhits@n'], 3)))
    print("\t-- tail  >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
        n, round(dres['microrhits@n'], 3)))
    print("\t-- global >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
        n, round(dres['microghits@n'], 3)))
    return


def hitn(lis):
    return np.mean(np.asarray(lis, dtype=np.int32) <=10)*100


# In[27]:

# main snippet_4 --------- parallel for loop using joblib
num_cores = multiprocessing.cpu_count()
print(num_cores)
timeref = time.time()
if neval is not None:
    results = Parallel(n_jobs=num_cores, max_nbytes=1e7)(delayed(FilteredPNerrht)(embedding_np, relation_np, h, r, t, true_triples, 14951) for h, r, t in zip(idxth, idxtl, idxtt))
else:
    results = Parallel(n_jobs=num_cores, max_nbytes=1e7)(delayed(FilteredPNerrht)(embedding_np, relation_np, h, r, t, true_triples, 14951) for h, r, t in zip(idxth[:neval], idxtl[:neval], idxtt[:neval]))
fres = list(zip(*results))
print('the evaluation took %s' % (time.time() - timeref))
output('test_' + bestmodel, fres, n)

idx2cat = {}
cat2idx = {}
for lidx in range(0, 1345):
    lo = np.argwhere(true_triples[:,1] == lidx).flatten()

    hl = true_triples[lo, 0]
    _, hlcounts = np.unique(hl, return_counts=True)

    lt = true_triples[lo, 2]
    _, tlcounts = np.unique(lt, return_counts=True)
    idx2cat[lidx] = convert2label(hlcounts.mean(), tlcounts.mean())
    cat2idx.setdefault(convert2label(hlcounts.mean(), tlcounts.mean()), []).append(lidx)

vlist = list(idx2cat.values())
print([(s, vlist.count(s)) for s in set(vlist)])

hrank = fres[0]
trank = fres[1]

print()

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

