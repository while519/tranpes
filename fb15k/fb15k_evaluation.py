#! /usr/bin/python
import os
import sys
import pickle
import scipy.sparse as sp
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import *
import time


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


# ----------------------------------------------------------------------------------------------------------------------
def evaluation():
    n = 10
    f = open('state.pkl', 'rb')
    state = pickle.load(f)
    f.close()

    state.neval = 10
    state.bestvalid = -1
    print(state)

    # select the best valid model
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


    for model in ['model400', 'model410', 'model420', 'model430', 'model440', 'model450', 'model460', 'model470',
                   'model480',
                   'model490', 'model500']:
         f = open(state.savepath + '/' + model + '.pkl', 'rb')
         embeddings = pickle.load(f)
         f.close()

         rankhfunc = RankHeadFnIdx(simfn, embeddings, subtensorspec=state.Nbsyn)
         ranktfunc = RankTailFnIdx(simfn, embeddings, subtensorspec=state.Nbsyn)

         timeref = time.time()
         resvalid = RankingScoreIdx(rankhfunc, ranktfunc, validhidx, validlidx, validtidx)
         fresvalid = FilteredRankingScoreIdx(rankhfunc, ranktfunc, validhidx, validlidx, validtidx, true_triples)
         print('the evaluation took %s' % (time.time() - timeref))

         state.valid = np.mean(resvalid[0] + resvalid[1])
         if state.bestvalid == -1 or state.valid < state.bestvalid:
             output('valid_' + model, resvalid, n)
             output('Filteredvalid' + model, fresvalid, n)
             restest = PRankingScoreIdx(rankhfunc, ranktfunc, testhidx, testlidx, testtidx)
             restrain = PRankingScoreIdx(rankhfunc, ranktfunc, trainhidx, trainlidx, traintidx)
             state.bestvalid = state.valid
             state.besttrain = np.mean(restrain[0] + restrain[1])
             state.besttest = np.mean(restest[0] + restest[1])
             state.bestmodel = model
             print('New best valid >> train:   Mean Rank>>>>%s  Hit@%s>>>>%s%%' % (state.besttrain, n,
                                                                                   np.mean(np.asarray(
                                                                                           restrain[0] + restrain[
                                                                                               1]) <= n) * 100))
             print('New best valid >> test:   Mean Rank>>>>%s  Hit@%s>>>>%s%%' % (state.besttest, n,
                                                                                  np.mean(np.asarray(
                                                                                          restest[0] + restest[
                                                                                              1]) <= n) * 100))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    f = open('best_state.pkl', 'wb')
    pickle.dump(state, f, -1)
    f.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #state.bestmodel = 'model400'
    # Fully evaluation
    f = open(state.savepath + '/' + state.bestmodel + '.pkl', 'rb')
    embeddings = pickle.load(f)
    f.close()

    rankhfunc = RankHeadFnIdx(simfn, embeddings, subtensorspec=state.Nbsyn)
    ranktfunc = RankTailFnIdx(simfn, embeddings, subtensorspec=state.Nbsyn)

    timeref = time.time()
    #res = PRankingScoreIdx(rankhfunc, ranktfunc, validhidx, validlidx, validtidx)
    res = FilteredPRankingScoreIdx(rankhfunc, ranktfunc, idxth, idxtl, idxtt, true_triples)
    #res = PRankingScoreIdx(rankhfunc, ranktfunc, idxth, idxtl, idxtt)
    print('the evaluation took %s' % (time.time() - timeref))
    output('test_' + state.bestmodel, res, n)

    f = open('Filtered_eval.pkl', 'wb')
    pickle.dump(fres, f, -1)
    f.close()


def RankingEvalf():
    n = 10
    f = open('best_state.pkl', 'rb')
    state = pickle.load(f)
    f.close()
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


    g = open(state.savepath + '/' + state.bestmodel + '.pkl', 'rb')
    embeddings = pickle.load(g)
    g.close()

    rankhfunc = RankHeadFnIdx(simfn, embeddings, subtensorspec=state.Nbsyn)
    ranktfunc = RankTailFnIdx(simfn, embeddings, subtensorspec=state.Nbsyn)

    timeref = time.time()
    fres = FilteredPRankingScoreIdx(rankhfunc, ranktfunc, idxth, idxtl, idxtt, true_triples)
    print('the evaluation took %s' % (time.time() - timeref))
    output('Filteredtest_' + state.bestmodel, fres, n)

    f = open('Filtered_eval.pkl', 'wb')
    pickle.dump(fres, f, -1)
    f.close()

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
    print([(s, vlist.count(s)) for s in set(vlist)])

    hrank = fres[0]
    trank = fres[1]

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

        # listrel = set(idxr)
        # dictrelres = {}
        # dictrellmean = {}
        # dictrelrmean = {}
        # dictrelgmean = {}
        # dictrellmedian = {}
        # dictrelrmedian = {}
        # dictrelgmedian = {}
        # dictrellrn = {}
        # dictrelrrn = {}
        # dictrelgrn = {}
        #
        # for i in listrel:
        #     dictrelres.update({i: [[], []]})
        #
        # for i, j in enumerate(resvalid[0]):
        #     dictrelres[idxr[i]][0] += [j]
        #
        # for i, j in enumerate(resvalid[1]):
        #     dictrelres[idxr[i]][1] += [j]
        #
        # for i in listrel:
        #     dictrellmean[i] = np.mean(dictrelres[i][0])
        #     dictrelrmean[i] = np.mean(dictrelres[i][1])
        #     dictrelgmean[i] = np.mean(dictrelres[i][0] + dictrelres[i][1])
        #     dictrellmedian[i] = np.median(dictrelres[i][0])
        #     dictrelrmedian[i] = np.median(dictrelres[i][1])
        #     dictrelgmedian[i] = np.median(dictrelres[i][0] + dictrelres[i][1])
        #     dictrellrn[i] = np.mean(np.asarray(dictrelres[i][0]) <= n) * 100
        #     dictrelrrn[i] = np.mean(np.asarray(dictrelres[i][1]) <= n) * 100
        #     dictrelgrn[i] = np.mean(np.asarray(dictrelres[i][0] +
        #                                        dictrelres[i][1]) <= n) * 100
        #
        # dres.update({'dictrelres': dictrelres})
        # dres.update({'dictrellmean': dictrellmean})
        # dres.update({'dictrelrmean': dictrelrmean})
        # dres.update({'dictrelgmean': dictrelgmean})
        # dres.update({'dictrellmedian': dictrellmedian})
        # dres.update({'dictrelrmedian': dictrelrmedian})
        # dres.update({'dictrelgmedian': dictrelgmedian})
        # dres.update({'dictrellrn': dictrellrn})
        # dres.update({'dictrelrrn': dictrelrrn})
        # dres.update({'dictrelgrn': dictrelgrn})
        #
        # dres.update({'macrolmean': np.mean(list(dictrellmean.values()))})
        # dres.update({'macrolmedian': np.mean(list(dictrellmedian.values()))})
        # dres.update({'macrolhits@n': np.mean(list(dictrellrn.values()))})
        # dres.update({'macrormean': np.mean(list(dictrelrmean.values()))})
        # dres.update({'macrormedian': np.mean(list(dictrelrmedian.values()))})
        # dres.update({'macrorhits@n': np.mean(list(dictrelrrn.values()))})
        # dres.update({'macrogmean': np.mean(list(dictrelgmean.values()))})
        # dres.update({'macrogmedian': np.mean(list(dictrelgmedian.values()))})
        # dres.update({'macroghits@n': np.mean(list(dictrelgrn.values()))})
        #
        # print("### MACRO:")
        # print("\t-- head   >> mean: %s, median: %s, hits@%s: %s%%" % (
        #     round(dres['macrolmean'], 5), round(dres['macrolmedian'], 5),
        #     n, round(dres['macrolhits@n'], 3)))
        # print("\t-- tail  >> mean: %s, median: %s, hits@%s: %s%%" % (
        #     round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
        #     n, round(dres['macrorhits@n'], 3)))
        # print("\t-- global >> mean: %s, median: %s, hits@%s: %s%%" % (
        #     round(dres['macrogmean'], 5), round(dres['macrogmedian'], 5),
        #     n, round(dres['macroghits@n'], 3)))
        #
        # idx2entity = pickle.load(open(state.datapath + '/idx2entity', mode='rb'))
        # offset = 0
        # if type(embeddings) is list:
        #     r = r[-embeddings[1].N:, :]
        #     offset = h.shape[0] - embeddings[1].N
        # for i in np.sort(list(listrel)):
        #     print("### RELATION %s:" % idx2entity[offset + i])
        #     print("\t-- head   >> mean: %s, median: %s, hits@%s: %s%%, N: %s" % (
        #         round(dictrellmean[i], 5), round(dictrellmedian[i], 5),
        #         n, round(dictrellrn[i], 3), len(dictrelres[i][0])))
        #     print("\t-- tail  >> mean: %s, median: %s, hits@%s: %s%%, N: %s" % (
        #         round(dictrelrmean[i], 5), round(dictrelrmedian[i], 5),
        #         n, round(dictrelrrn[i], 3), len(dictrelres[i][1])))
        #     print("\t-- global >> mean: %s, median: %s, hits@%s: %s%%, N: %s" % (
        #         round(dictrelgmean[i], 5), round(dictrelgmedian[i], 5),
        #         n, round(dictrelgrn[i], 3),
        #         len(dictrelres[i][0] + dictrelres[i][1])))

if __name__ == '__main__':
    evaluation()
    RankingEvalf()
