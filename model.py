import numpy as np
import theano
from collections import OrderedDict
import theano.tensor as T
import theano.sparse as S
import theano.tensor.nlinalg as NL
from joblib import Parallel, delayed
import multiprocessing


def L1sim(left, right): # batchsize x D
    return - T.sum(T.abs_(left - right), axis=1)


def L2sim(left, right):
    return - T.sqrt(T.sum(T.sqr(left - right), axis=1))


def ProjecVec(A, vec):
    """
    This function calculate the projection of vec onto the
    column plane spanned by A
    """
    return  T.dot(T.dot(T.dot(A, NL.matrix_inverse(T.dot(A.T, A) + 1e-8*T.eye(2))), A.T), vec)

def margeCost(pos, neg, marge):
    out = neg - pos + marge
    return T.sum(out * (out > 0)), out > 0

def regEmb(embedding, subtensorspec, alpha=1.):
    out = T.sum(embedding.E[:,subtensorspec]**2, axis=0) - 1
    return alpha * T.sum(out * (out > 1e-7)), out > 1e-7

def regLink(lembedding, subtensorspec, beta=1.):
    out = T.sum(lembedding.E[:, subtensorspec]**2, axis=0)
    return beta * T.sum(out)

def RankTailFnIdx(simfn, embeddings, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'tail' entities given couples of relation and 'head' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding = embeddings[0]
    relation = embeddings[1]

    # Inputs scalar
    idxh = T.iscalar('idxh')
    idxr = T.iscalar('idxr')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        ts = (embedding.E[:, :subtensorspec]).T
    else:
        ts = embedding.E.T
    zerosmat = T.zeros_like(ts)
    hs = (embedding.E[:, idxh]).flatten().dimshuffle('x', 0) + zerosmat
    rel = (relation.E[:, idxr]).flatten().dimshuffle('x', 0) + zerosmat

    simi = tranPES3(simfn, T.concatenate([hs, ts], axis=1).reshape((hs.shape[0], 2, hs.shape[1])).dimshuffle(0, 2, 1), hs, rel, ts)
    """
    Theano function inputs.
    :input idxh: index value of the 'head' member.
    :input idxr: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxh, idxr], simi, on_unused_input='ignore')

def RankHeadFnIdx(simfn, embeddings, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'head' entities given couples of relation and 'tail' entities (as
    index values).

    :param embeddings: an Embeddings instance.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding = embeddings[0]
    relation = embeddings[1]

    # Inputs scalar
    idxt = T.iscalar('idxt')
    idxr = T.iscalar('idxr')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        hs = (embedding.E[:, :subtensorspec]).T
    else:
        hs = embedding.E.T

    zerosmat = T.zeros_like(hs)
    ts = (embedding.E[:, idxt]).flatten().dimshuffle('x', 0) + zerosmat
    rel = (relation.E[:, idxr]).flatten().dimshuffle('x', 0) + zerosmat

    simi = tranPES3(simfn, T.concatenate([hs, ts], axis=1).reshape((hs.shape[0], 2, hs.shape[1])).dimshuffle(0, 2, 1), hs, rel, ts)
    """
    Theano function inputs.
    :input idxr: index value of the relation member.
    :input idxt: index value of the 'tail' member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxt], simi, on_unused_input='ignore')

def errht(sh, st, h, r, t):

    result1 = np.argsort(np.argsort(sh(r, t).flatten())[::-1]).flatten()[h] + 1
    result2 = np.argsort(np.argsort((st(h, r)).flatten())[::-1]).flatten()[t] + 1
    
    return result1, result2

def PRankingScoreIdx( sh, st, idxh, idxr, idxt):
    """
        This function computes the rank list of the hs and ts, over a list of
        hs, rs and ts indexes.
        
        :param sh: Theano function created with RankHeadFnIdx().
        :param st: Theano function created with RankTailFnIdx().
        :param idxh: list of 'head' indices.
        :param idxt: list of 'tail' indices.
        :param idxr: list of relation indices.
        """
    num_cores = multiprocessing.cpu_count()
    
    results = Parallel(n_jobs=num_cores)(delayed(errht)(sh, st, h, r, t) for h, r, t in zip(idxh, idxr, idxt))
    errh, errt = zip(*results)
    
    return errh, errt

def RankingScoreIdx(sh, st, idxh, idxr, idxt):
    """
    This function computes the rank list of the hs and ts, over a list of
    hs, rs and ts indexes.

    :param sh: Theano function created with RankHeadFnIdx().
    :param st: Theano function created with RankTailFnIdx().
    :param idxh: list of 'head' indices.
    :param idxt: list of 'tail' indices.
    :param idxr: list of relation indices.
    """
    errh = []
    errt = []
    for h, r, t in zip(idxh, idxr, idxt):
        errh += [np.argsort(np.argsort(
            sh(r, t).flatten())[::-1]).flatten()[h] + 1]
        errt += [np.argsort(np.argsort((
            st(h, r)).flatten())[::-1]).flatten()[t] + 1]
    return errh, errt


class Embeddings():
    def __init__(self, rng, D, N, tag=''):
        """
        Constructor.

        : param rng: numpy.random module for number generation.
        : param N: number of entities, relations or both.
        : param D: dimensions of the embeddings.
        : param tag: name of the embeddings for parameter declaration.
        """

        self.N = N
        self.D = D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(D,N))
        W_values = W_values / np.sqrt(np.sum(W_values**2, axis=0))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E'+  tag)
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E**2, axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)

#def tranPES1(simfn, hmat, rmat, tmat):
#    results, updates = theano.scan(fn=lambda h, r, t :  simfn(h + ProjecVec(T.concatenate([h, t], axis=0).reshape((2,h.shape[0])).T, r), t),
#                       outputs_info=None,
#                       sequences=[hmat, rmat, tmat])
#    return results
#
#def tranPES2(simfn, A, hmat, rmat, tmat):
#    results, updates = theano.scan(fn=lambda As, h, r, t :  simfn(h + ProjecVec(As, r), t),
#                       outputs_info=None,
#                       sequences=[A, hmat, rmat, tmat])
#    return results

def tranPES3(simfn, A, hmat, rmat, tmat):
    P, updates = theano.scan(fn=lambda As, r :  ProjecVec(As, r),
                       outputs_info=None,
                       sequences=[A, rmat])
    results = simfn(hmat + P, tmat)
    return results

def create_TrainFunc_tranPES(simfn, embeddings,  marge=0.5, alpha=1., beta=1.):

    # parse the embedding data
    embedding = embeddings[0] # D x N matrix
    lembedding = embeddings[1]

    # declare the symbolic variables for training triples
    hp = T.ivector('head positive') # N x batchsize matrix
    rp = T.ivector('relation')
    tp = T.ivector('tail positive')

    hn = T.ivector('head negative')
    tn = T.ivector('tail negative')

    lemb = T.scalar('embedding learning rate')
    lremb = T.scalar('relation learning rate')

    subtensorE = T.ivector('batch entities set')
    subtensorR = T.ivector('batch link set')

    # Generate the training positive and negative triples
    hpmat = embedding.E[:,hp].T #  batchsize x D dense matrix
    rpmat = embedding.E[:,rp].T
    tpmat = embedding.E[:,tp].T

    hnmat = embedding.E[:,hn].T
    tnmat = embedding.E[:,tn].T

    # calculate the score
    pos = tranPES3(simfn, T.concatenate([hpmat, tpmat], axis=1).reshape((hpmat.shape[0], 2, hpmat.shape[1])).dimshuffle(0, 2, 1), hpmat, rpmat, tpmat)


    negh = tranPES3(simfn, T.concatenate([hnmat, tpmat], axis=1).reshape((hnmat.shape[0], 2, hnmat.shape[1])).dimshuffle(0, 2, 1), hnmat, rpmat, tpmat)
    negt = tranPES3(simfn, T.concatenate([hpmat, tnmat], axis=1).reshape((hpmat.shape[0], 2, hpmat.shape[1])).dimshuffle(0, 2, 1), hpmat, rpmat, tnmat)

    costh, outh = margeCost(pos, negh, marge)
    costt, outt = margeCost(pos, negt, marge)

    embreg = regEmb(embedding, subtensorE, alpha)
    lembreg = regLink(lembedding, subtensorR, beta)
    

    cost = costh + costt + embreg[0] + lembreg
    out = T.concatenate([outh, outt])
    outc = embreg[1]

    # list of inputs to the function
    list_in = [lemb, lremb, hp, rp, tp, hn, tn, subtensorE, subtensorR]

    # updating the embeddings using gradient descend
    emb_grad = T.grad(cost, embedding.E)
    New_embedding = embedding.E - lemb*emb_grad

    remb_grad = T.grad(cost, lembedding.E)
    New_rembedding = lembedding.E - lremb * remb_grad

    updates = OrderedDict({embedding.E: New_embedding, lembedding.E: New_rembedding})

    return theano.function(list_in, [cost, T.mean(out), T.mean(outc), embreg[0], lembreg],
                          updates=updates, on_unused_input='ignore')
