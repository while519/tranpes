import numpy as np


class Embeddings():
    def __init__(self, rng, D, N):
        '''

        :param rng: numpy.random module for number generation
        :param D: dimensions of embeddings
        :param N: number of entities or links
        :return:
        '''
        self.D = D
        self.N = N
        wbounds = np.sqrt(6./D)
        W_values = np.random.uniform(low=-wbounds, high=wbounds, size=(D,N))
        W_values = W_values / np.sqrt(np.sum(W_values**2, axis=0))
        self.E = np.asarray(W_values, dtype=np.float32)


def L2sim(left, right):
    return np.sum((left-right)**2, axis=0)


def TranPES3(simfn, hmat, lmat, tmat):
    for hvec, lvec, tvec in zip(hmat.T, lmat.T, tmat.T):
        print(hvec)




def TranPES(state, simfn, embeddings, hp, lp, tp, hn, tn, subtensorE, subtensorR):
    '''

    :param state:
    :param simfn:
    :param embeddings:
    :param hp:
    :param rp:
    :param tp:
    :param hn:
    :param tn:
    :param subtensorE:
    :param subtensorR:
    :return:
    '''
    embedding = embeddings[0]
    lembedding = embeddings[1]

    hpT = hp.T
    hpmat = hpT.dot(embedding.E.T).T        # D x batchsize
    lpT = lp.T
    lpmat = lpT.dot(lembedding.E.T).T
    tpT = tp.T
    tpmat = tpT.dot(embedding.E.T).T

    hnT = hn.T
    hnmat = hnT.dot(embedding.E.T).T
    tnT = tn.T
    tnmat = tnT.dot(embedding.E.T).T


