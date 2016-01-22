import numpy as np
from numpy.linalg import inv
import copy
import cProfile


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
    return -np.sum((left-right)**2, axis=0)

def ProjecMat(A):
    """
    This function calculate the projection of vec onto the
    column plane spanned by A
    """
    return  np.dot(np.dot(A, inv(np.dot(A.T, A) + 1e-8*np.eye(2))), A.T)

def TranPES3(simfn, hvec, lvec, tvec):
    A = np.vstack((hvec, tvec)).T
    P = ProjecMat(A)
    return simfn(hvec + np.dot(P, lvec), tvec), P

def margeCost(pos, neg, marge):
    out = neg - pos + marge
    return out * (out > 0), out > 0

def regEmb(embedding, subtensorspec, alpha=1.):
    out = np.sum(embedding.E[:,subtensorspec]**2, axis=0) - 1
    return alpha * np.sum(out * (out > 1e-5)), out > 1e-5

def regLink(lembedding, subtensorspec, beta=1.):
    out = np.sum(lembedding.E[:, subtensorspec]**2, axis=0)
    return beta * np.sum(out)

def p_Gradh(hvec, lvec, tvec):
    A = np.vstack((hvec, tvec)).T
    PL = np.zeros((hvec.shape[0], hvec.shape[0]))
    for ii in range(hvec.shape[0]):
        A_gradhi = np.zeros_like(A)
        A_gradhi[ii,0] = 1
        ATA_gradhi = np.asarray([[2*hvec[ii], tvec[ii]], [tvec[ii], 0]])
        P_gradhi = np.dot(A_gradhi, np.dot(inv(np.dot(A.T, A) + 1e-8*np.eye(2)), A.T)) + \
                   np.dot(A, np.dot(inv(np.dot(A.T, A) + 1e-8*np.eye(2)), A_gradhi.T)) - \
                   np.dot(np.dot(np.dot(A, np.dot(inv(np.dot(A.T, A) + 1e-8*np.eye(2)), ATA_gradhi)), inv(np.dot(A.T, A) + 1e-8*np.eye(2))), A.T)
        p_gradhi = np.dot(P_gradhi, lvec)
        PL[ii,:] = p_gradhi
    return PL

def p_Gradt(hvec, lvec, tvec):
    A = np.vstack((hvec, tvec)).T
    PL = np.zeros((hvec.shape[0], hvec.shape[0]))
    for ii in range(hvec.shape[0]):
        A_gradti = np.zeros_like(A)
        A_gradti[ii,1] = 1
        ATA_gradti = np.asarray([[0, hvec[ii]], [hvec[ii], 2*tvec[ii]]])
        P_gradti = np.dot(A_gradti, np.dot(inv(np.dot(A.T, A) + 1e-8*np.eye(2)), A.T)) + \
                   np.dot(A, np.dot(inv(np.dot(A.T, A) + 1e-8*np.eye(2)), A_gradti.T)) - \
                   np.dot(np.dot(A,np.dot(inv(np.dot(A.T,A) + 1e-8*np.eye(2)), np.dot(ATA_gradti, inv(np.dot(A.T,A) + 1e-8*np.eye(2))))), A.T)
        PL[ii, :] = np.dot(P_gradti, lvec)
    return PL

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
    pr = cProfile.Profile()
    pr.enable()
    embedding = embeddings[0]
    lembedding = embeddings[1]

    New_embedding = np.zeros_like(embedding.E)
    New_lembedding = np.zeros_like(lembedding.E)
    cost = 0.
    out = []


    # updating the entities embeddings according to each triplet cost
    for ii in range(hp.shape[0]):
        hpvec = embedding.E[:, hp[ii]]        # D x batchsize
        lpvec = lembedding.E[:, lp[ii]]
        tpvec = embedding.E[:, tp[ii]]

        hnvec = embedding.E[:, hn[ii]]
        tnvec = embedding.E[:, tn[ii]]

        pos, posP = TranPES3(simfn, hpvec, lpvec, tpvec)
        negh, neghP = TranPES3(simfn, hnvec, lpvec, tpvec)
        negt, negtP = TranPES3(simfn, hpvec, lpvec, tnvec)

        costh, outh = margeCost(pos, negh, state.marge)
        costt, outt = margeCost(pos, negt, state.marge)
        cost += costh + costt
        out += [outh, outt]

        if outh:
            New_lembedding[:, lp[ii]] -= np.dot(2*posP.T, hpvec + np.dot(posP, lpvec) - tpvec) \
                                         - np.dot(2*neghP.T, (hnvec + np.dot(neghP, lpvec)) - tpvec)
            New_embedding[:, hp[ii]] -= np.dot(np.eye(state.ndim) + p_Gradh(hpvec, lpvec, tpvec), 2*(hpvec + np.dot(posP, lpvec) - tpvec))
            New_embedding[:, hn[ii]] -= -np.dot(np.eye(state.ndim) + p_Gradh(hnvec, lpvec, tpvec), 2*(hpvec + np.dot(neghP, lpvec) - tpvec))
            New_embedding[:, tp[ii]] -= np.dot(p_Gradt(hpvec, lpvec, tpvec) - np.eye(state.ndim), 2*(hpvec + np.dot(posP, lpvec) - tpvec)) - \
                                        np.dot(p_Gradt(hnvec, lpvec, tpvec) - np.eye(state.ndim), 2*(hnvec + np.dot(neghP, lpvec) - tpvec))
        if outt:
            New_lembedding[:, lp[ii]] -= np.dot(2*posP.T, (hpvec + np.dot(posP, lpvec)) - tpvec) \
                                         - np.dot(2*negtP.T, (hpvec + np.dot(negtP, lpvec)) - tnvec)
            New_embedding[:, hp[ii]] -= np.dot(np.eye(state.ndim) + p_Gradh(hpvec, lpvec, tpvec), 2*(hpvec + np.dot(posP, lpvec) - tpvec)) - \
                                        np.dot(np.eye(state.ndim) + p_Gradh(hpvec, lpvec, tnvec), 2*(hpvec + np.dot(negtP, lpvec) - tnvec))
            New_embedding[:, tp[ii]] -= np.dot(p_Gradt(hpvec, lpvec, tpvec) - np.eye(state.ndim), 2*(hpvec + np.dot(posP, lpvec) - tpvec))
            New_embedding[:, tn[ii]] -= -np.dot(p_Gradt(hpvec, lpvec, tnvec) - np.eye(state.ndim), 2*(hpvec + np.dot(negtP, lpvec) - tnvec))

    embreg = regEmb(embedding, subtensorE, state.alpha)
    emblink = regLink(lembedding, subtensorR, state.beta)

    cost += embreg[0] + emblink
    New_embedding[:, subtensorE[embreg[1]]] -= 2*state.alpha*embedding.E[:,subtensorE[embreg[1]]]
    New_lembedding[:, subtensorR] -= 2*state.beta*lembedding.E[:,subtensorR]
    embedding.E = embedding.E + New_embedding*state.lremb
    lembedding.E = lembedding.E + New_lembedding*state.lrparam

    pr.disable()
    pr.print_stats(sort="call")
    return cost, np.mean(out)
















