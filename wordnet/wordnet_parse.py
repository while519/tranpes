import numpy as np
import os
import pickle
import scipy.sparse as sp

'''
read the ASCII data and stores it into matrix form.
'''

if 'data' not in os.listdir():
    os.makedirs('data')

def parseline(line):
    hs, ls, ts = line.split('\t')
    return hs, ls, ts


# Creation of the entity/indices dictionary
for datatype in ['train']:
    f = open('wordnet-mlj12-%s.txt' % datatype, mode='rt')
    lines = f.readlines()
    f.close()
    hlist = []
    llist = []
    tlist = []
    for line in lines:
        hs, ls, ts = parseline(line[:-1])
        hlist += [hs]
        llist += [ls]
        tlist += [ts]

headset = np.sort(list(set(hlist) - set(tlist)))
shareset = np.sort(list(set(hlist) & set(tlist)))
tailset = np.sort(list(set(tlist) - set(hlist)))
labelset = np.sort(list(set(llist)))

entity2idx = {}
idx2entity = {}

idx = 0

for i in tailset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbtail = idx

for i in shareset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbshared = idx - nbtail

for i in headset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbhead = idx - nbshared - nbtail

print('Number of only_head/shared/only_tail entities: %s/%s/%s' % (nbhead, nbshared, nbtail))

# add the relation at the end of the dictionary
for i in labelset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nblabel = idx - nbhead - nbshared - nbtail

print('Number of relations: %s' % nblabel)

f = open('data/entity2idx.pkl', 'wb')
g = open('data/idx2entity.pkl', 'wb')

pickle.dump(entity2idx, f, protocol=-1)
pickle.dump(idx2entity, g, protocol=-1)


# Creation of the dataset files
for datatype in ['train', 'valid', 'test']:
    f = open('wordnet-mlj12-%s.txt' % datatype, mode='rt')
    lines = f.readlines()
    f.close()

    headmat = sp.lil_matrix((nblabel + nbtail + nbshared +nbhead,len(lines)))
    labelmat = sp.lil_matrix((nblabel + nbtail + nbshared +nbhead,len(lines)))
    tailmat = sp.lil_matrix((nblabel + nbtail + nbshared +nbhead,len(lines)))
    unseen_entities = []
    remove_triplets = []

    ct = 0
    for line in lines:
        hs, ls, ts = parseline(line[:-1])
        if hs in entity2idx and ls in entity2idx and ts in entity2idx:
            headmat[entity2idx[hs], ct] = 1
            labelmat[entity2idx[ls], ct] = 1
            tailmat[entity2idx[ts], ct] = 1
            ct += 1
        else:
            if ls not in entity2idx:
                unseen_entities += [ls]
            if hs not in entity2idx:
                unseen_entities += [hs]
            if ts not in entity2idx:
                unseen_entities += [ts]
            remove_triplets += [i[:-1]]

    f = open('data/WN-%s-hs.pkl' % datatype, 'wb')
    g = open('data/WN-%s-ls.pkl' % datatype, 'wb')
    h = open('data/WN-%s-ts.pkl' % datatype, 'wb')
    pickle.dump(headmat.tocsr(), f, -1)
    pickle.dump(labelmat.tocsr(), g, -1)
    pickle.dump(tailmat.tocsr(), h, -1)

    f.close()
    g.close()
    h.close()

unseen_ent_set = list(set(unseen_entities))
print(len(unseen_ent_set))

remove_set = list(set(remove_triplets))
print(len(remove_set))

for i in remove_set:
    print(i)



