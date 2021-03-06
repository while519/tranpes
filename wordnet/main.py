import wordnet_train
import wordnet_evaluation

# training the model
wordnet_train.launch(test_all=10, totepochs=1000, neval=1000, marge=1., nbatches=100,
            alpha=1., beta=0.01, lremb=0.002, lrparam=0.002, ndim=20, seed=132)

# evaluation for the model
wordnet_evaluation.evaluation()
