import fb15k_train
import fb15k_evaluation

# Training the model
fb15k_train.launch(test_all=10, totepochs=500, neval=1, marge=0.3, nbatches=100, alpha=1., beta=0.01, lremb=0.002, lrparam=0.002)

# Evaluate the model
fb15k_evaluation.evaluation()