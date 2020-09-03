from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from time import time
import numpy as np
from sklearn.metrics import confusion_matrix
from logger import Logger
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--CLAUSES', type=int, default=4000)
parser.add_argument('-T', type=int, default = 7500) #weighted
parser.add_argument('-S', type=float, default=10.0)
parser.add_argument('--DATA', default='dataset.pkl')
args=parser.parse_args()

# Lodaing dataset
dataset_name = args.DATA
with open(dataset_name, 'rb') as f:
    dataset = pickle.load(f)

Y_train = np.asarray(dataset[1])
Y_test = np.asarray(dataset[3])

X_train_transformed = np.load("X_train_transformed.npz")['arr_0']
X_test_transformed = np.load("X_test_transformed.npz")['arr_0']

# Tsetlin machine
clauses = args.CLAUSES
T = args.T
s = args.S

tm = MultiClassTsetlinMachine(clauses, T, s, append_negated=False, weighted_clauses=True)
log = Logger(dataset_name, X_train_transformed, "2layerTsetlin", clauses, T, s)

print(type(X_train_transformed), X_train_transformed.shape)
print(type(Y_train), len(Y_train))


print("\nAccuracy over 250 epochs:\n")
max_accuracy = 0.0
for i in range(250):
    start_training = time()
    tm.fit(X_train_transformed, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    pred = tm.predict(X_test_transformed)
    result = 100*(pred == Y_test).mean()
    stop_testing = time()

    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
    conf_matrix = confusion_matrix(np.asarray(Y_test), pred)
    print('sum predict:', sum(pred), 'sum validation:', sum(Y_test))
    accuracy = 100*(pred == np.asarray(Y_test)).mean()
    print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, accuracy, stop_testing-start_training))
    print('Confusion matrix:')
    print(conf_matrix)
    log.add_epoch(np.asarray(Y_test), pred)
    log.save_log()
print('done')
