from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
from time import time
import numpy as np
from itertools import chain
import random
from sklearn.metrics import confusion_matrix
from logger import Logger
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--MASK', type=int, default=8)
parser.add_argument('--CLAUSES', type=int, default=4000)
parser.add_argument('-T', type=int, default = 75)
parser.add_argument('-S', type=float, default=10.0)
parser.add_argument('--DATA', default='dataset.pkl')
args=parser.parse_args()

dataset_name = args.DATA
with open(dataset_name, 'rb') as f:
    dataset = pickle.load(f)

# Lodaing dataset
x_train = np.asarray(dataset[0], np.uint8)
y_train = dataset[1]
x_val = np.asarray(dataset[2], np.uint8)
y_val = dataset[3]

# Tsetlin machine
clauses = args.CLAUSES
mask = args.MASK
T = args.T
s = args.S

tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (mask, mask))

log = Logger(dataset_name, x_train, "TM2D", clauses, T, s, (mask, mask))
print(x_train[0][0])
print(x_train.shape)
print('predicting 400 epochs')
for i in range(400):
    start = time()
    tm.fit(x_train, y_train, epochs=1, incremental=True)
    stop = time()
    pred = tm.predict(x_val)
    conf_matrix = confusion_matrix(np.asarray(y_val), pred)
    print('sum predict:', sum(pred), 'sum validation:', sum(y_val))
    accuracy = 100*(pred == np.asarray(y_val)).mean()
    print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, accuracy, stop-start))
    print('Confusion matrix:')
    print(conf_matrix)
    log.add_epoch(np.asarray(y_val), pred)
    log.save_log()

print('done')

