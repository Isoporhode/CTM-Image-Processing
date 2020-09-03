from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from time import time
import numpy as np
from itertools import chain
from sklearn.metrics import confusion_matrix
from logger import Logger
import argparse
import pickle

def read_x_file_to_list(filename):
    with open(filename, 'r+') as file:
        x_list = [[int(pixel) for pixel in line.split()] for line in file]
    return np.asarray(x_list)

def read_y_file_to_list(filename):
    return list(chain.from_iterable(read_x_file_to_list(filename)))

def tsetlin_with_log(clauses, T, s, x_train, y_train, x_val, y_val,
                     name_dataset):
    tm = MultiClassTsetlinMachine(clauses, T, s)

    log = Logger(name_dataset, x_train, 'mctm', clauses, T, s)
    print(log.metadata)
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dataset.pkl')
parser.add_argument('--clauses', type=int, default=2000)
parser.add_argument('-T', type=int, default=50)
parser.add_argument('-s', type=float, default=10.0)
args = parser.parse_args()

with open(args.dataset, 'rb') as f:
    dataset = pickle.load(f)

x_train = np.asarray(dataset[0], np.uint8)
y_train = dataset[1]
x_val = np.asarray(dataset[2], np.uint8)
y_val = dataset[3]

print(args.clauses, args.T, args.s)
# Tsetlin machine
tsetlin_with_log(args.clauses, args.T, args.s, x_train, y_train, x_val, y_val,
                     args.dataset)

