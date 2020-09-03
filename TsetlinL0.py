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

# Lodaing dataset
dataset_name = args.DATA
with open(dataset_name, 'rb') as f:
    dataset = pickle.load(f)

X_train = np.asarray(dataset[0], np.uint8)
Y_train = dataset[1]
X_test = np.asarray(dataset[2], np.uint8)
Y_test = dataset[3]

# Tsetlin machine
clauses = args.CLAUSES
mask = args.MASK
T = args.T
s = args.S

tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (mask, mask))

print("\nAccuracy over 2 epochs:\n")
max_accuracy = 0.0
for i in range(2):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	if result > max_accuracy:
		max_accuracy = result
		max_ta_state = tm.get_state()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

tm.set_state(max_ta_state)

print("\nTransforming datasets")
start_transformation = time()
X_train_transformed = tm.transform(X_train)
X_test_transformed = tm.transform(X_test)
stop_transformation = time()
print("Transformation time: %.fs" % (stop_transformation - start_transformation))

print("Saving transformed datasets")
np.savez_compressed("X_train_transformed.npz", X_train_transformed)
np.savez_compressed("X_test_transformed.npz", X_test_transformed)
