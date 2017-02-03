__author__ = 'Rakesh'

import launcher
import sys

class Defaults:

	# NETWORK HYPER-PARAMETERS

	MAX_EPOCH = 1000
	LEARNING_RATE = 0.0025
	MOMENTUM = 0.95
	NUM_HIDDEN = 2
	SIZES = "256, 64"
	ACTIVATION = 'sigmoid'
	LOSS = 'ce'
	OPTIMIZER = 'gd'
	BATCH_SIZE = 32
	ANNEAL_LR = True
	SAVE_DIR = '../models/'
	EXPT_DIR = '../logs/'
	DATA_PATH = '../data/mnist.pkl.gz'

if __name__ == "__main__":
	launcher.start(sys.argv[1:], Defaults, __doc__)