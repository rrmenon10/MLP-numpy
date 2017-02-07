__author__ = 'Rakesh'

import launcher
import sys

class Defaults:

	# NETWORK HYPER-PARAMETERS

	MAX_EPOCH = 1000
	LEARNING_RATE = 0.1
	MOMENTUM = 0.95
	NUM_HIDDEN = 2
	SIZES = "256, 64"
	ACTIVATION = 'sigmoid'
	LOSS = 'ce'
	OPTIMIZER = 'gd'
	BATCH_SIZE = 20
	ANNEAL_LR = False
	SAVE_DIR = '../models/'
	EXPT_DIR = '../logs/'
	DATA_PATH = '../data/mnist.pkl.gz'

if __name__ == "__main__":
	launcher.start(sys.argv[1:], Defaults, __doc__)