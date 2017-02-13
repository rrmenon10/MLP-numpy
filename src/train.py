__author__ = 'Rakesh'

import launcher
import sys

class Defaults:

	# NETWORK HYPER-PARAMETERS

	MAX_EPOCH = 100
	LEARNING_RATE = 0.0025
	MOMENTUM = 0.6
	NUM_HIDDEN = 2
	SIZES = "50, 50"
	ACTIVATION = 'sigmoid'
	LOSS = 'ce'
	OPTIMIZER = 'adam'
	BATCH_SIZE = 20
	ANNEAL_LR = False
	SAVE_DIR = '../models/'
	EXPT_DIR = '../logs/'
	DATA_PATH = '../data/mnist.pkl.gz'

	########################
	# LOAD MODEL PARAMETERS#
	########################
	LOAD_MODEL = False
	LOAD_EPOCH = MAX_EPOCH

if __name__ == "__main__":
	launcher.start(sys.argv[1:], Defaults, __doc__)