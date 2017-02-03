__author__ = 'Rakesh'

import os
import argparse
import cPickle
import numpy as np
import main

def process_args(args, defaults, description):

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--max_epoch', dest="max_epoch", type=int, default=defaults.MAX_EPOCH,
                        help=('Number of Epochs to train network'+
                              '(default: %(default)s)'))
    
    parser.add_argument('--lr', dest="lr_start", type=float, default=defaults.LEARNING_RATE,
                        help=('Learning rate for network'+
                              '(default: %(default)s)'))

    parser.add_argument('--momentum', dest="momentum", type=float, default=defaults.MOMENTUM,
                        help=('Momentum term for Nesterov momentum.'+
                              '(default: %(default)s)'))

    parser.add_argument('--num_hidden', dest="num_hidden", type=int, default=defaults.NUM_HIDDEN,
                        help=('Number of hidden layers'+
                              '(default: %(default)s)'))

    parser.add_argument('--sizes', dest="hid_sizes", default=defaults.SIZES,
                        help=('Number of neurons in each layer'+
                              '(default: %(default)s)'))

    parser.add_argument('--activation', dest="act",
                        default=defaults.ACTIVATION,
                        help=('Activation function in each layer (tanh/sigmoid)'+
                              '(default: %(default)s)'))

    parser.add_argument('--loss', dest="loss",
                        default=defaults.LOSS,
                        help=('Loss function (sq,ce)'+
                              '(default: %(default)s)'))

    parser.add_argument('--opt', dest="optimizer",
                        default=defaults.OPTIMIZER,
                        help=('Optimizer to use (gd/momentum/nag/adam)'
                        '(default: %(default)s)'))

    parser.add_argument('--batch_size',
                        dest="minibatch_size",
                        default=defaults.BATCH_SIZE, type=int,
                        help='Batch size. (default: %(default)s)')

    parser.add_argument('--anneal', dest="anneal", default=defaults.ANNEAL_LR,
                        help=('Annealing learning rate ' +
                              '(default: %(default)s)'))

    parser.add_argument('--save_dir', dest="save_dir",
                        type=str, default=defaults.SAVE_DIR,
                        help=('Model saving directory (default: %(default)s)'))

    parser.add_argument('--expt_dir', dest="expt_dir",
                        type=str, default=defaults.EXPT_DIR,
                        help=('Log files saving directory (default: %(default)s)'))

    parser.add_argument('--mnist', dest="data_path",
                        type=str, default=defaults.DATA_PATH,
                        help=('Data directory (default: %(default)s)'))

    parameters = parser.parse_args(args)

    return parameters

def start(arg, defaults, description):

	args = process_args(arg, defaults, description)

	rng = np.random.RandomState(1234)

	main.main(rng, args)

if __name__ == "__main__":
	pass
