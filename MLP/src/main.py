__author__ = 'Rakesh'

import numpy as np
import cPickle
import gzip

class HiddenLayer():

	def __init__(self, rng, n_input, n_output, act = 'sigmoid'):

		self.W = np.array(
					rng.uniform(
                    low=-np.sqrt(6. / (n_input + n_output)),
                    high=np.sqrt(6. / (n_input + n_output)),
                    size=(n_input, n_output)
                	)
                	, dtype=float)

		self.b = np.zeros((n_output,), dtype=float)
		if act=="tanh":
			self.act = self.tanh
		elif act=="sigmoid":
			self.act = self.sigmoid

	def forward_prop(self, input):

		lin_output = np.dot(input,self.W) + self.b
		self.output = self.act(lin_output)

		return self.output

	def backward_prop(self,input, loss):

		grad_W = input*loss
		grad_b = loss
		grad_input = np.dot(loss,self.W)
		return grad_input, (grad_W, grad_b)

	def tanh(self, input):
		return np.tanh(input)

	def sigmoid(self, input):
		return 1/(1 + np.exp(-x))

	def grad_tanh(self, input):
		return 1.0 - np.tanh(x)**2

	def grad_sigmoid(self, input):
		return sigmoid(input)*(1-sigmoid(input))


class MLP():
	
	def __init__(self, rng, args, n_in, n_out):

		self.max_epoch 		= args.max_epoch
		self.lr 	   		= args.lr_start
		self.momentum  		= args.momentum
		self.num_hidden 	= args.num_hidden
		self.sizes 			= eval(args.hid_sizes)
		self.activation 	= args.act
		self.loss 			= args.loss
		self.optimizer 		= args.optimizer
		self.minibatch_size = args.minibatch_size
		self.anneal 		= args.anneal
		self.n_in 			= n_in
		self.n_out			= n_out
		self.rng			= rng

		self.buildNetwork()

	def buildNetwork(self):

		self.layers = []
		self.layers.append(HiddenLayer(self.rng, self.n_in, self.sizes[0], self.activation))

		for i in xrange(self.num_hidden-1):
			self.layers.append(HiddenLayer(self.rng, self.sizes[i], self.sizes[i+1], self.activation))

		self.layers.append(HiddenLayer(self.rng, self.sizes[-1], self.n_out, self.activation))


	#def train():

		#Code for forward-prop


		#Code for backward-prop

def loadData(args):
	
	f = gzip.open(args.data_path, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	train_x, train_y 	= train_set
	valid_x, valid_y 	= valid_set
	test_x, test_y 		= test_set

	return train_set, valid_set, test_set
	


def main(rng,args):

	train, valid, test = loadData(args)

	n_in 	= np.shape(train[0])[1]
	n_out 	= 10   							#For MNIST
	
	network = MLP(rng, args, n_in, n_out)

	# network.train()


if __name__ == "__main__":
	main()

