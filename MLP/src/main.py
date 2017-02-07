__author__ = 'Rakesh'


# TODO:
# 1. Saving Model
# 2. Dropouts
# 3. Batch-norm
import numpy as np
import cPickle
import gzip

def create_onehot(target_class, shape1, shape2):
	one_hot_vec = np.zeros((shape1, shape2))

	idx = 0
	for i in range(shape1):
		idx = target_class[i]
		one_hot_vec[i][idx] = 1
	return one_hot_vec

class FinalLayer():

	def __init__(self, rng, n_input, n_output, args, W=None, b=None):

		if W == None:
			self.W = np.array(
						rng.uniform(
	                    low=-np.sqrt(6. / (n_input + n_output)),
	                    high=np.sqrt(6. / (n_input + n_output)),
	                    size=(n_input, n_output)
	                	)
	                	, dtype=float)
		else:
			self.W = W

		if b == None:
			self.b = np.zeros((n_output,), dtype=float)
		else:
			self.b = b

		if args.loss == "ce":
			self.act = self.softmax
		elif args.act == "sq":
			self.act = self.identity

		self.args = args
		self.lr = args.lr_start
		self.delta_w = np.zeros((n_input, n_output))
		self.delta_b = np.zeros((n_output,))
		self.momentum_w = np.zeros((n_input, n_output))
		self.momentum_b = np.zeros((n_output,))
		self.v_w = np.zeros((n_input, n_output))
		self.v_b = np.zeros((n_output,))
		self.epsilon = 0.001
		self.beta1 = 0.9
		self.beta2 = 0.999

	def anneal_lr(self):
		self.lr = self.lr/2

	def forward_pass(self, input):

		self.input = input
		lin_output = np.dot(self.input,self.W) + np.reshape(self.b, (1, self.b.shape[0]));
		self.output = self.act(lin_output)

		return self.output

	def softmax(self, input):
		return (np.exp(input).T / np.sum(np.exp(input), axis=1)).T

	def identity(self, input):
		return input

	def backward_prop(self, one_hot_vec):

		if self.args.loss == "ce":
			gradOutput = self.grad_calc(-(one_hot_vec - self.output)) 
		elif self.args.loss == "sq":
			gradOutput == self.grad_calc((one_hot_vec - self.output))

		return gradOutput

	def grad_calc(self, grad_a):

		grad_W = np.dot(self.input.T,grad_a)
		grad_b = grad_a
		gradInput = np.dot(grad_a,self.W.T)
		self.updateParameters(grad_W, grad_b)
		return gradInput

	def updateParameters(self, grad_W, grad_b):

		if self.args.optimizer == "gd":
			self.gd(grad_W, grad_b)
		elif self.args.optimizer == "adam":
			self.adam(grad_W, grad_b)
		elif self.args.optimizer == "nag":
			self.nag(grad_W,grad_b)
		elif self.args.optimizer == "momentum":
			self.momentum(grad_W, grad_b)

	def gd(self, grad_W, grad_b):

		self.delta_w = self.lr*grad_W
		self.delta_b = self.lr*grad_b
		self.update(self.delta_w, self.delta_b)

	def adam(self, grad_W, grad_b):

		self.momentum_w = self.beta1 * self.momentum_w + (1-self.beta1) * grad_W
		self.momentum_b = self.beta1 * self.momentum_b + (1-self.beta1) * grad_b

		self.v_w = self.beta2 * self.v_w + (1-self.beta2) * grad_W**2
		self.v_b = self.beta2 * self.v_b + (1-self.beta2) * grad_b**2

		self.delta_w = self.lr/(np.sqrt(self.v_w + self.epsilon))*self.momentum_w
		self.delta_b = self.lr/(np.sqrt(self.v_b + self.epsilon))*self.momentum_b

		self.update(self.delta_w, self.delta_b)

	def nag(self, grad_W, grad_b):
		raise ValueError('Not applicable yet')

	def momentum(self, grad_W, grad_b):

		self.delta_w = args.momentum * self.delta_w + self.lr * grad_W
		self.delta_b = args.momentum * self.delta_b + self.lr * grad_W
		self.update(self.delta_w, self.delta_b)

	def update(self, delta_w, delta_b):

		self.W -= self.delta_w
		self.b -= self.delta_b.mean(0)

class HiddenLayer():

	def __init__(self, rng, n_input, n_output, args, W=None, b=None):

		if W==None:
			self.W = np.array(
						rng.uniform(
	                    low=-np.sqrt(6. / (n_input + n_output)),
	                    high=np.sqrt(6. / (n_input + n_output)),
	                    size=(n_input, n_output)
	                	)
	                	, dtype=float)
		else:
			self.W = W

		if b==None:
			self.b = np.zeros((n_output,), dtype=float)
		else:
			self.b = b

		if args.act=="tanh":
			self.act = self.tanh
			self.grad_act = self.grad_tanh
		elif args.act=="sigmoid":
			self.act = self.sigmoid
			self.grad_act = self.grad_sigmoid

		self.args = args
		self.lr = args.lr_start
		self.delta_w = np.zeros((n_input, n_output))
		self.delta_b = np.zeros((n_output,))
		self.momentum_w = np.zeros((n_input, n_output))
		self.momentum_b = np.zeros((n_output,))
		self.v_w = np.zeros((n_input, n_output))
		self.v_b = np.zeros((n_output,))
		self.epsilon = 0.001
		self.beta1 = 0.9
		self.beta2 = 0.999

	def anneal_lr(self):
		self.lr = self.lr/2

	def forward_pass(self, input):

		self.input = input
		lin_output = np.dot(self.input,self.W) + np.reshape(self.b, (1, self.b.shape[0]));
		self.output = self.act(lin_output)

		return self.output

	def backward_prop(self,gradOutput):

		grad_a = np.multiply(self.grad_act(), gradOutput)
		grad_W = np.dot(self.input.T,grad_a)
		grad_b = grad_a
		gradInput = np.dot(grad_a,self.W.T)
		self.updateParameters(grad_W, grad_b)
		return gradInput

	def updateParameters(self, grad_W, grad_b):

		if self.args.optimizer == "gd":
			self.gd(grad_W, grad_b)
		elif self.args.optimizer == "adam":
			self.adam(grad_W, grad_b)
		elif self.args.optimizer == "nag":
			self.nag(grad_W,grad_b)
		elif self.args.optimizer == "momentum":
			self.momentum(grad_W, grad_b)


	def tanh(self, input):
		return np.tanh(input)

	def sigmoid(self, input):
		return 1.0/(1 + np.exp(-input))

	def grad_tanh(self):
		return 1.0 - self.output**2

	def grad_sigmoid(self):
		return np.multiply(self.output,(1-self.output))

	def gd(self, grad_W, grad_b):

		self.delta_w = self.lr*grad_W
		self.delta_b = self.lr*grad_b
		self.update(self.delta_w, self.delta_b)

	def adam(self, grad_W, grad_b):

		self.momentum_w = self.beta1 * self.momentum_w + (1-self.beta1) * grad_W
		self.momentum_b = self.beta1 * self.momentum_b + (1-self.beta1) * grad_b

		self.v_w = self.beta2 * self.v_w + (1-self.beta2) * grad_W**2
		self.v_b = self.beta2 * self.v_b + (1-self.beta2) * grad_b**2

		self.delta_w = self.lr/(np.sqrt(self.v_w + self.epsilon))*self.momentum_w
		self.delta_b = self.lr/(np.sqrt(self.v_b + self.epsilon))*self.momentum_b

		self.update(self.delta_w, self.delta_b)

	def nag(self, grad_W, grad_b):
		raise ValueError('Not applicable yet')

	def momentum(self, grad_W, grad_b):

		self.delta_w = args.momentum * self.delta_w + self.lr * grad_W
		self.delta_b = args.momentum * self.delta_b + self.lr * grad_W
		self.update(self.delta_w, self.delta_b)

	def update(self, delta_w, delta_b):

		self.W -= self.delta_w
		self.b -= self.delta_b.mean(0)


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
		self.args 			= args

		self.buildNetwork()

	def buildNetwork(self):

		self.layers = []
		if np.asarray([self.sizes]).ndim > 1:
			self.layers.append(HiddenLayer(self.rng, self.n_in, self.sizes[0], self.args))

			for i in xrange(self.num_hidden-1):
				self.layers.append(HiddenLayer(self.rng, self.sizes[i], self.sizes[i+1], self.args))

			self.layers.append(FinalLayer(self.rng, self.sizes[-1], self.n_out, self.args))
		else:
			self.layers.append(HiddenLayer(self.rng, self.n_in, self.sizes, self.args))
			self.layers.append(FinalLayer(self.rng, self.sizes, self.n_out, self.args))


	def train(self, train, valid, test):

		train_x, train_y 	= train
		valid_x, valid_y 	= valid
		test_x, test_y 		= test

		best_valid_loss = 0
		best_test_loss = 0
		best_test_steps = 0
		best_test_epochs = 0
		prev_train_loss = 0
		for j in xrange(self.max_epoch):

			n_train_batches = train_x.shape[0] // self.minibatch_size

			train_loss = []

			for i in xrange(n_train_batches):

				x = train_x[i*self.minibatch_size:(i+1)*self.minibatch_size]
				y = train_y[i*self.minibatch_size:(i+1)*self.minibatch_size]
				y_onehot = create_onehot(y, self.minibatch_size, self.n_out)

				#Code for forward-prop

				input = x
				for ii in range(len(self.layers)):
					output = self.layers[ii].forward_pass(input)
					input = output

				train_score = 0
				for ii in xrange(y.shape[0]):
					if (y[ii]==np.argmax(output[ii,:])):
						train_score +=1
				train_loss.append(float(train_score)/self.minibatch_size)

				# Loss Computation

				if self.loss=="ce":
					saving_loss = np.mean(np.sum(np.multiply(y_onehot,output), axis=1), axis=0)
				elif self.loss=="sq":
					saving_loss = np.mean(np.sum((y_onehot-output)**2, axis=1), axis=0)


				# #Code for backward-prop
				gradOutput = y_onehot
				for ii in range(len(self.layers)):
					gradOutput = self.layers[len(self.layers)-(ii+1)].backward_prop(gradOutput)

				if (i+1)%100==0:

					# f = open(self.args.expt_dir+"log_loss_train.txt",'a')
					# np.savetxt(f, "Epoch %d, Step %d, Loss: %.4f, lr: %.4f"%(j,i,saving_loss, self.layers[-1].lr))
					# f.close()
					# f = open(self.args.expt_dir+"log_error_train.txt",'a')
					# np.savetxt(f, "Epoch %d, Step %d, Error: %.4f, lr: %.4f"%(j,i,np.mean(train_loss), self.layers[-1].lr))
					# f.close()

					# VALIDATION
					n_valid_minibatches = valid_x.shape[0] // self.minibatch_size
					valid_loss = []

					for k in xrange(n_valid_minibatches):
						
						input = valid_x[k*self.minibatch_size:(k+1)*self.minibatch_size]
						y = 	valid_y[k*self.minibatch_size:(k+1)*self.minibatch_size]
						y_onehot = create_onehot(y, self.minibatch_size, self.n_out)

						for ii in range(len(self.layers)):
							output = self.layers[ii].forward_pass(input)
							input = output

						# f = open(self.args.expt_dir+"epoch_" + j + "/steps_" + i +"/" + "valid_predictions.txt",a)
						# np.savetxt(f,output)
						# f.close()

						valid_score = 0
						for ii in xrange(y.shape[0]):
							if (y[ii]==np.argmax(output[ii,:])):
								valid_score +=1
						valid_loss.append(float(valid_score)/self.minibatch_size)

					if np.mean(valid_loss) > best_valid_loss:
						best_valid_loss = np.mean(valid_loss)
					else:
						if self.args.anneal and (prev_train_loss < np.mean(train_loss)):
							for ii in range(len(self.layers)):
								self.layers[ii].anneal_lr()
						#print("TRUE")

					if self.loss=="ce":
						saving_loss = np.mean(np.sum(np.multiply(y_onehot,output), axis=1), axis=0)
					elif self.loss=="sq":
						saving_loss = np.mean(np.sum((y_onehot-output)**2, axis=1), axis=0)

					# f = open(self.args.expt_dir+"log_loss_valid.txt",'a')
					# np.savetxt(f, "Epoch %d, Step %d, Loss: %.4f, lr: %.4f"%(j,i,saving_loss, self.layers[-1].lr))
					# f.close()
					# f = open(self.args.expt_dir+"log_error_valid.txt",'a')
					# np.savetxt(f, "Epoch %d, Step %d, Error: %.4f, lr: %.4f"%(j,i,np.mean(valid_loss), self.layers[-1].lr))
					# f.close()


					# TESTING
					n_test_minibatches = test_x.shape[0] // self.minibatch_size
					test_loss = []

					for k in xrange(n_test_minibatches):
						
						input = test_x[k*self.minibatch_size:(k+1)*self.minibatch_size]
						y 	  = test_y[k*self.minibatch_size:(k+1)*self.minibatch_size]
						y_onehot = create_onehot(y, self.minibatch_size, self.n_out)

						for ii in range(len(self.layers)):
							output = self.layers[ii].forward_pass(input)
							input = output

						# f = open(self.args.expt_dir+"epoch_" + j + "/steps_" + i +"/" + "test_predictions.txt",a)
						# np.savetxt(f,output)
						# f.close()

						test_score = 0
						for ii in xrange(y.shape[0]):
							if (y[ii]==np.argmax(output[ii,:])):
								test_score +=1
						test_loss.append(float(test_score)/self.minibatch_size)

					if self.loss=="ce":
						saving_loss = np.mean(np.sum(np.multiply(y_onehot,output), axis=1), axis=0)
					elif self.loss=="sq":
						saving_loss = np.mean(np.sum((y_onehot-output)**2, axis=1), axis=0)

					# f = open(self.args.expt_dir+"log_loss_test.txt",'a')
					# np.savetxt(f, "Epoch %d, Step %d, Loss: %.4f, lr: %.4f"%(j,i,saving_loss, self.layers[-1].lr))
					# f.close()
					# f = open(self.args.expt_dir+"log_error_test.txt",'a')
					# np.savetxt(f, "Epoch %d, Step %d, Error: %.4f, lr: %.4f"%(j,i,np.mean(test_loss), self.layers[-1].lr))
					# f.close()

					if np.mean(test_loss) > best_test_loss:
						best_test_loss = np.mean(test_loss)
						best_test_steps = i
						best_test_epoch = j

					print "Epoch : %d Steps : %d Train Accuracy : %.4f Validation Accuracy : %.4f Test Accuracy : %.4f LR : %.4f"%(j, i+1, 100*(np.mean(train_loss)), 100*(np.mean(valid_loss)), 100*(np.mean(test_loss)), self.layers[-1].lr)
				prev_train_loss = np.mean(train_loss)
		print "Best test accuracy on dataset : %.4f after %d epochs and %d steps of training."%(best_test_loss, best_test_epoch, best_test_steps)

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

	idx = np.arange(np.shape(train[0])[0])
	rng.shuffle(idx)
	train = train[0][idx], train[1][idx]

	idx = np.arange(np.shape(valid[0])[0])
	rng.shuffle(idx)
	valid = valid[0][idx], valid[1][idx]

	idx = np.arange(np.shape(test[0])[0])
	rng.shuffle(idx)
	test = test[0][idx], test[1][idx]

	network.train(train, valid, test)


if __name__ == "__main__":
	main()

