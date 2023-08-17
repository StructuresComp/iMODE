import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os

from collections import OrderedDict

from torchdiffeq import odeint as odeint_orig
from torchdiffeq import odeint_adjoint as odeint

import argparse
import copy

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# meta parameter settings
parser = argparse.ArgumentParser('iMODE')
args = parser.parse_args()
args.atol = 1e-4
args.rtol = 1e-4
args.method = 'dopri5'
args.train_len = 100
args.test_len = args.train_len
args.eval_len = 500
args.inner_num_updates = 5
args.num_layers = 4
args.inner_lr = 1e-1
args.outer_lr = 1e-2
args.outer_num_updates = 2e2
args.num_tasks = 5
args.batch_size = 20
device = torch.device("cuda:0")
args.use_best = True
args.latent_input_num = 1

# define neural network functionals
class MLP(nn.Module):
	def __init__(self, NeuronsPerLayer=32, NumLayer=5):
		super(MLP,self).__init__()
		layer = []
		for i in range(NumLayer):
			layer.append(
				nn.Sequential(
				nn.Linear(NeuronsPerLayer, NeuronsPerLayer), 
				nn.Softplus(beta=1e1),
				)
			)
		self.net = nn.Sequential(*layer)

	def forward(self, X):
		for blk in self.net:
			X = blk(X)
		return X

class DenseBlock(nn.Module):
    def __init__(self, NeuronsPerLayer, NumLayer):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(NumLayer):
            layer.append(
                nn.Sequential(
                nn.Linear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer), 
                nn.Softplus(beta=1e1)
                )
            )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=-1)
        return X

class ODEFunc(nn.Module):
	def __init__(self, Net, latent_input=torch.tensor([1]).to(device)):
		super(ODEFunc, self).__init__()
		self.net = Net
		self.latent_input = latent_input

		# physical related matrices
		self.m = 1
		self.register_buffer('coeffMatrix1',torch.zeros(2,2).float())
		self.coeffMatrix1[1:,:1] = torch.eye(1).float()
		self.register_buffer('coeffMatrix2',torch.zeros(2,2).float())
		self.coeffMatrix2[0:1,1:] = -torch.diag(torch.tensor([1/self.m])).float()

	def forward(self, t, y):
		# the size of y in the last dimension is 3: position, velocity, latent_param
		with torch.enable_grad():
			yp = y[...,0:1].requires_grad_(True)

			Yshape = torch.tensor(yp.shape).tolist()
			Yshape[-1] = args.latent_input_num

			ya = torch.cat((yp,self.latent_input*torch.ones(Yshape).to(device)),dim=-1)
			yan = torch.cat((-yp,self.latent_input*torch.ones(Yshape).to(device)),dim=-1)
			out = self.net(ya) + self.net(yan)

			deriv = torch.autograd.grad([out.sum()],[yp],retain_graph=True,create_graph=True)
			grad = deriv[0]
			if grad is not None:
				acc = torch.matmul(y[...,0:2],self.coeffMatrix1) + torch.matmul(torch.cat((grad,torch.zeros_like(grad)),-1),self.coeffMatrix2)
				return acc

class CommonNet(nn.Module):
	def __init__(self, t, NeuronsPerLayer=32, NumLayer=5):
		super(CommonNet,self).__init__()
		self.NeuronsPerLayer = NeuronsPerLayer
		self.NumLayers = NumLayer
		self.weights = torch.tensor([0])
		self.batch_t = t.to(device)

		self.net = nn.Sequential(
			nn.Linear(1+args.latent_input_num,self.NeuronsPerLayer),
			MLP(self.NeuronsPerLayer,self.NumLayers), 
			nn.Linear(self.NeuronsPerLayer,1,bias=False)
		)

		self.func = ODEFunc(self.net)

		# physical related matrices
		self.m = 1
		self.register_buffer('coeffMatrix1',torch.zeros(2,2).float())
		self.coeffMatrix1[1:,:1] = torch.eye(1).float()
		self.register_buffer('coeffMatrix2',torch.zeros(2,2).float())
		self.coeffMatrix2[0:1,1:] = -torch.diag(torch.tensor([1/self.m])).float()

		self.param_input = torch.nn.Parameter(torch.randn(args.latent_input_num))

	def copy_weights(self, net):
		''' Set this module's weights to be the same as those of 'net' '''
		for m_from, m_to in zip(net.modules(), self.modules()):
			if isinstance(m_to, nn.Linear): # or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
				m_to.weight.data = m_from.weight.data.clone()
				if m_to.bias is not None:
					m_to.bias.data = m_from.bias.data.clone()

		self.param_input.data = net.param_input.data.clone()

	def net_forward(self, X, weights=None):
		return self.forward(X, weights)

	def forward(self, X, weights=None, batch_t=None): # todo: this should be changed, because forward means a different thing in MAML NODE. one forward means one ODE update for some time.
		if batch_t == None:
			batch_t = self.batch_t

		if weights == None: 
			x = odeint_orig(self.func, X, batch_t, atol=args.atol, rtol=args.rtol)
		else: 
			self.func.latent_input = torch.abs(weights['param_input'])
			x = odeint_orig(self.func, X, batch_t, atol=args.atol, rtol=args.rtol)
		return x

class InnerNet(CommonNet):
	def __init__(self, t, NeuronsPerLayer=32, NumLayer=5, Stepsize=1e-3):
		super(InnerNet,self).__init__(t, NeuronsPerLayer, NumLayer)

		self.stepsize = Stepsize
		self.num_updates = int(args.inner_num_updates)
		self.loss_fn = nn.MSELoss()

	def forward(self, task, num_updates=None, batch_t=None):
		if num_updates == None:
			num_updates	= self.num_updates

		inner_weights = OrderedDict((name, param) for (name, param) in self.named_parameters() if name == 'param_input')
		inner_weights_backup = copy.deepcopy(inner_weights)

		for i in range(num_updates):
			in_, target = task.X, task.Y
			in_.requires_grad_(True)

			loss, _ = self.forward_pass(in_, target, inner_weights, batch_t)
			grads = torch.autograd.grad(loss, inner_weights.values(), retain_graph=True)
			inner_weights = OrderedDict((name, param - self.stepsize*grad) for ((name, param), grad) in zip(inner_weights.items(), grads))
		tloss = loss

		# Compute the meta gradient and return it
		grads = torch.autograd.grad(loss, self.parameters())
		meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
		metrics = tloss
		return metrics, meta_grads, inner_weights

	def net_forward(self, X, weights=None, batch_t=None):
		return super(InnerNet, self).forward(X, weights, batch_t)

	def forward_pass(self, in_, target, weights=None, batch_t=None):
		''' Run data through net, return loss and output '''
		out = self.net_forward(in_, weights, batch_t)
		loss = self.loss_fn(out, target)
		return loss, out

class Net(nn.Module):
	def __init__(self, Task, NeuronsPerLayer=32, NumLayer=5, beta=1e-3):
		super(Net,self).__init__()

		self.num_updates = int(args.outer_num_updates) # perform how many rounds of update on the meta-parameter
		self.t = torch.linspace(0,10,1001)
		self.test_t = torch.linspace(0,10,1001)[:args.test_len].to(device)
		self.eval_t = torch.linspace(0,10,1001)[:args.eval_len].to(device)
		self.batchtime = args.train_len
		self.loss_fn = nn.MSELoss()
		self.net = CommonNet(self.t[:self.batchtime], NeuronsPerLayer, NumLayer)
		self.net.to(device)

		self.best_net = copy.deepcopy(self.net)
		self.lowest_loss = torch.tensor([100])

		self.inner_net = InnerNet(self.t[:self.batchtime], NeuronsPerLayer, NumLayer, args.inner_lr)
		self.inner_net.to(device)
		self.stepsize = beta
		self.task = Task
		self.num_tasks = args.num_tasks
		self.batchsize = args.batch_size
		
		self.opt = optim.Adam(self.net.parameters(), lr=self.stepsize)
		self.metrics_his = []

	def meta_update(self, task, grads):
		print('\n')
		in_, target = task.X, task.Y
		# We use a dummy forward / backward pass to get the correct grads into self.net
		out = self.net.net_forward(in_)
		# print('before loss_fn')
		loss = self.loss_fn(out, target)
		# Unpack the list of grad dicts
		gradients = {k: sum(d[k] for d in grads) for k in grads[0].keys()}
		# Register a hook on each parameter in the net that replaces the current dummy grad
		# with our grads accumulated across the meta-batch
		hooks = []
		for (k,v) in self.net.named_parameters():
			def get_closure():
				key = k
				def replace_grad(grad):
					return gradients[key]
				return replace_grad
			hooks.append(v.register_hook(get_closure()))
		# Compute grads for current step, replace with summed gradients as defined by hook
		self.opt.zero_grad()
		loss.backward()
		# Update the net parameters with the accumulated gradient according to optimizer
		self.opt.step()
		# Remove the hooks before next training phase
		for h in hooks:
			h.remove()
		print('the training param_input is: ' + str(self.inner_net.param_input.clone().cpu().detach().numpy()))

	def train(self):
		for itr in range(self.num_updates):
			grads = []
			tloss = 0.0
			for i in range(self.num_tasks):
				# here we assume that we can read all data for the ith task at once
				# get data for task i
				task = self.task.getTask(i,self.batchsize,self.batchtime)

				self.inner_net.copy_weights(self.net)
				# for the i task, the following call generates the gradient of the ith loss with respect to the meta-parameter
				trl, g, _ = self.inner_net.forward(task)
				print('Task: ' + str(i) + '; Loss: ' + str(trl.item()))
				# accumulate all the grads for the meta-parameter
				grads.append(g)
				tloss += trl
			print('train loss is: \033[1;32;43m{:.6f}\033[0m'.format(tloss.item()))
			self.metrics_his.append(tloss)

			if tloss.item() < self.lowest_loss:
				self.best_net = copy.deepcopy(self.net)
				self.lowest_loss = tloss.item()

			# Perform the meta update
			print('Meta update' + str(itr))
			self.meta_update(task, grads)

	def test(self, task, num_updates_test=None, test_t=None, use_best=False):
		tloss = 0.0
		if test_t == None:
			test_t = self.test_t

		if use_best:
			self.inner_net.copy_weights(self.best_net)
		else:
			self.inner_net.copy_weights(self.net)

		# for the i task, the following call generates the gradient of the ith loss with respect to the meta-parameter
		trl, g, self.test_inner_weights = self.inner_net.forward(task, num_updates_test, test_t)
		# accumulate all the grads for the meta-parameter
		tloss += trl
		print('test loss is: ' + str(tloss))
		print('the testing param_input is: ' + str(self.test_inner_weights['param_input'].clone().cpu().detach().numpy()))

	def evaluate(self, task, eval_t=None):
		if eval_t == None:
			eval_t = self.eval_t
		self.inner_net.eval()
		loss, output = self.inner_net.forward_pass(task.X, task.Y, self.test_inner_weights, eval_t)
		self.inner_net.train()
		return output

class Task():
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

class PendulumTask():
	def __init__(self, X, labels):
		self.data = X
		self.labels = labels
	def getTask(self, i, batch_size, batch_time):
		index = torch.randint(0, self.data.size()[1]-batch_time-1, (batch_size,))
		return Task(self.data[i,index,:].view(-1,2), torch.stack([self.data[i,index+j,:] for j in range(batch_time)], dim=0)) 

def readData():
	ks = torch.tensor([1,3,5,7,9,2,4,6,8,10,3.5,5.1,6.9]).float()
	numTimeSteps = 1001
	X = torch.zeros([len(ks),numTimeSteps,2])
	for ii in range(len(ks)):
		# read in the file
		X[ii,...] = torch.from_numpy(np.loadtxt('./data/pendulum/{:.1f}.txt'.format(ks[ii]),delimiter=',')).float()
	return X[:,0:-1:1], ks

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

	seed_torch()

	# make training dataset
	X, ks = readData()
	X = X.to(device)
	PTask = PendulumTask(X, ks)

	# create the neural network and train
	net = Net(PTask, 32, args.num_layers, args.outer_lr)
	net.train()

	# plot loss history
	fig, ax = plt.subplots()
	ax.semilogy(np.array(net.metrics_his), linewidth=2.0)
	plt.show()

	# testing the performance
	param_input_his = np.zeros((13,args.latent_input_num))
	param_true_his = np.zeros((13,1))
	for j in range(13):
		# adaptation/test task
		TestTask = PTask.getTask(j, args.batch_size, args.test_len)

		# adaptation
		net.test(TestTask, use_best=args.use_best)
		param_input_his[j,:] = net.test_inner_weights['param_input'].clone().detach().cpu().numpy()
		param_true_his[j,:] = ks[j].clone().detach().cpu().numpy()

		# evaluation
		EvaTask = PTask.getTask(j, 1, args.eval_len)
		Output = net.evaluate(EvaTask)

		fig, ax = plt.subplots()
		ax.scatter(EvaTask.Y[:,0,0].detach().cpu().numpy(), EvaTask.Y[:,0,1].detach().cpu().numpy(), linewidth=2.0)
		ax.scatter(Output[:,0,0].detach().cpu().numpy(), Output[:,0,1].detach().cpu().numpy(), linewidth=1.0, marker='s')
		plt.show()
