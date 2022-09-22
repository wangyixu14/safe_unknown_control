import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import scipy.io as sio


if __name__ == '__main__':


	class Controller(nn.Module):
	    def __init__(self, data_size=13, hidden_size=64, u_size=3):
	        super(Controller, self).__init__()
	        self.lin1 = nn.Linear(data_size, hidden_size)
	        self.lin2 = nn.Linear(hidden_size, hidden_size)
	        self.lin3 = nn.Linear(hidden_size, u_size)
	        self.relu=nn.ReLU()

	    def forward(self, state):
	        x = state
	        x = self.relu(self.lin1(x))
	        x = self.relu(self.lin2(x))
	        return self.lin3(x)
	
	# state_collection = []
	# control_collection = []
	net = Controller()

	# for i in range(100):
	# 	trial = sio.loadmat('PDP_OC_results_trial_'+str(i)+'.mat')
	# 	state = trial['results']['true_solution'][0, 0][0, 0][0][:-1]
	# 	control = trial['results']['true_solution'][0, 0][0, 0][1]
	# 	state_collection.append(state)
	# 	control_collection.append(control)

	# state_collection = torch.from_numpy(np.reshape(state_collection, (-1, 13))).float()
	# control_collection = torch.from_numpy(np.reshape(control_collection, (-1, 3))).float()

	state_collection = torch.from_numpy(np.load('state.npy')).float()
	action_collection = torch.from_numpy(np.load('action.npy')).float()
	print(state_collection.shape, action_collection.shape)
	# assert False
	# y = torch.from_numpy(np.reshape(dataset[:, -1], (len(dataset[:, -1]), 1))).float()
	# x = torch.from_numpy(dataset[:, :3]).float()



	def train(inputdata, label, net):
		optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)
		criterion = torch.nn.MSELoss()  

		BATCH_SIZE = 100
		EPOCH = 100

		torch_dataset = Data.TensorDataset(inputdata, label)

		loader = Data.DataLoader(
			dataset=torch_dataset, 
			batch_size=BATCH_SIZE, 
			shuffle=True, num_workers=2,)

		for epoch in range(EPOCH):
			loss_list = []
			for step, (batch_x, batch_y) in enumerate(loader, 0): 
				prediction = net(batch_x)    
				loss = criterion(prediction, batch_y)     
				loss_list.append(loss.data.numpy())
				optimizer.zero_grad()   
				loss.backward()         
				optimizer.step()       
			print(np.sum(loss_list), len(loss_list))
		torch.save(net.state_dict(), './rocket_nominal_control.pth')

	def fgsm(model, X, y, epsilon=0.04):
	    delta = torch.zeros_like(X, requires_grad=True)
	    loss = -nn.MSELoss()(model(X + delta), y)
	    loss.backward()
	    return epsilon * delta.grad.detach().sign()

	def robust_train(inputdata, label, net):
		optimizer = torch.optim.Adam(net.parameters(), weight_decay=5e-5)
		criterion = torch.nn.MSELoss()  

		BATCH_SIZE = 100
		EPOCH = 100

		torch_dataset = Data.TensorDataset(inputdata, label)

		loader = Data.DataLoader(
			dataset=torch_dataset, 
			batch_size=BATCH_SIZE, 
			shuffle=True, num_workers=2,)

		for epoch in range(EPOCH):
			loss_list = []
			for step, (batch_x, batch_y) in enumerate(loader, 0):
				if np.random.uniform(low=0, high=1, size=1)[0] > 0.7:
					delta = fgsm(net, batch_x, batch_y)
					prediction = net(batch_x+delta)
				else:
					prediction = net(batch_x)    
				loss = criterion(prediction, batch_y)     
				loss_list.append(loss.data.numpy())
				optimizer.zero_grad()   
				loss.backward()         
				optimizer.step()       
			print(np.sum(loss_list), len(loss_list))
		torch.save(net.state_dict(), './robust_distill.pth')

	train(state_collection, action_collection, net)
	# robust_train(x, y, Individual)