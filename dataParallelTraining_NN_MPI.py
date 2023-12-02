
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from mpi4py import MPI
import argparse


class RegressionDataset(torch.utils.data.Dataset):  
    '''
    Prepare the dataset for regression
    '''

    def __init__(self, X, y, scale_data=True):
        
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
              X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
      



class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(2, 3),
          nn.ReLU(),
          nn.Linear(3, 1)
        )

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)




def dist_train(args):

    """Schedule a distributed training job."""
        
    # Fetch MPI environment settings:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    
    if rank == 0:
    
        # set the random seed to be different for each process:
        torch.manual_seed(rank)
        
        # Create dataset
        X, y = make_regression(n_samples=16, n_features=2, noise=1, random_state=42)
        XY = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        dims = XY.shape  

    else:
        
        XY = None
        dims = None
        grad_list = None
        
    
    # Build a fresh model:
    model = MLP()
    
    # Send model parameters from root process (rank=0) to workers so all processes have the same model
    dict_init = comm.bcast(model.state_dict(), root=0)
    model.load_state_dict(dict_init)
            
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # Loss function
    loss_function = nn.MSELoss()
    
    # Send dimensions of the dataset to every worker
    (h, w) = comm.bcast(dims, root=0)
    
    # Check if the legnth of the dataset is a multiple of the number of processes
    result, residue = divmod(h, nprocs)
    if residue == 0:
        
        rows = result 
        columns = w
        XY_recv = np.empty((rows, columns), dtype=np.float64)

        # Partition of the dataset by rows 
        comm.Scatter(XY, XY_recv, root=0)
    
    else:
        
        if rank == 0:
            
            sendbf = XY.flatten()
            
            # count: the size of each sub-task
            count = [result + 1 if p < residue else result for p in range(nprocs)]
            count = w*np.array(count, dtype=np.int8)

            # displacement: the starting index of each sub-task
            displ = [sum(count[:p]) for p in range(nprocs)]
            displ = np.array(displ)

        else:
        
            sendbf = None
            # initialize count on worker processes
            count = np.empty(nprocs, dtype=np.int8)
            displ = None

        
        # broadcast count
        comm.Bcast([count, MPI.INT], root=0)
        
        # initialize recvbuf on all processes
        recvbf = np.zeros(count[rank])

        comm.Scatterv([sendbf, count, displ, MPI.DOUBLE], recvbf, root=0)


    # Define torch dataset and dataloader for each worker
    if residue != 0:
        XY_recv = recvbf.reshape(int(count[rank]/w), w)
        
    dataset = RegressionDataset(XY_recv[:, 0:2], XY_recv[:, 2])
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=XY_recv.shape[0], shuffle=True) 
    

    # Main loop over epochs
    for i in range(args.nepochs):
        
        print("[ = = = = = Epoch {} = = = = = ]".format(i))
        
        # Loop over over batches
        for j, data in enumerate(trainloader, 0):
            
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            #print('Batch ', j, 'data ', data)

            # Set model to train
            model.train()

            # Zero the gradients
            optimizer.zero_grad()
                
            # Perform forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Obtain gradients of the loss function with respect to the model parameters in each worker
            gradients = []
            with torch.no_grad():
                for k, param in enumerate(model.parameters()):
                    gradients.append(param.grad)
                    
            # Gather gradients from workers to root
            grad_list = comm.gather(gradients, root=0)

            # Compute the average of the gradients in root and send the result to the workers
            if rank == 0:
            
                avg_grads = []
                len_grads_in_model = len(grad_list[0])
                for k in range(len_grads_in_model):
                    grad_by_proc = 0
                    for l in range(nprocs):
                        grad_by_proc += grad_list[l][k]
                        
                    avg_grads.append(grad_by_proc/nprocs)
                
                [comm.send(avg_grads, k) for k in range(1,nprocs)]
                
            else:
            
                avg_grads = comm.recv(source=0)
                
            # Overwrite gradients of the parameters in each worker with the averaged gradient
            with torch.no_grad():
                for k, param in enumerate(model.parameters()):
                    param.grad = avg_grads[k]
            
            # Perform step using averaged gradient
            optimizer.step()
            
            '''
            # Validation of the model only in rank 0
            if rank == 0:
            
                model.eval()
                with torch.no_grad():
                    output_val = model(x_val)
                    loss_val = loss_function(output_val, y_val)'''


        
        print(f'loss in worker {rank}: {loss.item()}')
        
    
    '''
    if rank == 0:
        
        # Predict
        pass
        
        model.eval()
        with torch.no_grad():
            output_test = model(x_test)
            loss_test = loss_function(output_test, y_test)'''
        
        

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train network across multiple distributed processes.")
    parser.add_argument("--lr", dest="lr", default=0.001,
                        help="Learning rate for SGD optimizer. [0.9]")
    parser.add_argument("--momentum", dest="momentum", default=0.9,
                        help="Momentum for SGD optimizer [0.9].")
    parser.add_argument("--batch_size", dest="batch_size", default=4,
                        help="Batch size to use for each process.")
    parser.add_argument("--nepochs", dest="nepochs", default=3, type=int,
                        help="Number of epochs (times to loop through the dataset).")
    args = parser.parse_args()
    
    dist_train(args)

