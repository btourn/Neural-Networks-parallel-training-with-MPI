# Neural-Networks-parallel-training-with-MPI

This work is about a parallel implementation of a neural network training using the MPI standard in Python, with the $mpi4py$ library. The NN is developed using PyTorch.

## Basic explanation
An NN model is created and replicated over all the processing cores. The dataset used for training the NN is splitted into a number of subsets matching the number of cores, also contemplating the case where the number of rows in the dataset is not a multiple of the number of cores. Each worker computes de loss function and the gradient, using the assigned subset. Then, the gradients computed at each worker are gathered in the root, and averaged. Then, the model parameters are updated using such averaged gradient. 


## Script "dataParallelTraining_NN_MPI.py":
The script "dataParallelTraining_NN_MPI.py" can be executed in a PC (or eventually in a cluster, I didn't test it yet) using the following command in the console:

$$\texttt{mpiexec -n numprocs python dataParallelTraining_NN_MPI.py}$$

where $\texttt{numprocs}$ referes to the number of processes used for the execution. Additional input arguments can be passed to the script (see the "$\texttt{if __name__=='__main__'}$" block of code).

