'''
This file contains the minimal standalone implementation of KeySDL.
For adaptation to more complex use cases, see the repository containing the full scripts used in the KeySDL paper at [link].

Usage example:

./python3 data.csv [out_dir] [compositional]

Inputs:

data.csv [string]: a sv file where each column is a microbe and each row is a steady-state observation.
The header row must contain microbe names, and there must not be an index column.

out_dir [string]: path to folder for output files. If not provided, KeySDL creates a folder called "out" in the current working directory.

compositional [boolean]: whether to model as compositional replicator or absolute GLV dynamics.
Default value of True is appropriate for all sequencing datasets without quantification

Output files:

A.csv: GLV/replicator interactions matrix A.
r.csv: Growth rates r (all ones for replicator system).
dropout_keystones.csv: Simulated impact on removal for each microbe.
simulated_abundance.csv: Simulated mean relative abundance of each microbe across 500 random steady states.

A python3 environment with numpy, pandas, and (py)torch is required. See the repository at [link] for directions

'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import os

class glv_simulator:
    def __init__(self,A,r):
        self.A = A
        self.r = r
        return

    def ss_from_assemblage(self,assemblage,compositional=True):
        # performs perturbation calculations
        # assemblage > 0 where a microbe is present
        n = self.A.shape[0]
        result = np.zeros(n)
        num_idx = np.argwhere(assemblage > 0)
        while True:
            A_p = self.A[num_idx,:].reshape(len(num_idx),n)
            A_p = A_p[:,num_idx].reshape(len(num_idx),len(num_idx))
            r_p = self.r[num_idx]
            A_p_i = np.linalg.inv(A_p)
            result[:] = 0
            result[num_idx] = -np.matmul(A_p_i,r_p)
            if not (result < 0).any():
                break
            else:
                num_idx = num_idx[np.where(result[num_idx] > 0)[0]]
        if compositional:
            return result/np.sum(result)
        else:
            return result
    
    def compute_dropouts(self,compositional=True):
        n = self.A.shape[0]
        # compute baseline
        baseline = self.ss_from_assemblage(np.ones(n))
        dropout = np.zeros((n,n))
        for i in range(n):
            dropout[i,:] = self.ss_from_assemblage(np.arange(n)!=i)
        if compositional:
            return baseline/np.sum(baseline),(dropout.T/np.sum(dropout,axis=1)).T
        else:
            return baseline,dropout
        
    def bcd_keystones(self):
            baseline,dropout = self.compute_dropouts(compositional=True) # bcd operates on compositions
            baseline = np.repeat(baseline.reshape(1,-1),dropout.shape[1],axis=0)
            baseline = baseline*(dropout!=0)
            return np.sum(np.abs(dropout-baseline),axis=1)/np.sum(np.abs(dropout+baseline),axis=1)

def random_training_samples(A,r,n_train_samples,p_zero_train=0.1, seed=None):
    n = A.shape[0]
    rng = np.random.default_rng(seed=seed)
    train_samples = np.where(rng.random((n_train_samples,n)) < p_zero_train,0.,1.)

    for i,idx in enumerate(train_samples):
        idx = np.array(idx)
        if sum(idx) != 0:
            num_idx = np.argwhere(idx).reshape(-1)
            sub_A = A[num_idx,:]
            sub_A = sub_A[:,num_idx]
            sub_A_i = np.linalg.inv(sub_A)
            sub_r = r[num_idx]
            ss = -np.matmul(sub_A_i,sub_r)
            train_samples[i,:] = 0
            train_samples[i,num_idx] = ss
    return train_samples

#%%

class ss_optim(nn.Module):
    def __init__(self,n,A=None,r=None,compositional=False):
        super().__init__()
        if A is None:
            # A must have negative diagonal to represent finite carrying capacity
            A_init=-torch.eye(n)
            # A will be mostly negative so start it on that side of 0
            A_init = torch.where(A_init == 0,-1e-6,A_init)
            self.A = nn.Parameter(A_init)
        else: # allow passing of starting value of A
            self.A = nn.Parameter(A)
        if r is None:
            self.r = nn.Parameter(torch.ones(n))
        else: # allow passing of starting value of r
            self.r = nn.Parameter(r)
        self.compositional=compositional
        
    def forward(self,x,x_mask):
        if not self.compositional:
            z = self.fun(x)
        else:
            x = x/torch.sum(x)
            f = self.fun(x)
            theta = torch.sum(x*f)
            z = f - theta
        # masking to prevent gradient updates for extinct species
        return (z*x_mask, self.A)
    
    def fun(self,x):
        if not self.compositional:
            return self.r+torch.matmul(self.A,x.T).T
        else:
            return torch.matmul(self.A,x.T).T

class ss_optim_loss(nn.Module):
    def __init__(self,alpha=1e-15):
            super().__init__()
            self.alpha = alpha
    def forward(self,ss_residual,A):
        det = torch.abs(torch.det(A)) # determinant term to enforce invertibility
        det_loss = torch.where(det < 0.1,1/det,0)
        diag = torch.diag(A) # diagonal to prevent self-influence from becoming positive
        diag_loss = 1e6*torch.norm(torch.where(diag > -0.1,diag+0.1,0),p=1)
        l1 = self.alpha * torch.norm(A,p=1) # l1 norm to enforce sparsity
        return (torch.norm(ss_residual,p=2) + diag_loss + det_loss + l1)

# %%
def reconstruct_from_ss(X,
                        compositional=True,
                        max_iter=10000, 
                        lr=1e-3, 
                        alpha=1e-15, 
                        batch_size=32, 
                        A_init=None, 
                        r_init=None, 
                        verbose=False,):
    '''
    The core reconstruction function of KeySDL. Finds the GLV or replicator model that best explains the observed steady states.

    Parameters
    ----------
    X : array-like of shape (steady states, features)
        Feature values (e.g.microbial abundances) of observed steady states.
    compositional: bool, default = True
        Whether to model as GLV (False) or replicator (True). Default value is True because most experimental datasets are inherently compositional.
    max_iter : int, default = 10000
        Number of gradient descent iterations used. This should rarely require adjustment.
    lr : float, default = 1e-3
        Gradient descent learning rate. This should rarely require adjustment.
    alpha : float, default = 1e-15
        L1 penalty. This should rarely require adjustment.
    batch_size : int, default = 32
        Gradient descent batch size. This should rarely require adjustment.
    A_init: array-like of shape (features,features), default = None
        Initial value for interactions matrix A, default value of None will initialize with the identity matrix.
    r_init: array-like of shape (features,), default = None
        Initial value for growth rates r, default value of None will initialize with ones

    Returns:
    --------
    A_pred: predicted interactions matrix A
    r_pred: predicted growth rates r
    
    '''
    X = torch.from_numpy(X).float()
    X_mask= torch.where(X==0,0,1).float() # mask to prevent gradient updates to extinct species
        
    model = ss_optim(n=X.shape[1],compositional=compositional,A=A_init,r=r_init)
    loss_fn = ss_optim_loss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()
        batch_idx = torch.randperm(X.shape[0])[:batch_size]
        pred,coef = model(X[batch_idx,:],X_mask[batch_idx,:])
        loss = loss_fn(pred,coef)
        loss.backward()
        optimizer.step()
        if verbose:
            print(f'Train Loss: {loss.item()}')

    A_pred = model.A.cpu().detach().numpy()
    r_pred = model.r.cpu().detach().numpy()
    return A_pred, r_pred
    
def self_consistency_score(data,A,r,compositional=True):
    sim = glv_simulator(A=A,r=r)
    pred_ss = np.zeros_like(data)
    for steady_state in range(data.shape[0]):
        pred_ss[steady_state,:] = sim.ss_from_assemblage(data[steady_state,:],compositional=compositional)
    bcd = np.sum(np.abs(pred_ss-data),axis=1)/np.sum(np.abs(pred_ss+data),axis=1)
    return 1-np.mean(bcd)
#%%

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('No input file provided. Please see file header for input format.')
        sys.exit()
    data_file = sys.argv[1]

    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]+'/KeySDL_out'
        print(f'Writing to: {out_dir}')
    else:
        out_dir = os.getcwd()+'/KeySDL_out'
        print(f'No output directory provided. Writing to: {out_dir}')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    if len(sys.argv) >= 4:
        compositional = bool(sys.argv[3])
    else:
        compositional = True
    
    if compositional:
        print('Modeling compositional replicator system.')
    else:
        print('Modeling absolute GLV system.')

    data = pd.read_csv(os.getcwd()+'/'+data_file)
    if compositional: # Ensure total sum scaling if modeling compositionally
        data = (data.T/data.sum(axis=1)).T

    # Reconstruct GLV/replicator system
    A,r = reconstruct_from_ss(X=data.values,compositional=compositional)
    # Create simulator object
    sim = glv_simulator(A,r)
    # Compute simulated impact on removal
    dropout_keystones = sim.bcd_keystones()

    # Print self-consistency score
    s_sc = self_consistency_score(data.values,A,r)
    print(f'Self-Consistency Score: {s_sc}')

    # Generate 500 random steady states
    perturbed = random_training_samples(A=A,r=r,n_train_samples=500)
    if compositional:
        perturbed = (perturbed.T/perturbed.sum(axis=1)).T
    # Compute simulated mean relative abundance
    simulated_abundance = perturbed.mean(axis=0)

    # Convert arrays to pandas DataFrame and Series for labeled export
    A = pd.DataFrame(A,index=data.columns,columns=data.columns)
    r = pd.Series(r,index=data.columns)
    dropout_keystones = pd.Series(dropout_keystones,index=data.columns)
    simulated_abundance = pd.Series(simulated_abundance,index=data.columns)

    # Write output files
    A.to_csv(out_dir+'/A.csv')
    r.to_csv(out_dir+'/r.csv')
    dropout_keystones.to_csv(out_dir+'/dropout_keystones.csv',header=False)
    simulated_abundance.to_csv(out_dir+'/simulated_abundance.csv',header=False)
