import numpy as np
import utils as u


def IHT_inpainting(Y,clipped_samples_mat,D,alg_param):
    """
    Perform declipping using Iterative Hard Thresholding for inpainting
    (discards the clipped samples)
    Inputs:        
        - Y: array of size NxT containing T clipped signals of size N
        - clipped_samples_mat: boolean array describing the clipped indices of Y
        - D: fixed dictionary
        - alg_param["Nit"]: number of iterations
        - alg_param["A_init"]: initial sparse matrix
        - alg_param["loud"]: 1 to print the objective at each iteration, 0 otherwise
         
    Outputs:
        - A: sparse activation matrix
        - cost: vector containing the value of the cost at each iteration
    """
     
    
    # Initialize parameters
    
    if 'A_init' not in alg_param:
        alg_param["A_init"] = np.zeros((D.shape[1],Y.shape[1]))
    
    if 'loud' not in alg_param:
        alg_param["loud"] = 0
    
    
    mu = 1/np.linalg.norm(D,2)**2  # gradient descent parameter
    
    cost = np.nan*np.empty(alg_param["Nit"]+1) # save cost at each iteration 
    
    # Declip:
    
    # initialize sparse coefficient matrix:
    A = alg_param["A_init"]
    
    # compute residual:
    ResidualMat = Y-D@A
    ResidualMat[clipped_samples_mat] = 0  # ignore clipped samples
    
    cost[0] = np.sum(ResidualMat**2)
    
    if alg_param["loud"]:
        print('initial cost: %.3f' % cost[0])
    
    it = 0;
    
    while it < alg_param["Nit"]:
        it = it+1;
    
        # gradient descent step:
        A = A + mu * D.T@ResidualMat
        
        # hard thresholding:
        A = u.hard_threshold(A, alg_param["K"])
        
        # compute residual:
        ResidualMat = Y-D@A
        ResidualMat[clipped_samples_mat] = 0  # ignore clipped samples
        
        # compute cost:
        cost[it] = np.sum(ResidualMat**2)
        
        if alg_param["loud"]:
            print('it = {0}, cost: {1:.3f}'.format(it, cost[it]))
        
    

    return A, cost
    
    
    
def DictionaryLearning_inpainting(Y,clipped_samples_mat,paramDL):
    """
    Perform declipping using gradient-descent based dictionary learning for inpainting
    (discards clipped samples)
    
    Inputs:
        - Y: array of size NxT containing T clipped signals of size N
        - clipped_samples_mat: boolean array describing the clipped indices of Y
        - paramDL["K"]: number of non-zero atoms 
        - paramDL["Nit"]: number of dictionary learning iterations
        - paramDL["Nit_sparse_coding"]: number of iterations sparse coding step
        - paramDL["Nit_dict_update"]: number of iterations dictionary update step
        - paramDL["warm_start"]: 1 to perform warm start at each iteration
        - paramDL["A_init"]: initial sparse coefficient matrix
        - paramDL["D_init"]: initial dictionary
        - paramDL["loud"]: 1 to print results
            
    Outputs:
        - D: estimated dictionary
        - A: sparse activation matrix
        - cost: vector containing the value of the cost at each iteration
      """
    
    # Initialize parameters:
    
    if "warm_start" not in paramDL:
        paramDL.warm_start = 1
    
    
    if "loud" not in paramDL:
        paramDL.loud = 0

    
    A = paramDL["A_init"]
    D = paramDL["D_init"]
    
    cost = np.nan*np.empty(2*paramDL["Nit"]+1)  # save cost at each iteration 
    
    # compute residual:
    ResidualMat = Y-D@A
    ResidualMat[clipped_samples_mat] = 0 # ignore clipped samples
    
    cost[0] = np.sum(ResidualMat**2)
    
    if paramDL["loud"]:
        print('initial cost: %.3f' % cost[0])
    
    
    # DL iterations
    
    it = 0
    
    while it < paramDL["Nit"]:
        it += 1
    
        # Sparse coding:
        
        # parameters for sparse coding step:
        paramSC = {}
        paramSC["K"] = paramDL["K"]
        paramSC["Nit"] = paramDL["Nit_sparse_coding"]
        
        if paramDL["warm_start"]:
            paramSC["A_init"] = A # warm_start
        else:
            paramSC["A_init"] = np.zeros_like(A)
        
        
        A, cost_SC = IHT_inpainting(Y,clipped_samples_mat,D,paramSC)
        
        cost[2*it-1] = cost_SC[-1]
        
        if paramDL["loud"]:
            print("it {0}, sparse coding step: cost: {1:.3f}".format(it, cost[2*it-1]))
        
        
        # Prune unused atoms:
        unused = np.where(np.sum(A**2,1) == 0)[0]
        
        if unused.size == 1 and paramDL["loud"]:
            print('  %d atom pruned' % unused.size)
        elif unused.size>1 and paramDL["loud"]:
            print('  %d atoms pruned' % unused.size)
            
        D = np.delete(D,unused,1)
        A = np.delete(A,unused,0)
        
        # Dictionary Update:
        
        ResidualMat = Y-D@A
        ResidualMat[clipped_samples_mat] = 0  # ignore clipped samples
    
        mu = 1/np.linalg.norm(A,2)**2  # gradient descent parameter
    
        # Gradient descent:
    
        for j in range(paramDL["Nit_dict_update"]):
        
            # gradient descent:
            D = D + mu * ResidualMat@(A.T)
            
            # normalize atoms:
            D = u.normalize_dictionary(D)
        
            # update residual:
            ResidualMat = Y-D@A
            ResidualMat[clipped_samples_mat] = 0  # ignore clipped samples
            
            
        cost[2*it] = np.sum(ResidualMat**2)
        
        if paramDL["loud"]:
            print('it {0},   dict update step: cost: {1:.3f}'.format(it, cost[2*it]))
        
        
    return D, A, cost
          
       


def consistentIHT(Y,clipped_samples_mat,D,alg_param):
    """
    Perform declipping using consistent Iterative Hard Thresholding [1]
    (enforces the clipped samples to be above the clipping level)
    Inputs:        
        - Y: array of size NxT containing T clipped signals of size N
        - clipped_samples_mat: boolean array describing the clipped indices of Y
        - D: fixed dictionary
        - alg_param["Nit"]: number of iterations
        - alg_param["A_init"]: initial sparse matrix
        - alg_param["loud"]: 1 to print the objective at each iteration, 0 otherwise
         
    Outputs:
        - A: sparse activation matrix
        - cost: vector containing the value of the cost at each iteration

    Reference:        
    [1] : Consistent iterative hard thresholding for signal declipping, Kitic et al, ICASSP 2013         
        
    """
     
    
    # Initialize parameters
    
    if 'A_init' not in alg_param:
        alg_param["A_init"] = np.zeros((D.shape[1],Y.shape[1]))
    
    if 'loud' not in alg_param:
        alg_param["loud"] = 0
    
    
    mu = 1/np.linalg.norm(D,2)**2  # gradient descent parameter
    
    clipped_pos_mat = np.logical_and(clipped_samples_mat, Y>=0) # positive clipped samples
    clipped_neg_mat = np.logical_and(clipped_samples_mat, Y<=0) # negative clipped samples

    cost = np.nan*np.empty(alg_param["Nit"]+1) # save cost at each iteration 
    
    # Declip:
    
    # initialize sparse coefficient matrix:
    A = alg_param["A_init"]
    
    # compute residual:
    ResidualMat = Y-D@A
    # enforce clipping consistency:
    ResidualMat[clipped_pos_mat] = np.maximum(ResidualMat[clipped_pos_mat],0)
    ResidualMat[clipped_neg_mat] = np.minimum(ResidualMat[clipped_neg_mat],0)
    
    cost[0] = np.sum(ResidualMat**2)
    
    if alg_param["loud"]:
        print('initial cost: %.3f' % cost[0])
    
    it = 0;
    
    while it < alg_param["Nit"]:
        it = it+1;
    
        # gradient descent step:
        A = A + mu * D.T@ResidualMat
        
        # hard thresholding:
        A = u.hard_threshold(A, alg_param["K"])
        
        # compute residual:
        ResidualMat = Y-D@A
        # enforce clipping consistency:
        ResidualMat[clipped_pos_mat] = np.maximum(ResidualMat[clipped_pos_mat],0)
        ResidualMat[clipped_neg_mat] = np.minimum(ResidualMat[clipped_neg_mat],0)
        
        # compute cost:
        cost[it] = np.sum(ResidualMat**2)
        
        if alg_param["loud"]:
            print('it = {0}, cost: {1:.3f}'.format(it, cost[it]))
        
    

    return A, cost
    
    
    
def consistentDictionaryLearning(Y,clipped_samples_mat,paramDL):
    """
    Perform declipping using consistent dictionary learning [2]
    (enforces the clipped samples to be above the clipping level)
    
    Inputs:
        - Y: array of size NxT containing T clipped signals of size N
        - clipped_samples_mat: boolean array describing the clipped indices of Y
        - paramDL["K"]: number of non-zero atoms 
        - paramDL["Nit"]: number of dictionary learning iterations
        - paramDL["Nit_sparse_coding"]: number of iterations sparse coding step
        - paramDL["Nit_dict_update"]: number of iterations dictionary update step
        - paramDL["warm_start"]: 1 to perform warm start at each iteration
        - paramDL["A_init"]: initial sparse coefficient matrix
        - paramDL["D_init"]: initial dictionary
        - paramDL["loud"]: 1 to print results
            
    Outputs:
        - D: estimated dictionary
        - A: sparse activation matrix
        - cost: vector containing the value of the cost at each iteration
        
    [2] Reference: Consistent dictionary learning for signal declipping, Rencker, Bach, Wang and Plumbley,
            Latent Variable Analysis and Signal Separation (LVA/ICA), Guildford, UK, 2018
      """
    
    # Initialize parameters:
    
    if "warm_start" not in paramDL:
        paramDL.warm_start = 1
    
    
    if "loud" not in paramDL:
        paramDL.loud = 0

    
    A = paramDL["A_init"]
    D = paramDL["D_init"]
    
    cost = np.nan*np.empty(2*paramDL["Nit"]+1)  # save cost at each iteration 
    
    clipped_pos_mat = np.logical_and(clipped_samples_mat, Y>=0) # positive clipped samples
    clipped_neg_mat = np.logical_and(clipped_samples_mat, Y<=0) # negative clipped samples
    
    # compute residual:
    ResidualMat = Y-D@A
    # enforce clipping consistency:
    ResidualMat[clipped_pos_mat] = np.maximum(ResidualMat[clipped_pos_mat],0)
    ResidualMat[clipped_neg_mat] = np.minimum(ResidualMat[clipped_neg_mat],0)
    
    cost[0] = np.sum(ResidualMat**2)
    
    if paramDL["loud"]:
        print('initial cost: %.3f' % cost[0])
    
    
    # DL iterations
    
    it = 0
    
    while it < paramDL["Nit"]:
        it += 1
    
        # Sparse coding:
        
        # parameters for sparse coding step:
        paramSC = {}
        paramSC["K"] = paramDL["K"]
        paramSC["Nit"] = paramDL["Nit_sparse_coding"]
        
        if paramDL["warm_start"]:
            paramSC["A_init"] = A # warm_start
        else:
            paramSC["A_init"] = np.zeros_like(A)
        
        
        A, cost_SC = consistentIHT(Y,clipped_samples_mat,D,paramSC)
        
        cost[2*it-1] = cost_SC[-1]
        
        if paramDL["loud"]:
            print("it {0}, sparse coding step: cost: {1:.3f}".format(it, cost[2*it-1]))
        
        
        # Prune unused atoms:
        unused = np.where(np.sum(A**2,1) == 0)[0]
        
        if unused.size == 1 and paramDL["loud"]:
            print('  %d atom pruned' % unused.size)
        elif unused.size>1 and paramDL["loud"]:
            print('  %d atoms pruned' % unused.size)
            
        D = np.delete(D,unused,1)
        A = np.delete(A,unused,0)
        
        # Dictionary Update:
        
        ResidualMat = Y-D@A
        ResidualMat[clipped_samples_mat] = 0  # ignore clipped samples
    
        mu = 1/np.linalg.norm(A,2)**2  # gradient descent parameter
    
        # Gradient descent:
    
        for j in range(paramDL["Nit_dict_update"]):
        
            # gradient descent:
            D = D + mu * ResidualMat@(A.T)
            
            # normalize atoms:
            D = u.normalize_dictionary(D)
        
            # compute residual:
            ResidualMat = Y-D@A
            # enforce clipping consistency:
            ResidualMat[clipped_pos_mat] = np.maximum(ResidualMat[clipped_pos_mat],0)
            ResidualMat[clipped_neg_mat] = np.minimum(ResidualMat[clipped_neg_mat],0)
            
            
        cost[2*it] = np.sum(ResidualMat**2)
        
        if paramDL["loud"]:
            print('it {0},   dict update step: cost: {1:.3f}'.format(it, cost[2*it]))
        
        
    return D, A, cost
          
       

    
    


    
    

