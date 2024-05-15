import torch
import numpy as np
from tqdm.notebook import tqdm

# Set of functions that implement NObSP in SVM and NN models.
#
# @Copyrigth:  Alexander Caicedo, April 2023

def ObSP(X,Y):

    # Function to compute the oblique projection onto X along the reference space defined by Y.
    # The input data is the following:
    # X: a matriz of size Nxd, containing the basis for the subspace where the data will be projected
    # Y: a matrix of size Nxd, containing the basis for the reference subspace
    # 
    # The function returns the oblique projection matrix of size NxN.

    # Converting the input data into torch tensors

    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    
    if not torch.is_tensor(Y):
        Y = torch.from_numpy(Y)
    
    N, d = X.size() # computing the size of X
        
    P = Y@torch.linalg.pinv(torch.t(Y)@Y)@torch.t(Y) # Conputing the orthogonal projection matriz onto the subsapce given by Y    
    Q = torch.eye(N,N)-P # Computing the complement of P
    
    P_xy = X@torch.linalg.pinv(torch.t(X)@Q@X)@torch.t(X)@torch.t(Q) # Computing the oblique projection matriz onto X along Y
    
    return P_xy


def obsp_regression(X, Z, y):
    
    # Oblique projection using regression approach.
 
    # Parameters:
    # X (numpy.ndarray): Matrix representing the basis for the target subspace.
    # Z (numpy.ndarray): Matrix representing the basis for the reference subspace.
    # y (numpy.ndarray): Vector to be projected.
 
    # Returns:
    # beta (numpy.ndarray): Coefficients of the projection in the regression problem.
    
    
    # Compute the matrix Q
    P = Z@torch.linalg.pinv(torch.t(Z)@Z)@torch.t(Z) # Computing the orthogonal projection matriz onto the subsapce given by Y    
    Q = torch.eye(Z.shape[0])-P # Computing the complement of P
    
    # Compute the matrix (X^T Q X)^{-1} X^T Q
 
    aux = torch.linalg.pinv(torch.t(X)@Q@X)@torch.t(X)@Q
    
    # Compute the beta coefficients
    beta = aux @ y
    
    return beta

    
def NObSP_NN_single_MultiOutput(X, y_est, model):
    
    # Function to decompose the output of a NN regression model using oblique subspace projections. The function computes 
    # appropriate evalautions of the netwrok that define the subspace of the nonlinear transformation of the input variables. 
    # These subspaces lie in the same space where the output data is located. This function uses as input the following variables:
    # 
    # X: a matrix of size Nxd, contining the input variables
    # y_est: a matrix of size Nxp, containing the estimated outputs for the input data X
    # model: the NN model using pytorch
    # 
    # The function returns d oblique projection matrices of size NxN, the estimated contribution of each input variable on the output,
    # and the alpha coefificents for the out-of-sample extension
    
    model.eval() # Setting the model in evaluation mode
    N = np.size(X,0) # computing the size of X along dimension 0
    d = np.size(X,1) # computing the size of X along dimension 1
    p = list(model.children())[-1].out_features # computing the size of X along dimension 1
    P_xy = np.zeros((N,N,d)) # Initializing proyection matrices
    y_e = torch.from_numpy(np.zeros((N,d,p))).type(torch.float) # Initializing Matriz where the estimated nonlinear contributions will be stored, converting to a tensor object
    neurons_last = list(model.children())[-1].in_features # Obtaining the dimension of the subspace where the data lies (number of neurons in the last layer)
    Alpha = torch.zeros(neurons_last,d*p) # Initializing the matrix for the Alpha coefficients, out-of-sample extension, using the dimension where the transformed data lies
    
    # Computing the transformation of the input data using inference mode, and extracting the evaluation of the input on the network up to the last layer.
    with torch.inference_mode():
        y_target, X_target_tot, y_lin = model(X)
    
    #X_target_tot = X_target_tot-torch.mean(X_target_tot,dim=0) # Centaring the data
    P_x_target = torch.linalg.pinv(torch.t(X_target_tot)@X_target_tot)@torch.t(X_target_tot) # Finding the projection matrix onto the matrix tused to find the alpha coeficients, out-of-sample extension 
    
    for l in range(p):
        for i in range(d):
        
            # Defining the input matrix that will be used to find the subspace of the nonlinear transformation of 
            # the input variables x_i, onto which the output will be projected
            X_target = np.zeros((N,d))
            X_target[:,i] = X[:,i]
        
            # Defining the input matrix that will be used to find the reference subspace, along which the data 
            # will be projected.
            X_reference = np.copy(X)
            X_reference[:,i] = 0
        
            # transforming the matrices to tensor objects to be used in pytorch
            X_target = torch.from_numpy(X_target).type(torch.float)
            X_reference = torch.from_numpy(X_reference).type(torch.float)
        
            # Computing the transformation of the input data using inference mode in order to find the basis for the susbpace of the transformations
            with torch.inference_mode():
                y_target, X_target_sub, y_lin = model(X_target) # X_target_sub is a basis for the nonlienar transformation of the data in X_target
                y_reference, X_reference_sub, y_lin = model(X_reference) # X_reference_sub is a basis for the nonlienar transformation of the data in X_reference
        
            # Centering the bassis of the target and reference subspaces
            X_target_sub = (X_target_sub-torch.mean(X_target_sub,dim=0)) 
            X_reference_sub = (X_reference_sub-torch.mean(X_reference_sub,dim=0))
            
            # Computing the oblique projection onto the susbspace defined by the nonlienar transformation of x_i along 
            # the reference subspace, which contains the nonlinear transofrmation of all variables except x_i
            P_xy[:,:,i] = ObSP(X_target_sub,X_reference_sub)

            P = torch.from_numpy(P_xy[:,:,i]).type(torch.float) # Converting form pytorch to numpy
            y_e[:,[i],[l]] = torch.unsqueeze(P@(y_est[:,l]-torch.mean(y_est[:,l])),-1) # Using the projection matrices to ptoject the output vector and find the nonlinear contribution of each variable.
        
        Alpha[:,l*d:l*d+d] = torch.linalg.lstsq(X_target_tot, y_e[:,:,l].squeeze(), rcond=None, driver='gelsd')[0]
        #print(f'initial column {l*d} final column {l*d+d}')
    
    #print(f'Size Alpha {Alpha.size()}')
    return P_xy, y_e, Alpha

def NObSP_NN_single_MultiOutput_reg(X, y_est, model):
    
    # Function to decompose the output of a NN regression model using oblique subspace projections. The function computes 
    # appropriate evalautions of the netwrok that define the subspace of the nonlinear transformation of the input variables. 
    # These subspaces lie in the same space where the output data is located. This function uses as input the following variables:
    # 
    # X: a matrix of size Nxd, contining the input variables
    # y_est: a matrix of size Nxp, containing the estimated outputs for the input data X
    # model: the NN model using pytorch
    # 
    # The function returns d oblique projection matrices of size NxN, the estimated contribution of each input variable on the output,
    # and the alpha coefificents for the out-of-sample extension
    
    model.eval() # Setting the model in evaluation mode
    N = np.size(X,0) # computing the size of X along dimension 0
    d = np.size(X,1) # computing the size of X along dimension 1
    p = list(model.children())[-1].out_features # computing the size of X along dimension 1
    #beta_tensor = torch.from_numpy(np.zeros((N,d,p))).type(torch.float) # Initializing Matriz where the estimated nonlinear contributions will be stored, converting to a tensor object
    neurons_last = list(model.children())[-1].in_features # Obtaining the dimension of the subspace where the data lies (number of neurons in the last layer)
    betas_tensor = torch.zeros((neurons_last, d, p))
    Alpha = torch.zeros(neurons_last,d*p) # Initializing the matrix for the Alpha coefficients, out-of-sample extension, using the dimension where the transformed data lies
    
    # Computing the transformation of the input data using inference mode, and extracting the evaluation of the input on the network up to the last layer.
    with torch.inference_mode():
        y_target, X_target_tot, y_lin = model(X)
    
    #X_target_tot = X_target_tot-torch.mean(X_target_tot,dim=0) # Centaring the data
    P_x_target = torch.linalg.pinv(torch.t(X_target_tot)@X_target_tot)@torch.t(X_target_tot) # Finding the projection matrix onto the matrix tused to find the alpha coeficients, out-of-sample extension 
    
    for l in tqdm(range(p), desc="Number of tensors"):
        for i in tqdm(range(d), desc="Lenght of tensors"):
        
            # Defining the input matrix that will be used to find the subspace of the nonlinear transformation of 
            # the input variables x_i, onto which the output will be projected
            X_target = np.zeros((N,d))
            X_target[:,i] = X[:,i]
        
            # Defining the input matrix that will be used to find the reference subspace, along which the data 
            # will be projected.
            X_reference = np.copy(X)
            X_reference[:,i] = 0
        
            # transforming the matrices to tensor objects to be used in pytorch
            X_target = torch.from_numpy(X_target).type(torch.float)
            X_reference = torch.from_numpy(X_reference).type(torch.float)
                    
            # Computing the transformation of the input data using inference mode in order to find the basis for the susbpace of the transformations
            with torch.inference_mode():
                y_target, X_target_sub, y_lin = model(X_target) # X_target_sub is a basis for the nonlienar transformation of the data in X_target
                y_reference, X_reference_sub, y_lin = model(X_reference) # X_reference_sub is a basis for the nonlienar transformation of the data in X_reference
        
            # Centering the bassis of the target and reference subspaces
            X_target_sub = (X_target_sub-torch.mean(X_target_sub,dim=0))
            X_reference_sub = (X_reference_sub-torch.mean(X_reference_sub,dim=0))
            
            # Calcula beta para la variable actual 'i' y la salida 'l'
            beta = obsp_regression(X_target_sub, X_reference_sub, y_est[:,l]-torch.mean(y_est[:,l]))

            # Guarda el tensor beta en la posición correcta dentro del tensor pre-alojado
            betas_tensor[:, i, l] = beta
    
    #print(f'Size Alpha {Alpha.size()}')
    return betas_tensor

def to_categorical(y, num_classes):
    # Converting to categorical the ouput of the model
    return np.eye(num_classes)[y]
