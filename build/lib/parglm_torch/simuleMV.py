import torch
import numpy as np

def Floc(x, xdata):
    # x is an array of constants, xdata is an array of shape (n,2)
    y = np.zeros(xdata.shape[0])
    for i in range(xdata.shape[0]):
        x1 = xdata[i,0]
        x2 = xdata[i,1]
        if x1 <= 3:
            y[i] = x[0]*(x[1]-x1)*((np.log(x2)/np.log(x[2]))**(x[3]*np.exp(-x[4]*x1)))
        elif x1 <= 7:
            y[i] = x[5]*(x[6]-x1)*((np.log(x2)/np.log(x[7]))**(x[8]*np.exp(-x[9]*x1)))
        else:
            y[i] = x[10]*(x[11]-x1)*((np.log(x2)/np.log(x[12]))**(x[13]*np.exp(-x[14]*x1)))
    return y

def ADICOV(covariance_matrix, data_matrix):
    # covariance_matrix: [vars x vars] tensor
    # data_matrix: [n_samples x vars] tensor
    try:
        L = torch.linalg.cholesky(covariance_matrix)
    except RuntimeError as e:
        # If the covariance matrix is not positive definite, add small noise
        eps = 1e-6
        covariance_matrix += eps * torch.eye(covariance_matrix.shape[0])
        L = torch.linalg.cholesky(covariance_matrix)
    X = data_matrix @ L.T
    return X

def simuleMV(obs, vars, LevelCorr=5, Covar=None):
    import torch
    import numpy as np

    # Check arguments
    assert isinstance(obs, int) and obs > 0, "Parameter 'obs' must be a positive integer."
    assert isinstance(vars, int) and vars > 0, "Parameter 'vars' must be a positive integer."
    assert LevelCorr >= 0 and LevelCorr <= 10, "Parameter 'LevelCorr' must be between 0 and 10."
    if Covar is not None:
        Covar = torch.tensor(Covar, dtype=torch.float32)
        assert Covar.shape == (vars, vars), "Parameter 'Covar' must be a [vars x vars] matrix."
    else:
        Covar = None

    # Main code
    if Covar is None:
        uselevel = True
        corM = torch.eye(vars)
    else:
        uselevel = False
        corM = Covar

    if uselevel:
        if LevelCorr > 0:
            x = np.array([0.2837, 17.9998, 19.3749, 2.0605, 0.3234, 
                          0.4552, 12.0737, 16.6831, 5.2423, 0.5610,
                          0.3000, 14.7440, 4.4637e+04, 7.1838, 0.8429])

            xdata = np.array([[LevelCorr, vars]])
            obs2 = np.round(Floc(x, xdata)[0]**2)
            obs2 = max(2, obs2)
            if obs < obs2:
                print('Warning: correlation level too low. Resulting matrix may show a higher correlation due to structural constraints.')

            # Generate initial data
            data_matrix = torch.randn(int(obs2), vars)
            X = ADICOV(torch.eye(vars), data_matrix)
            # Preprocess X
            Xs = X - torch.mean(X, dim=0)
            # Compute covariance
            COV = (Xs.T @ Xs) / (X.shape[0] -1)
            # Adjust covariance matrix
            corM = COV + 0.01 * torch.eye(vars)
        else:
            if obs < vars:
                print('Warning: correlation level too low. Resulting matrix may show a higher correlation due to structural constraints.')

    # Generate final data
    data_matrix = torch.randn(obs, vars)
    X = ADICOV(corM, data_matrix)
    return X


