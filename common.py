import numpy as np



def sigmoid(x):
    output = np.zeros(x.shape)
    ind1 = (x >= 0)
    ind2 = (x  < 0)
    output[ind1] = 1 / (1 + np.exp(-x[ind1]))
    output[ind2] = np.divide(np.exp(x[ind2]), (1 + np.exp(x[ind2])))

    return output

def log_likelihood(phi, pred, t, dot_product, weight, reg= 1):
    prior = -0.5* np.sum(np.power(weight, 2))
    likelihood = np.multiply(t[0], np.log(pred+1e-5)) + np.multiply(1.0- t[0], np.log(1.0-pred+1e-5))
    likelihood = np.sum(likelihood, axis= 0)

    return prior + likelihood


def hessian(phi, pred, t, dot_product, reg= 1, regression= "logistic"):
    R = np.eye(pred.shape[0])
    if regression == "logistic":
        for i in range(pred.shape[0]):
            R[i,i] = pred[i,0] * (1- pred[i,0])
    elif regression == "probit":
        for i in range(pred.shape[0]):
            y_n  = pred[i,0]
            t_n  = t[i,0] 
            dotp = dot_product[i, 0]
            pdf  = norm.pdf(dotp)

            term1 = 1/ (y_n * (1- y_n) + 1e-5)
            term2 = (y_n - t_n)/(y_n**2 * (1- y_n) + 1e-5)
            term3 = (y_n - t_n)/((1- y_n)**2 * y_n + 1e-5)
            term4 = (y_n - t_n)* dotp/(y_n * (1- y_n) * pdf + 1e-5)

            R[i,i] = (term1 - term2 + term3 - term4)*(pdf**2)

    # Add regularization            
    hessian = np.matmul(np.matmul(phi.T, R), phi) + np.eye(phi.shape[1])/reg
    return hessian

def gass_hermite_quad(f, degree, m, b):
    '''
    Calculate the integral (1) numerically.
    :param f: target function, takes a array as input x = [x0, x1,...,xn], and return a array of function values f(x) = [f(x0),f(x1), ..., f(xn)]
    :param degree: integer, >=1, number of points
    :return:
    '''

    points, weights = np.polynomial.hermite.hermgauss( degree)

    #function values at given points
    f_x = f(points, m= m, b= b)

    #weighted sum of function values
    F = np.sum( f_x  * weights)

    return F