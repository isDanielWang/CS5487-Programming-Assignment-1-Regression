'''
@Auther: Shixiang WANG
@EID: sxwang6
@Date: 14/02/2022

@Discription:
CS5487 Programming Assignment1-Part1-Polynommial function-(a)
a:  Implement the above 5 regression algorithms for the K-th order polynomial given in (2). 
    In the next problem, you will use these regression methods with a diﬀerent feature transformation φ (x). 
    Hence, it would be better to separate the regression algorithm and the feature transformation in your implementation.
'''

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import random


def dataloader(sample =True, shuffle = False, ratio = 1):
    """load data

    Parameters
    ----------
    sample, optional
        if true, load sample data else load vaules for the true function, by default True
    shuffle, optional
        whether shuffle the loaded data, by default False
    ratio, optional
        the percentage of the loaded data, by default 1

    Returns
    -------
        input and output values
    """
    file_name = "polydata_data_samp" if sample else "polydata_data_poly"
    file_path = "PA-1-data-text/"
    x = np.loadtxt(file_path+file_name+"x.txt")
    y = np.loadtxt(file_path+file_name+"y.txt")
    xy = np.ndarray((len(x), 2))

    xy[:, 0] = x
    xy[:, 1] = y

    if shuffle:
        np.random.shuffle(xy)

    num_sample = int(ratio * len(x))
    x = xy[:num_sample, 0]
    y = xy[:num_sample, 1]

    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)
    return x, y


def transformer(feature, k):
    """transform the input feature to the K-th order polynomial feature matrix

    Parameters
    ----------
    feature
        the input value
    k
        the polynomial's order

    Returns
    -------
        phi
    """    
    phi = np.ndarray(shape = (k+1, len(feature)), dtype=np.float64)
    for i in range(k+1):
        for j in range(len(feature)):
             phi[i][j] = feature[j]**i
    return phi

def LS_train(phi, y):
    """train process for least squares

    Parameters
    ----------
    phi
        polynomial feature
    y
        observations

    Returns
    -------
        parameter estimate theta
    """    
    phi_t = np.transpose(phi)
    theta = np.linalg.inv(phi @ phi_t) @ phi @ y
    return theta

def LS_prediction(new_x, theta):
    """make prediction using least-squares

    Parameters
    ----------
    new_x
        the test new input value
    theta
        parameter estimate theta

    Returns
    -------
        prediction value
    """    
    prediction = np.transpose(new_x) @ theta
    return prediction

def RLS_train(phi, y, lamb = 0.1):
    """trian process for regularied least squares

    Parameters
    ----------
    phi
        input polynomial feature
    y
        observation
    lamb
        regualrization hyperparameter, large lamb means large punishing. by default 0.1

    Returns
    -------
        parameter estimate theta
    """    
    phi_t = np.transpose(phi)
    n = phi.shape[0]
    identity = np.identity(n)
    theta = np.linalg.inv(phi @ phi_t + lamb * identity) @ phi @ y
    return theta

def RLS_prediction(new_x, theta):
    """make prediction using regularized squares

    Parameters
    ----------
    new_x
        new test input value
    theta
        parameter estimate theta

    Returns
    -------
        prediction
    """    
    prediction  = np.transpose(new_x) @ theta
    return prediction

def LASSO_train(phi, y, lamb = 1.0):
    """train process of LASSO.
    Using standard QP solver.
    Objective: min (1/2)*x'*H*x + f'*x s.t. x >= 0
    Where H = [[phi@phi_t, -1*phi@phi_t], [-1*phi@phi_t, phi@phi_t]], f = lamd*ones - [phi@y, -phi@y], x = [theta_+, theta_-], theta = theta_+ - theta_-

    Parameters
    ----------
    phi
        input polynomial feature
    y
        observation
    lamb
        regularization hyperparameter

    Returns
    -------
        theta
    """
    phi_phi_t = phi @ np.transpose(phi)
    H_up = np.concatenate((phi_phi_t, -1*phi_phi_t), axis = 1)
    H_down = np.concatenate((-1*phi_phi_t, phi_phi_t), axis = 1)
    H = np.concatenate((H_up, H_down), axis = 0)
    phi_y = phi @ y
    tmp = np.concatenate((phi_y, -1*phi_y), axis = 0)
    f  = [[1.0 * lamb] for i in range(len(tmp))] -tmp
    # Solves a quadratic program
    # minimize    (1/2)*x'*P*x + q'*x
    # subject to  G*x <= h
    #             A*x = b.
    G = -1 * np.identity((len(H)))
    h = np.zeros((len(H), 1))
    theta_tmp  = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h))['x']
    theta = np.matrix([theta_tmp[i] - theta_tmp[i + phi.shape[0]] for i in range(int(len(theta_tmp) / 2))]).transpose()
    return theta

def LASSO_prediction(new_x, theta):
    """make prediction using LASSO

    Parameters
    ----------
    new_x
        the new input testing values
    theta
        parameter estimate theta

    Returns
    -------
        prediction
    """    
    prediction = np.transpose(new_x) @ theta
    return prediction

def RR_train(phi, y):
    """train process of robust regression.
    Using standard lp solver。
    Objective: min f'x s.t. Ax <= b 
    Where f = [0, 1], A =[[-phi', -I], [phi', -I]], b = [-y, y], x = [theta, t]

    Parameters
    ----------
    phi
        input polynomial feature
    y
        observation

    Returns
    -------
        theta
    """ 
    n = phi.shape[1]
    tmp = np.identity(n)
    A_up = np.concatenate((-1*np.transpose(phi), -1*tmp), axis = 1)
    A_down = np.concatenate((np.transpose(phi), -1*tmp), axis = 1)
    A = np.concatenate((A_up, A_down), axis = 0)
    f = np.concatenate((np.zeros((phi.shape[0], 1)), np.ones((n, 1))), axis=0)
    # Solves a pair of primal and dual LPs

    #     minimize    c'*x
    #     subject to  G*x + s = h
    #                 A*x = b
    #                 s >= 0

    #     maximize    -h'*z - b'*y
    #     subject to  G'*z + A'*y + c = 0
    #                 z >= 0.
    b = np.concatenate((-1 * y, y), axis=0)
    theta = solvers.lp(matrix(f), matrix(A), matrix(b))['x'][0:phi.shape[0]]
    return theta

def RR_prediction(x_new, theta):
    """make prediction of Robust regression

    Parameters
    ----------
    x_new
        new input test value
    theta
        estimate parameter theta

    Returns
    -------
        prediction
    """    
    prediction = np.transpose(x_new) @ theta
    return prediction

def BR_train(phi, y, alpha):
    """train process of bayesian regression
    mu = 1/sigmoid^2 * var * phi * y, and var = invert(1/alpha + 1/sigmoid^2 * phi * phi'), sigmoid^2 = 5

    Parameters
    ----------
    phi
        input polynomial feature
    y
        observations
    alpha
        hyperparameter

    Returns
    -------
        estimate mu and var
    """ 
    v = 1 / 5
    n = phi.shape[0]
    var = np.linalg.inv((1 / alpha) * np.identity(n) + v * phi @ np.transpose(phi))
    mu = v * var @ phi @ y
    return mu, var

def BR_prediction(x_new, mu, var):
    """make prediction using Bayesian regression

    Parameters
    ----------
    x_new
        new input polynomial feature
    mu
        parameter estimate mu
    var
        parameter estimate variance

    Returns
    -------
        prediction, variance
    """    
    prediction = np.transpose(x_new) @ mu
    variance = np.transpose(x_new) @ var @ x_new
    return prediction, variance

def MSE(observation, prediction):
    """Calculate the mean square error

    Parameters
    ----------
    observation
        real output
    prediction
        predicted output

    Returns
    -------
        mean square error
    """
    error = np.mean(np.square(observation -prediction))
    return error

def MAE(observation, prediction):
    """Calculate the mean absolute error

    Parameters
    ----------
    observation
        real output
    prediction
        predicted output

    Returns
    -------
        mean absolute error
    """
    error = np.mean(np.absolute(observation - prediction))
    return error

def add_outliers(simpy, ratio = 0.2):
    """Add some outliers output values (e.g. add large number to a few values in sampy)

    Parameters
    ----------
    simpy
        sample output values (y i), each entry is an output.
    ratio
        number of modeified sampy / number of sampy

    Returns
    -------
        samply with ratio*len(samply) of outliers
    """    
    num_sampy = len(simpy)
    num_outliers = int(ratio * num_sampy)
    index = random.sample(range(num_sampy), num_outliers)
    sampy_with_outliers = simpy.copy()
    for i in index:
        noise = random.randint(10, 30)
        sampy_with_outliers[i] += noise
    return sampy_with_outliers