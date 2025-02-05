import numpy as np
from numba import jit, prange

"""
Rossler System with Higher-Order Interactions

This module implements numerical integration of a Rössler system
with two- and three-body interactions

"""

@jit(nopython=True)
def rossler_fun(N, list_neighbors, triangles_list, a, b, c, s1, s2, h, numsteps):
    """
    Simulates the Rossler system on a network with pairwise and three-body interactions.
    """
    X, Y, Z = np.random.random((3, N))
    X_hist, Y_hist, Z_hist = np.zeros((3, N, numsteps + 1))
    theta = np.zeros((N, numsteps + 1))
    
    X_hist[:, 0], Y_hist[:, 0], Z_hist[:, 0] = X, Y, Z
    theta[:, 0] = np.arctan2(Y, X)
    
    for t in range(numsteps):
        for i in range(N):
            first_order = sum(X_hist[j, t] - X_hist[i, t] for j in list_neighbors[i])
            second_order = sum(
                ((X_hist[j, t] ** 2) * X_hist[k, t] - X_hist[i, t] ** 3) +
                ((X_hist[k, t] ** 2) * X_hist[j, t] - X_hist[i, t] ** 3)
                for n, j, k in triangles_list if i in (n, j, k)
            )
            
            X_hist[i, t + 1] = X_hist[i, t] + h * (-Y_hist[i, t] - Z_hist[i, t] + s1 * first_order + s2 * second_order)
            Y_hist[i, t + 1] = Y_hist[i, t] + h * (X_hist[i, t] + a * Y_hist[i, t])
            Z_hist[i, t + 1] = Z_hist[i, t] + h * (b + Z_hist[i, t] * (X_hist[i, t] - c))
            theta[i, t + 1] = np.arctan2(Y_hist[i, t + 1], X_hist[i, t + 1])
    
    return [X_hist, Y_hist, Z_hist, theta]

@jit(nopython=True)
def rossler_fun_natural_coupling_type2(N, list_neighbors, triangles_list, a, b, c, s1, s2, h, numsteps):
    """
    Simulates the type II Rossler system on a network with pairwise and three-body interactions
    """
    X, Y, Z = np.random.random((3, N))
    X_hist, Y_hist, Z_hist = np.zeros((3, N, numsteps + 1))
    theta = np.zeros((N, numsteps + 1))
    
    X_hist[:, 0], Y_hist[:, 0], Z_hist[:, 0] = X, Y, Z
    theta[:, 0] = np.arctan2(Y, X)
    
    for t in range(numsteps):
        for i in range(N):
            first_order = sum(Y_hist[j, t]**3 - Y_hist[i, t]**3 for j in list_neighbors[i])
            second_order = sum(
                ((Y_hist[j, t] ** 2) * Y_hist[k, t] - Y_hist[i, t] ** 3) +
                ((Y_hist[k, t] ** 2) * Y_hist[j, t] - Y_hist[i, t] ** 3)
                for n, j, k in triangles_list if i in (n, j, k)
            )
            
            X_hist[i, t + 1] = X_hist[i, t] + h * (-Y_hist[i, t] - Z_hist[i, t] )
            Y_hist[i, t + 1] = Y_hist[i, t] + h * (X_hist[i, t] + a * Y_hist[i, t]+ s1 * first_order + s2 * second_order
            Z_hist[i, t + 1] = Z_hist[i, t] + h * (b + Z_hist[i, t] * (X_hist[i, t] - c))
            theta[i, t + 1] = np.arctan2(Y_hist[i, t + 1], X_hist[i, t + 1])
    
    return [X_hist, Y_hist, Z_hist, theta]



@jit(nopython=True,parallel=True)
def rossler_errors_grid_natural_coupling_type2(N,list_neighbors,triangles_list,a,b,c,s1_,s2_,h,numsteps,num_s2,n_resampling):
    """
    Generate the data for Figure 3(c)
    """
    errors_grid = np.zeros((num_s2,np.shape(s1_)[0]))
    
    order_grid = np.zeros((num_s2,np.shape(s1_)[0]))
    
    for idxs2 in prange(num_s2):
        s2 = s2_[idxs2]        
        for idxs1,s1 in enumerate(s1_):
            results_ = rossler_fun_natural_coupling_type2(N,list_neighbors,triangles_list,a,b,c,s1,s2,h,numsteps)

            X_,Y_,Z_,theta_ = results_
            
            results_resampled_ = resampling_ts(N,X_,Y_,Z_,theta_,n_resampling,numsteps)

            X_resampled,Y_resampled,Z_resampled,theta_resampled = results_resampled_
            n_steps_resampled = np.shape(X_resampled[0])[0]

            sync_errors = sync_error_fun(N,X_resampled,Y_resampled,Z_resampled,n_steps_resampled)
            
            r = r_order(N,theta_resampled, n_steps_resampled)
            
            errors_grid[idxs2,idxs1] = np.mean(np.array(sync_errors[20:]))
            
            order_grid[idxs2,idxs1] = np.mean(r[20:])

    return(errors_grid,order_grid)


@jit(nopython=True)
def rossler_fun_natural_coupling_type3(N, list_neighbors, triangles_list, a, b, c, s1, s2, h, numsteps):
    """
    Simulates the type II Rossler system on a network with pairwise and three-body interactions
    """
    X, Y, Z = np.random.random((3, N))
    X_hist, Y_hist, Z_hist = np.zeros((3, N, numsteps + 1))
    theta = np.zeros((N, numsteps + 1))
    
    X_hist[:, 0], Y_hist[:, 0], Z_hist[:, 0] = X, Y, Z
    theta[:, 0] = np.arctan2(Y, X)
    
    for t in range(numsteps):
        for i in range(N):
            first_order = sum(X_hist[j, t]**3 - X_hist[i, t]**3 for j in list_neighbors[i])
            second_order = sum(
                ((X_hist[j, t] ** 2) * X_hist[k, t] - X_hist[i, t] ** 3) +
                ((X_hist[k, t] ** 2) * X_hist[j, t] - X_hist[i, t] ** 3)
                for n, j, k in triangles_list if i in (n, j, k)
            )
            
            X_hist[i, t + 1] = X_hist[i, t] + h * (-Y_hist[i, t] - Z_hist[i, t] + s1 * first_order + s2 * second_order)
            Y_hist[i, t + 1] = Y_hist[i, t] + h * (X_hist[i, t] + a * Y_hist[i, t])
            Z_hist[i, t + 1] = Z_hist[i, t] + h * (b + Z_hist[i, t] * (X_hist[i, t] - c))
            theta[i, t + 1] = np.arctan2(Y_hist[i, t + 1], X_hist[i, t + 1])
    
    return [X_hist, Y_hist, Z_hist, theta]


@jit(nopython=True,parallel=True)
def rossler_errors_grid_natural_coupling_type3(N,list_neighbors,triangles_list,a,b,c,s1_,s2_,h,numsteps,num_s2,n_resampling):
    """
    Generate the data for Figure 3(d)
    """
    errors_grid = np.zeros((num_s2,np.shape(s1_)[0]))
    order_grid = np.zeros((num_s2,np.shape(s1_)[0]))
    
    for idxs2 in prange(num_s2):
        s2 = s2_[idxs2]        
        for idxs1,s1 in enumerate(s1_):
            results_ = rossler_fun_natural_coupling_type3(N,list_neighbors,triangles_list,a,b,c,s1,s2,h,numsteps)
            X_,Y_,Z_,theta_ = results_
            results_resampled_ = resampling_ts(N,X_,Y_,Z_,theta_,n_resampling,numsteps)

            X_resampled,Y_resampled,Z_resampled,theta_resampled = results_resampled_
            n_steps_resampled = np.shape(X_resampled[0])[0]

            sync_errors = sync_error_fun(N,X_resampled,Y_resampled,Z_resampled,n_steps_resampled)
            
            r = r_order(N,theta_resampled, n_steps_resampled)
            
            errors_grid[idxs2,idxs1] = np.mean(np.array(sync_errors[20:]))
            
            order_grid[idxs2,idxs1] = np.mean(r[20:])

    return(errors_grid,order_grid)



@jit(nopython=True)
def sync_error_fun(N, X_, Y_, Z_, n_steps_):
    """
    Computes the synchronization error over time.
    """
    sync_errors = np.zeros(n_steps_)
    for t in range(n_steps_):
        errors = [np.linalg.norm([X_[i, t] - X_[j, t], Y_[i, t] - Y_[j, t], Z_[i, t] - Z_[j, t]], ord=2) 
                  for i in range(N) for j in range(N)]
        sync_errors[t] = np.mean(errors)
    return sync_errors

@jit(nopython=True)
def r_order(N, theta_mem, n_steps_):
    """
    Computes the order parameter r(t) for phase synchronization.
    """
    r = np.zeros(n_steps_)
    for t in range(n_steps_):
        real_sum = np.sum(np.cos(theta_mem[:, t]))
        im_sum = np.sum(np.sin(theta_mem[:, t]))
        r[t] = np.sqrt(real_sum ** 2 + im_sum ** 2) / N
    return r

@jit(nopython=True)
def resampling_ts(N, X_, Y_, Z_, theta_, n_resampling, numsteps):
    """
    Resamples the time series at a lower resolution.
    """
    idxxx = np.arange(0, numsteps, n_resampling)
    X_resample_ = X_[:, idxxx]
    Y_resample_ = Y_[:, idxxx]
    Z_resample_ = Z_[:, idxxx]
    theta_resample_ = theta_[:, idxxx]
    return [X_resample_, Y_resample_, Z_resample_, theta_resample_]

@jit(nopython=True, parallel=True)
def rossler_errors_grid(N, list_neighbors, triangles_list, a, b, c, s1_, s2_, h, numsteps, num_s2, n_resampling):
    """
    Computes synchronization errors and order parameters over a grid of coupling values.
    """
    errors_grid = np.zeros((num_s2, len(s1_)))
    order_grid = np.zeros((num_s2, len(s1_)))
    
    for idxs2 in prange(num_s2):
        s2 = s2_[idxs2]        
        for idxs1, s1 in enumerate(s1_):
            results_ = rossler_fun(N, list_neighbors, triangles_list, a, b, c, s1, s2, h, numsteps)
            X_, Y_, Z_, theta_ = results_
            results_resampled_ = resampling_ts(N, X_, Y_, Z_, theta_, n_resampling, numsteps)
            X_resampled, Y_resampled, Z_resampled, theta_resampled = results_resampled_
            n_steps_resampled = X_resampled.shape[1]
            sync_errors = sync_error_fun(N, X_resampled, Y_resampled, Z_resampled, n_steps_resampled)
            r = r_order(N, theta_resampled, n_steps_resampled)
            errors_grid[idxs2, idxs1] = np.mean(sync_errors[20:])
            order_grid[idxs2, idxs1] = np.mean(r[20:])
    
    return errors_grid, order_grid



"""
Code to generate Fig.3(c)
""""
"""
import networkx as nx


G = nx.from_edgelist(edges_list)
list_neighbors = []
N = G.order()
for i in range(N):
    list_neighbors.append(np.array(list(G.neighbors(i))))

n_steps = 1500

h = 1e-3
numsteps = int(n_steps/h)

a=0.2
b=0.2
c=9

n_resampling = 200

num_s2 = 25
s1_ = np.logspace(-6,0,num_s2)
s2_ = np.logspace(-6,0,num_s2)

errors_grid,order_grid= rossler_errors_grid_natural_coupling_type2(N,list_neighbors,triangles_list,a,b,c,s1_,s2_,h,numsteps,num_s2,n_resampling)

"""
