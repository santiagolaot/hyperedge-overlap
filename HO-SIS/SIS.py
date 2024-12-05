import numpy as np
import random
from numba import jit,prange

@jit(nopython=True)
def get_stationary_rho(infecteds,N, normed=True, last_k_values = 100):
    i = infecteds
    if len(i)==0:
        return 0
    if normed:
        i = infecteds/N
    if i[-1]==1:
        return 1
    elif i[-1]==0:
        return 0
    else:
        avg_i = np.mean(i[-last_k_values:])
        return avg_i
    
    
@jit(nopython=True)
def simplicial_contagion_stationary_states(N,t_max,mu,beta_1,beta_2,edges,triangles,iters,num_inf,fixed_init):

    I_star_runs = []
    I_star = 0
    nnn=0
    for h in range(0,iters):

        indddd = fixed_init[h]
        Infected=np.linspace(0,0,N)
        Infected[indddd] = 1
        
        t = 0
        I = np.linspace(0,0,t_max)
        I[0] = np.sum(Infected)
        while np.sum(Infected)>0 and t<t_max:
            t = t+1
            newInfected = np.copy(Infected)

            # We run over the 2-body interactions
            for _,[v1,v2] in enumerate(edges):

                if Infected[v1] == 0 and Infected[v2] == 1 and np.random.rand()<beta_1: 
                    newInfected[v1] = 1
                elif Infected[v2] == 0 and Infected[v1] == 1 and np.random.rand()<beta_1:
                    newInfected[v2] = 1

            # We run over the 3-body interactions
            for _,[n1,n2,n3] in enumerate(triangles):

                if Infected[n1] == 0 and Infected[n2] == 1 and Infected[n3] == 1 and np.random.rand()<beta_2:
                    newInfected[n1] = 1
                elif Infected[n2] == 0 and Infected[n1] == 1 and Infected[n3] == 1 and np.random.rand()<beta_2:
                    newInfected[n2] = 1
                elif Infected[n3] == 0 and Infected[n1] == 1 and Infected[n2] == 1  and np.random.rand()<beta_2:
                    newInfected[n3] = 1

            # We run the linear recovery process
            for i in range(N):

                if Infected[i] == 1 and np.random.rand()<mu:
                    newInfected[i]=0

            Infected = np.copy(newInfected)
            I[t] = np.sum(Infected)
        
        value_ = get_stationary_rho(I,N)
        I_star += value_
        I_star_runs.append(value_)

    I_star = I_star/iters
    

    return I_star,np.array(I_star_runs)


@jit(nopython=True,parallel=True)  
def I_stationary_intra_order_overlap(lambdas_1,lambda_2,num_lambdas1,init_frac,N,n_steps,mu,edges_list,list_triangles_intra_overlap,iters,fixed_init):
    ## num_lambdas1 = int(len(lambdas_1))
    results_all_runs = np.zeros((len(list_triangles_intra_overlap),len(lambdas_1),iters))

    for idx in prange(num_lambdas1):
        lambda_1 = lambdas_1[idx]
        for idx2 in range(len(list_triangles_intra_overlap)):
            triangles_list = list_triangles_intra_overlap[idx2]
            k_1 = 2*len(edges_list)/N
            k_2 = 3*len(triangles_list)/N
            beta_1 = (lambda_1/k_1)*mu
            beta_2 = (lambda_2/k_2)*mu
            num_inf = int(N*init_frac)
            I_star,I_star_runs = simplicial_contagion_stationary_states(N,n_steps,mu,beta_1,beta_2,edges_list,triangles_list,iters,num_inf,fixed_init)         
            results_all_runs[idx2,idx,:] = I_star_runs

    return results_all_runs


"""
code to obtain the lower branch of the stochastic simulations of the original paper:

lambda_2 = 3
iters = 100
lambdas_1 = np.arange(0,2.5,0.01)
num_lambdas1 = int(len(lambdas_1))

init_frac = 0.01
num_inf_lower = int(N*init_frac)

fixed_init = np.zeros((iters,num_inf_lower),dtype=np.int64)

for h in range(iters):
    fixed_init[h] = np.random.choice(np.array(list(range(N))), num_inf_lower,replace=False)
n_steps = 6000
mu=0.05

results_lower_all_runs = I_stationary_intra_order_overlap(lambdas_1,lambda_2,num_lambdas1,init_frac,N,n_steps,mu,edges_list,list_triangles_simulations,iters,fixed_init)


"""



"""

code to obtain the upper branch of the stochastic simulations of the original paper:

lambda_2 = 3
iters = 100
lambdas_1 = np.arange(0,2.5,0.01)
num_lambdas1 = int(len(lambdas_1))

init_frac = 0.9
num_inf_upper = int(N*init_frac)

fixed_init = np.zeros((iters,num_inf_upper),dtype=np.int64)

for h in range(iters):
    fixed_init[h] = np.random.choice(np.array(list(range(N))), num_inf_upper,replace=False)
n_steps = 6000
mu=0.05

results_upper_all_runs = I_stationary_intra_order_overlap(lambdas_1,lambda_2,num_lambdas1,init_frac,N,n_steps,mu,edges_list,list_triangles_simulations,iters,fixed_init)

""""
