import numpy as np
import math
import random
from numba import jit
from misc import *

@jit(nopython=True)
def inter_order_overlap(edges_list, triangles_list):
    """
    Function to evaluate inter-order hyperedge overlap for M=2 (between the sets of 1- and 2-hyperedges).

    Parameters
    -----
    edges_list : ndarray
      Array with a list of 1-hyperedges in the form [(0,1),(1,2),...etc] 
      Note that, the indices of the nodes must be continuous from 0 to N-1.
    triangles_list : ndarray
      Array with a list of 2-hyperedges in the form [(0,1,2),(2,3,4),...etc] 
      Note that, the indices of the nodes must be continuous from 0 to N-1.
    
    Output
    -----
    inter_order_overlap : float64
      Value of inter-order hyperedge overlap
    local_intra_order_overlap : ndarray
      Array with the values of intra-order overlap of each node.
    
    """
    # Use a set to store needed links for fast lookup
    needed_links = set()
    
    # Iterate over triangles to add all unique undirected edges (sorted)
    for triple in triangles_list:
        i, j, k = triple
        needed_links.add(sort_edge((i, j)))
        needed_links.add(sort_edge((i, k)))
        needed_links.add(sort_edge((j, k)))
    
    # Initialize set to track existing needed links and count duplicates
    existing_needed_links = set()
    repes = 0

    # Iterate over the edges list and check for needed links
    for pair in edges_list:
        edge = sort_edge(pair)
        
        if edge in needed_links:
            if edge not in existing_needed_links:
                existing_needed_links.add(edge)
            else:
                repes += 1

    # Calculate inclusiveness (inter_order_overlap)
    inter_order_overlap = len(existing_needed_links) / len(needed_links)
    
    return inter_order_overlap

@jit(nopython=True)#, fastmath=True)
def belongs_interorder(pair1,pair2,edges):
    a,b = pair1
    aa,bb = pair2
    res = False
    for pair in edges:
        if a in pair and b in pair:
            res = True
            break
        elif aa in pair and bb in pair:
            res = True
            break
    return res


@jit(nopython=True)
def minimizing_inter_order_overlap(edges_list,triangles_list,Iterations,thres_):
    """
    Minimization algorithm for the inter-order overlap.
    
    ----------
    Parameters
    edges_list : ndarray
      Array with a list of 1-hyperedges in the form [(0,1),(1,2),...etc] 
      Note that, the indices of the nodes must be continuous from 0 to N-1.
    triangles_list : ndarray
      Array with a list of 2-hyperedges in the form [(0,1,2),(2,3,4),...etc] 
      Note that, the indices of the nodes must be continuous from 0 to N-1.
    Iterations : int
      Number of maximum iterations of the minimization algorithm
    thres_ : int
      To prevent infinite loops, we set a maximum number of failures for the iteration
      
    ----------
    Output
    
    list_triangles : ndarray
      List of lists of 2-hyperedges for each value of inter-order hyperedge overlap.
    interorder_structures : ndarray 
      List of values of inter-order hyperedge overlap obtained through the minimization process.
    """

    New_P = edges_list.copy()
    
    interorder_structures = []

    
    Inter_order=[]   
    Inter_order.append(inter_order_overlap(edges_list,triangles_list))
    list_pairs = []
    list_pairs.append(edges_list)
    interorder_structures.append(Inter_order[0])
    i=0
    
    New_interorder= Inter_order[0]
    while i <= Iterations and New_interorder != 0:
        #We choose a link susceptible to be changed
        change=False
        mm = 0
        while (change==False) and mm < 1:
            P_aux = New_P.copy()
            pair1_idx = np.random.choice(np.array(list(range(np.shape(P_aux)[0]))),1,replace=False)[0]  #random.randint(0,len(New_P)-1)
            target_idx=np.random.choice(np.array([0,1]),1,replace=False)[0]
            #We choose a new source node to change the link
            #We keep the first column as it is to not change the in degree
            #If we wolud like to tune it as a SF one it would be also necessary to change it
            accepted=False
            while (accepted==False) and mm < thres_:
                pair1 = P_aux[pair1_idx]
                target = pair1[target_idx]
                #We set a possible source. This election could be done according to other critera, to include preferential attatchemtn to higher connectivities
                #Also could be used the energy landscape to choose it
                pair2_idx = np.random.choice(np.array(list(range(np.shape(P_aux)[0]))),1,replace=False)[0]
                pair2= P_aux[pair2_idx]
                #possible_source_idx= np.random.choice(triangles_list[num2],1,replace=False)[0]
                possible_source_idx= np.random.choice(np.array([0,1]),1,replace=False)[0]
                possible_source = pair2[possible_source_idx]
                #We check if it is an autolink or if the link already exists:
                if possible_source==target or pair1_idx == pair2_idx or target in pair2 or possible_source in pair1:
                    accepted=False
                    mm +=1
                else:
                    accepted=True
                    pair1 = np.delete(pair1,target_idx)
                    pair1 = np.append(pair1,possible_source)
                    pair2 = np.delete(pair2,possible_source_idx)
                    pair2 = np.append(pair2,target)
                    
                    if belongs_interorder(sorted(pair1),sorted(pair2),New_P) == False:
                        accepted = True
                    else:
                        accepted = False
                        mm +=1
            #Now we see how it affects to the overlap
            if accepted == True:
                P_aux[pair1_idx] = np.array(sorted(pair1))
                P_aux[pair2_idx] = np.array(sorted(pair2))
                
                New_interorder=inter_order_overlap(P_aux,triangles_list)
                Old_interorder=inter_order_overlap(New_P,triangles_list)
            #Decission rule. By the moment we just set that if the overlap is equal or less we accept the change
                if(New_interorder<Old_interorder):
                    New_P = P_aux.copy()
                    change=True
                    Inter_order.append(New_interorder)
                    list_pairs.append(New_P.copy())
                    interorder_structures.append(New_interorder)
                    print('Inter-order overlap:',New_interorder)
            else:
                mm +=1
        i+=1

    return(list_pairs,interorder_structures)



"""
Code to minimize the inter-order hyperedge overlap for a regular simplicial complex, with N=2000, k_1 =6, k_2= 3 andh maximum intra-order hyperedge overlap
"""
"""
N_start = 500
k_1 = 6
size = 4
thres_ = 100

triangles_list, list_k2_nodes, k_2, N = triples_list_maximum_intraorder_overlap(size,N_start)
edges_list = regular_simplicial_complex_pairs(N,k_1,triangles_list,thres_)

list_pairs,interorder_structures = minimizing_inter_order_oveminimizing_inter_order_overlap(edges_list,triangles_list,Iterations,thres_)
"""
