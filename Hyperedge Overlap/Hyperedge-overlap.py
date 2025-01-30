import numpy as np
from numba import jit,prange
import networkx as nx
import random

def regular_maximum_overlapped_simplicial_complex(n, k, seed=None):
    """
    -----------------------------------------------------------------------------------
    Thanks to Luca Gallo - https://scholar.google.com/citations?hl=it&user=sKiSU9AAAAAJ
    -----------------------------------------------------------------------------------
    
    Returns a regular simplicial complex of order 2 with maximum value of intra-order hyperedge overlap.

    Each node has exactly
    - 1-degree = k
    - 2-degree = (k-1)(k-2)/2
    We first fix a value of initial number of nodes n, the final structure will be composed of N = k*n nodes.
    
    Parameters
    ----------
    k : int
      The degree of each node.
    n : int
      The number of nodes for the starting k-regular random graph. 
      The value of $N \times k$ must be even.
      The number of nodes of the simplicial complex will be k*N.
    seed : int, random_state, or None (default)
        Indicator of random number generation state.
        
    Output
    ----------
    N : int
      The number of nodes in the simplicial complex.
    edges_list : ndarray
      Array composed by a list of tuples [(0,1),(2,3),...,etc] composing the set of 1-simplices
    triangles_list : ndarray
      Array composed by a list of tuples [(0,1,2),(2,3,4),...,etc] representing the set of 2-simplices
    """
    

    M = nx.random_regular_graph(k,n,seed=seed)
    
    graph_to_simplex = {ii:[ii*k+r for r in range(k)] for ii in range(n)}
    
    G = nx.disjoint_union_all([nx.complete_graph(k) for ii in range(n)])
    
    triangle_list = []
    for clique in nx.enumerate_all_cliques(G):
        if len(clique)==3:
            triangle_list.append(clique)
        
    for edge in M.edges():
        m1 = edge[0]
        m2 = edge[1]
        n1 = random.choice(graph_to_simplex[m1])
        n2 = random.choice(graph_to_simplex[m2])
        
        G.add_edge(n1,n2)
        
        graph_to_simplex[m1].remove(n1)
        graph_to_simplex[m2].remove(n2)
        
    edges_list = np.array(G.edges())
    N = n*k
    return N,np.array(edges_list), np.array(triangle_list)


@jit(nopython=True)#, fastmath=True)
def belongs_intraorder(triple1,triple2,triples):
    a,b,c = triple1
    aa,bb,cc = triple2
    res = False
    for triple in triples:
        if a in triple and b in triple and c in triple:
            res = True
            break
        elif aa in triple and bb in triple and cc in triple:
            res = True
            break
    return res
    
@jit(nopython=True)
def creating_list_k2_nodes(N,triangles_list):
    list_k2_nodes = np.zeros(N, dtype=np.int64)  # Changed to float64
    for a, b, c in triangles_list:
        list_k2_nodes[a] += 1
        list_k2_nodes[b] += 1
        list_k2_nodes[c] += 1
    return list_k2_nodes

@jit(nopython=True)
def minimizing_intra_order_overlap(N,triangles_list,Iterations,thres_):
    """
    Minimization algorithm for the intra-order overlap.
    
    ----------
    Parameters
    N : int
      Number of nodes
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
      List of lists of 2-hyperedges for each value of intra-order hyperedge overlap.
    intraorder_structures : ndarray 
      List of values of intra-order hyperedge overlap obtained through the minimization process.
    """
    New_P = triangles_list.copy()
    list_k2_nodes = creating_list_k2_nodes(N,triangles_list)
    intraorder_structures = []
    list_triangles = []
    list_triangles.append(triangles_list)
    intraorder_structures.append(global_intra_order_overlap_minimization(N, triangles_list,list_k2_nodes))
    i=0
    
    New_intraorder= intraorder_structures[0]
    while i <= Iterations and New_intraorder != 0:
        #We choose a link
        change=False
        mmm = 0
        while (change==False) and mmm < thres_:
            P_aux = New_P.copy()
            triple1_idx = np.random.choice(np.array(list(range(np.shape(P_aux)[0]))),1,replace=False)[0]  
            target_idx=np.random.choice(np.array([0,1,2]),1,replace=False)[0]
            #We choose a new source node to change the link in order to fix the connectivity
            accepted=False
            while (accepted==False) and mmm < thres_:
                P_aux = New_P.copy()
    
                triple1 = P_aux[triple1_idx]
                target = triple1[target_idx]
                #We set a possible source.
                triple2_idx = np.random.choice(np.array(list(range(np.shape(triangles_list)[0]))),1,replace=False)[0]
                triple2= P_aux[triple2_idx]
                possible_source_idx= np.random.choice(np.array([0,1,2]),1,replace=False)[0]
                possible_source = triple2[possible_source_idx]
                #We check if it is an autolink
                if possible_source==target or triple1_idx == triple2_idx or target in triple2 or possible_source in triple1:
                    accepted=False
                    mmm += 1
                #We check if the link already exists:
                else:
                    accepted=True
                    triple1 = np.delete(triple1,target_idx)
                    triple1 = np.append(triple1,possible_source)
                    triple2 = np.delete(triple2,possible_source_idx)
                    triple2 = np.append(triple2,target)
                    
                    if belongs_intraorder(triple1,triple2,P_aux) == False:

                        accepted = True
                    else:
                        accepted = False
                        mmm +=1
                    
            P_aux[triple1_idx] = np.array(sorted(triple1))
            P_aux[triple2_idx] = np.array(sorted(triple2))
            
            New_intraorder=global_intra_order_overlap_minimization(N, P_aux,list_k2_nodes)
            Old_intraorder=global_intra_order_overlap_minimization(N, New_P,list_k2_nodes)

            if(New_intraorder<Old_intraorder):
                New_P[triple1_idx] = np.array(sorted(triple1))
                New_P[triple2_idx] = np.array(sorted(triple2))              
                change=True
                list_triangles.append(New_P.copy())
                intraorder_structures.append(New_intraorder)
                print('Intra-order overlap:',New_intraorder)
            else:
                mmm+=1
        i+=1
        
    return(list_triangles,np.array(intraorder_structures))



@jit(nopython=True)
def global_intra_order_overlap_minimization(N, triangles_list,list_k2_nodes):
    """
    Global intra-order overlap minimization
    This is only for minimization. 
    To enhance the computational effiency we generate the list of 2-degrees of each node outside the function.
    """
    local_intra_order_overlap = np.ones(N, dtype=np.float64)
    
    for i in range(N):
        k2 = int(list_k2_nodes[i])  # Convert to int for calculations
        if k2 <= 1:
            continue
        
        max_cardinality = 2 * k2 + 1
        min_cardinality = np.ceil((3 + np.sqrt(1 + 8 * k2)) / 2)
        
        union_set = set()
        for triple in triangles_list:
            if i in triple:
                union_set.update(triple)
        
        union_cardinality = len(union_set)
        local_intra_order_overlap[i] =  1 - (union_cardinality - min_cardinality) / (max_cardinality - min_cardinality)
    
    return np.dot(local_intra_order_overlap, list_k2_nodes.astype(np.float64)) / np.sum(list_k2_nodes)


@jit(nopython=True)
def intra_order_overlap(triangles_list):
    """
    Function to calculate both local and global intra-order hyperedge overlap.

    Parameters
    -----
    N : int
      Number of nodes
    triangles_list : ndarray
      Array with a list of 2-hyperedges in the form [(0,1,2),(2,3,4),...etc] 
      Note that, the indices of the nodes must be continuous from 0 to N-1.
    
    Output
    -----
    global_intra_order_overlap : float64
      Value of global intra-order overlap
    local_intra_order_overlap : ndarray
      Array with the values of intra-order overlap of each node.
    
    """
    N = len(np.unique(triangles_list))
    list_k2_nodes = np.zeros(N, dtype=np.float64)  
    for a, b, c in triangles_list:
        list_k2_nodes[a] += 1
        list_k2_nodes[b] += 1
        list_k2_nodes[c] += 1
    
    local_intra_order_overlap = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        k2 = int(list_k2_nodes[i])  
        if k2 <= 1:
            continue
        
        max_cardinality = 2 * k2 + 1
        min_cardinality = np.ceil((3 + np.sqrt(1 + 8 * k2)) / 2)
        
        union_set = set()
        for triple in triangles_list:
            if i in triple:
                union_set.update(triple)
        
        union_cardinality = len(union_set)
        local_intra_order_overlap[i] = 1 - (union_cardinality - min_cardinality) / (max_cardinality - min_cardinality)    
    
    global_intra_order_overlap = np.dot(local_intra_order_overlap, list_k2_nodes.astype(np.float64)) / np.sum(list_k2_nodes)
    
    return global_intra_order_overlap,local_intra_order_overlap



"""
Code to obtain the structures of the original paper

init_nodes = 200
k1 = 5
N,edges_list,triangles_list = regular_maximum_overlapped_simplicial_complex(init_nodes, k1, seed=0)

Iterations = 10000
thres_ = 1
list_triangles,intraorder_graphs = minimizing_intra_order_overlap(N,triangles_list,Iterations,thres_)

list_intraorder = np.linspace(1,0,51)
idx_to_save = []
for intra_ in list_intraorder:
    idx = np.where(intraorder_graphs <= intra_)[0][0]
    idx_to_save.append(idx)
    
list_triangles_simulations = []

for idx__,hyperedges_list in enumerate(list_triangles):
    if idx__ in idx_to_save:
        list_triangles_simulations.append(hyperedges_list)

"""
