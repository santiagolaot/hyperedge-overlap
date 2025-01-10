import numpy as np
from scipy.optimize import fsolve
from scipy.special import comb
import itertools

#S+ boundary:
def Splus_boundary(m, k):
    return m*k
#S- boundary:
def Sminus_equation(n, m, k):
    return comb(n, m) - k
def Sminus_boundary(m, k, initial_guess=20):
    n=np.ceil(np.round(fsolve(Sminus_equation, initial_guess, args=(m, k))[0],3))
    return n

#Intra order hyperedge overlap
def intra_order_hyperedge_overlap(groups, order):
    """
    Function to calculate both local and global intra-order hyperedge overlap of an order m.

    Parameters
    -----
    groups: ndarray
      Array with a list of m-hyperedges. Example for the case m=2 [(0,1,2),(2,3,4),...etc] 
      order: int containing the order m.
      Note that, the indices of the nodes must be continuous from 0 to N-1.
    
    Output
    -----
    global_m_intra_order_overlap : float64
      Value of global m_intra-order overlap
    local_m_intra_order_overlap : ndarray
      Array with the values of m_intra-order overlap of each node.
    
    """
    N = len(np.unique(groups))
    list_km_nodes = np.zeros(N, dtype=np.float64)  
    for group in groups:
        for node in group:
            list_km_nodes[node] += 1
    
    local_intra_order_overlap = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        km = int(list_km_nodes[i])  
        if km <= 1:
            continue
        
        Splus = Splus_boundary(order, km)
        Sminus = Sminus_boundary(order, km)
        
        union_set = set()
        for group in groups:
            if i in group:
                union_set.update(group)
        
        union_cardinality = len(union_set)-1
        local_intra_order_overlap[i]= 1 - (union_cardinality - Sminus) / (Splus - Sminus)
    
    global_intra_order_overlap = np.dot(local_intra_order_overlap, list_km_nodes) / np.sum(list_km_nodes)
    
    return global_intra_order_overlap,local_intra_order_overlap

#Inter order hyperedge overlap
def inter_order_hyperedge_overlap(groups_m, m, groups_n, n):
    """
    Function to calculate the inter-order hyperedge overlap of two orders m and n.

    Parameters
    -----
    groups_m: ndarray
      Array with a list of m-hyperedges. Example for the case m=1 [(0,1),(2,4),...etc] 
    m: int containing the order m.
    groups_n: ndarray
      Array with a list of n-hyperedges. Example for the case m=2 [(0,1,2),(2,3,4),...etc] 
    n: int containing the order n.
    Note that, the indices of the nodes must be continuous from 0 to N-1.
    
    Output
    -----
    m_n_inter_order_overlap : float64
    """

    if m == n:
        raise ValueError("Inter-order overlap is not defined for the same order")
    elif m > n:
        return 0
    else:
        possible_cliques = set()
        for clique in groups_n: #+1 since they are sotred with the length and not with the order
            for combination in itertools.combinations(clique, m+1):
                possible_cliques.add(combination) 
        actual_cliques = set()
        for clique in groups_m:
            actual_cliques.add(tuple(clique))
        intersection = len(possible_cliques.intersection(actual_cliques))
        if len(possible_cliques)>0:
            return intersection/len(possible_cliques)
        else:
            return 0
        
def hyperedge_overlap_matrix(groups, orders):
    """
    Function to compute the hyperedge overlap matrix for an hypergraph.

    Parameters
    ----------
    groups : list of ndarray
        A list where each entry contains the hyperedges of a specific order.
        For example, `groups[i]` contains a list of i-hyperedges.

    orders : list of int
        A list of integers representing the order of hyperedges for each group in `groups`.

    Returns
    -------
    overlap_matrix : ndarray
        A 2D square matrix where entry (i, j) represents the overlap between the i-th and j-th orders.
        - Diagonal entries represent intra-order overlaps.
        - Off-diagonal entries represent inter-order overlaps.
    """

    if len(groups) != len(orders):
        raise ValueError("The length of 'groups' and 'orders' must be the same.")

    max_order = max(orders)
    overlap_matrix = np.zeros((max_order, max_order))

    for i in range(max_order):
        for j in range(max_order):
            if i == j:  # Intra-order overlap
                if orders[i] != 1:  # Links do not have hyperedge-overlap
                    overlap_matrix[i, j], _ = intra_order_hyperedge_overlap(groups[i], orders[i])
            else:  # Inter-order overlap
                overlap_matrix[i, j] = inter_order_hyperedge_overlap(groups[i], orders[i], groups[j], orders[j])

    return overlap_matrix