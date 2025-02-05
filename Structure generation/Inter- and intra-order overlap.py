
"""
Code to generate the structures used to study the combined effect of inter- and intra-order hyperedge overlap
"""

def regular_triples_patches(k_2, list_nodes):
    """
    We generate a regular set of patches that we will eventually use to construct the set of 2-hyperedges with maximized intra-order overlap
    """
    n = len(list_nodes)

    if k_2*n % 3 != 0:
        raise ValueError('k_2 * N must be a multiple of 3!')
    else:
        
        if n*(n-1)*(n-2)/6 < k_2*n/3:
            raise ValueError('You cannot obtain a regular set of 2-hyperedges with this k_2 and N!')

        elif n*(n-1)*(n-2)/6 >= k_2*n/3:

            triangles_list = []
            dict_k2 = {j:0 for j in list_nodes}
            counter = 0
            while sum(list(dict_k2.values())) != k_2*n: 
                counter+=1
                a = np.array(list(dict_k2.values())) 
                b = np.array(list(dict_k2.keys()))
                n1,n2,n3 = sorted(np.random.choice(b[a!=k_2],3,replace=False))
                if (n1,n2,n3) not in triangles_list:
                    triangles_list.append((n1,n2,n3))
                    dict_k2[n1] += 1 
                    dict_k2[n2] +=1 
                    dict_k2[n3] += 1
                else:
                    pass
                if counter == k_2*n*10:
                    raise ValueError("Too many attempts, try again")

    return np.array(triangles_list)


def triples_list_maximum_intraorder_overlap(size,N):
    '''size is the number of nodes of each fully overlapped cluster, becoming k_2=(size-1)*(size-2)/2
       N is the maximum number of nodes in play
       N_eff is the number of nodes of the structure (considering N_eff/size has to be integrer) 
    '''
    list_m_sizes=size*np.ones(int(N/size))
    list_m_sizes = np.asarray(list_m_sizes, dtype = 'int')
    N_eff=int(N/size)*size
    #We initialize the set of nodes
    nodes_set = set(range(N_eff))
    #And we select which nodes belong to each patch
    node_patches = []
    for m_ in list_m_sizes:
        patch = set(random.sample(sorted(nodes_set), m_)) # 1st random subset
        nodes_set -= patch
        node_patches.append(list(patch))
    #Once we have the node patches we can create the list of triples inside each patch
    triangles_list = []
    for idx,m_ in enumerate(list_m_sizes):

        k_2 = int(((m_-1)*(m_-2))/2)
        #for patch in node_patches:
        patch = node_patches[idx]
        triangles_list.append(regular_triples_patches(k_2, patch))
    #And we put them altogether
    triangles_list = np.array([item for sublist in triangles_list for item in sublist])
    #We compute the k_2 of each node
    list_k2_nodes = np.zeros(N_eff)
    for n in range(N_eff):
        for triple in triangles_list:
            if n in triple:
                list_k2_nodes[n] +=1
    #Note that the nodes start on zero
    #And the k_2 of the whole network
    k_2 = 3*(len(triangles_list))/N_eff
    return triangles_list, list_k2_nodes, k_2, N_eff


def regular_simplicial_complex_pairs(N,k_1,triangles_list,thres_):
    """
    This code generates a set of 1-hyperedges, for any given set of 2-hyperedges with regular distributions, with maximum value of inter-order hyperedge overlap (simplicial complex).
    """
    list_k1_nodes = np.zeros(N)
    total_link_expected = N*k_1
    print('Expected total number of link is %g'%total_link_expected)
    pairs_list=[]
    for tri_idx,tri in enumerate(triangles_list):
        a,b,c=tri
        if tri_idx == 0:
            pairs_list.append(np.sort([a,b]).tolist())
            list_k1_nodes[a] +=1
            list_k1_nodes[b] +=1
            pairs_list.append(np.sort([a,c]).tolist())
            list_k1_nodes[a] +=1
            list_k1_nodes[c] +=1
            pairs_list.append(np.sort([b,c]).tolist())
            list_k1_nodes[b] +=1
            list_k1_nodes[c] +=1
        else:
            if belongs_regular_pairs([a,b],np.array(pairs_list)) == False:
                pairs_list.append(np.sort([a,b]).tolist())
                list_k1_nodes[a] +=1
                list_k1_nodes[b] +=1

            if belongs_regular_pairs([a,c],np.array(pairs_list)) == False:
                pairs_list.append(np.sort([a,c]).tolist())
                list_k1_nodes[a] +=1
                list_k1_nodes[c] +=1
            if belongs_regular_pairs([b,c],np.array(pairs_list)) == False:
                pairs_list.append(np.sort([b,c]).tolist())
                list_k1_nodes[b] +=1
                list_k1_nodes[c] +=1

    res = 0
    groups_idx = np.where(list_k1_nodes < k_1)[0]

    while np.sum(list_k1_nodes) != N*k_1 and res < thres_ and len(groups_idx) > 1:
        new_edge = np.sort(np.random.choice(groups_idx,size=2,replace=False)).tolist()
        i,j = new_edge
        if i != j:
            if belongs_regular_pairs(new_edge,np.array(pairs_list)) == False:
                pairs_list.append(new_edge)
                res = 0
                a,b = new_edge
                list_k1_nodes[a] +=1
                list_k1_nodes[b] +=1
                groups_idx = np.where(list_k1_nodes < k_1)[0]
            else:
                res +=1
        else:
            res +=1
            pass
        
    total_link_created = len(pairs_list)
    link_not_created = (total_link_expected - 2*total_link_created)
    
    if nx.is_connected(nx.from_edgelist(pairs_list)) == False:
        print('Pairwise backbone structure not connected, process reinizialized')
        pairs_list = regular_simplicial_complex_pairs(N,k_1,triangles_list,thres_)
        
        
    if link_not_created == 0:
        print('The Regular Simplicial Complex has succesfully been created!')
    else:
        print("%g edges have not been created! repeating the procedure!" %link_not_created)
        pairs_list = regular_simplicial_complex_pairs(N,k_1,triangles_list,thres_)

    return np.array(pairs_list)


@jit(nopython=True)
def belongs_regular_pairs(pair1,edges):
    a,b = pair1
    res = False
    for pair in edges:
        if a in pair and b in pair:
            res = True
            break
    return res


"""
Example of code to generate a Regular Simplicial Complex with N=2000, k_1 = 6 and k_2 = 3, exhibiting maximum values for both intra- and inter-order hyperedge overlap.
k_1 represents the number of 1-hyperedges per node and k_2 is the number of 2-hyperedges connected to each node
"""
"""
N_start = 500
k_1 = 6
size = 4
thres_ = 100

triangles_list, list_k2_nodes, k_2, N = triples_list_maximum_intraorder_overlap(size,N_start)
edges_list = regular_simplicial_complex_pairs(N,k_1,triangles_list,thres_)
"""