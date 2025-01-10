import numpy as np
import pandas as pd
import random
import itertools
from itertools import combination
import matplotlib.pyplot as plt

def compute_laplacian(File, order):
    '''
    Computes the generalized laplacian for a given order of interactions according to Eq. (1) and (2). This was first introduced in Eq. (28) of L.V. Gambuzza et al. Nature communications (2021)

    Input:  File: pandas DataFrame with the groups of the given order of the network.

    Output: L_tensor: numpy array with the Laplacian tensor of the given order.

            k_array: numpy array with the degree array of the given order   
    '''

    N = File.max().max() + 1
    tensor_shape = tuple([N] * (order + 1))
    A_tensor = np.zeros(tensor_shape)

    # Populate adjacency tensor
    for _, row in File.iterrows():
        indices = row.values
        for perm in set(itertools.permutations(indices)):
            A_tensor[perm] = 1

    # Compute degree array
    k_array = A_tensor.sum(axis=tuple(range(1, order + 1))) / np.math.factorial(order)

    # Compute Laplacian tensor
    L_tensor = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                L_tensor[i, j] = -np.sum(A_tensor[i, j, ...]) / np.math.factorial(order - 1)
            else:
                L_tensor[i, j] = order * k_array[i]

    return np.array(L_tensor), k_array 

''' Example usage for specific orders

    def compute_pair_laplacian(File_Pairs):
        return compute_laplacian(File_Pairs, order=1)

    def compute_triplet_laplacian(File_Triplets):
        return compute_laplacian(File_Triplets, order=2)

    def compute_square_laplacian(File_Squares):
        return compute_laplacian(File_Squares, order=3)
        
'''

def compute_effective_laplacian(L_matrices, sigma_values):
    '''
    Computes the effective Laplacian as the weighted sum of multiple Laplacians.
    
    The effective Laplacian is computed according to Eq. (14) 
    
    Input:   L_matrices (list of numpy.ndarray): List of Laplacian matrices L^(m) for each instance.
             sigma_values (list of float): List of weights sigma^(m) for each Laplacian matrix.
    
    Returns: The computed effective Laplacian matrix.
    '''
    
    # Check if L_matrices and sigma_values have the same length
    if len(L_matrices) != len(sigma_values):
        raise ValueError("The number of Laplacians and sigma values must match.")
    
    # Initialize the effective Laplacian as a zero matrix
    L_eff = np.zeros_like(L_matrices[0], dtype=float)
    
    # Loop through each Laplacian matrix and its corresponding weight
    for L, sigma in zip(L_matrices, sigma_values):
        # Check if the Laplacian matrix is square
        if L.shape[0] != L.shape[1]:
            raise ValueError("Each Laplacian matrix must be square.")
        
        # Add the weighted Laplacian to the effective Laplacian
        L_eff += sigma * L
    
    return L_eff


def All_Eigenvalues_Computation(L_eff):
    '''
    Computes and returns the sorted eigenvalues of the provided effective laplacian matrix.

    Input: L_eff (numpy.ndarray): The effective Laplacian matrix.

    Returns: The sorted eigenvalues of the effective Laplacian matrix.
    '''
    
    # Ensure the input is a square matrix
    if L_eff.shape[0] != L_eff.shape[1]:
        raise ValueError("Input matrix must be square.")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(L_eff)
    
    # Sort eigenvalues in ascending order
    sorted_eigenvalues = np.sort(eigenvalues)
    
    return sorted_eigenvalues

def shoelace_area(vertices):
    '''
    Computes the area of a polygon using the Shoelace formula (also known as Gauss's area formula). The area is calculated as half the absolute value of the summation of cross-products of consecutive vertices. The polygon vertices must be ordered either clockwise or counterclockwise.
    
    Input:   vertices (list of tuple or list): A list of (x, y) tuples representing the vertices of the polygon in order (either clockwise or counterclockwise).
    
    Returns: The computed area of the polygon.
    '''
    # Ensure the vertices list is not empty and contains at least 3 points
    if len(vertices) < 3:
        raise ValueError("At least 3 vertices are required to calculate an area.")
    
    # Convert the list of vertices into x and y arrays
    x = np.array([v[0] for v in vertices])
    y = np.array([v[1] for v in vertices])

    # Shoelace formula to compute the polygon's area
    # We calculate half the absolute difference between two summations
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    return area

def get_area_from_matrix(bool_matrix):
    """
    Computes the area under a contour in a matrix, where the contour is defined at level 0. This function uses contour plots to define the region of interest and then calculates the area of that region using the Shoelace formula.

    Input:   2D-array-like boolean matrix, where points fulfilling the stability condition are "1" and points not fulfilling the stability condition are "0".
        
        IMPORTANT: THE BOOLEAN MATRIX MUST BE ALREADY IN LOG-SCALE AND WITHIN THE RANGE (0,1) TO MATCH THE RESULTS IN THE ARTICLE.

    Returns: tuple: A tuple containing:
                   - `area` (float): The area within the contour, computed using the Shoelace formula.
                   - `contour_points` (list of tuples): The list of contour vertices defining the region.
    
    Raises:
    ValueError: If the matrix has invalid dimensions or if no contour is found at level 0.
    """
    
    # Validate the matrix dimensions
    if len(bool_matrix.shape) != 2:
        raise ValueError("Input matrix must be a 2D array.")
    
    # Create a plot for contour generation
    fig, ax = plt.subplots()
    extent = [0, 1, 0, 1]  # Assuming unitary grid space for simplicity
    
    # Generate contour plot at level 0
    contours = ax.contour(
        bool_matrix.T,  # Transposed to align with the extent
        levels=[0],  # Define the level for the contour
        origin='lower',
        extent=extent,
        aspect='auto'
    )
    
    # Check if no contour is found
    if not contours.collections:
        raise ValueError("No contour found at the specified level.")
    
    # Extract contour vertices
    contour_points = []
    for collection in contours.collections:
        for path in collection.get_paths():
            vertices = path.vertices
            contour_points.extend(vertices)
    
    # Calculate the area using the Shoelace formula
    area = shoelace_area(contour_points)
    
    # Close the plot to release memory
    plt.close()
    
    return area, contour_points
