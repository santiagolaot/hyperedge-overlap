#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "Random_number_generator.h"
//Pi definition in case is not defined
#define M_PI 3.14159265358979323846



void Read_Links(int N, const char *Network_Type, int k_Pairs_teo, int k_Delta_teo, int **k_Pairs_out, int **V_out, int *k_Pairs_max_out) {
    // Allocate memory for k_Pairs and V
    int *k_Pairs = calloc(N, sizeof(int));
    int *V = calloc(N * N, sizeof(int));
    int k_Pairs_max = 0;

    // Temporary variables for reading links
    int ini, fin;

    // Construct the filename
    char fil_char[200];
    sprintf(fil_char, "Hypergraphs/%s_Pairs_N_%d_k_%d_k_delta_%d.txt", Network_Type, N, k_Pairs_teo, k_Delta_teo);

    // Open the file for reading
    FILE *fil = fopen(fil_char, "r");
    if (fil == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", fil_char);
        free(k_Pairs);
        free(V);
        return;
    }

    // Read the file line by line and initialize the adjacency matrix
    while (fscanf(fil, "%d %d\n", &ini, &fin) != EOF) {
        if (ini != fin) { // Skip self-loops
            ini -= 1; // Convert to 0-based index
            fin -= 1;

            // Record neighbors
            V[ini * N + k_Pairs[ini]] = fin;
            V[fin * N + k_Pairs[fin]] = ini;

            // Update degrees
            k_Pairs[ini]++;
            k_Pairs[fin]++;

            // Update maximum degree
            if (k_Pairs[ini] > k_Pairs_max) k_Pairs_max = k_Pairs[ini];
            if (k_Pairs[fin] > k_Pairs_max) k_Pairs_max = k_Pairs[fin];
        }
    }
    fclose(fil);

    // Return results
    *k_Pairs_out = k_Pairs;
    *V_out = V;
    *k_Pairs_max_out = k_Pairs_max;
}

void Degree_Distribution(int N, int k_Pairs_max, int *k_Pairs, int k_Pairs_teo, int k_Delta_teo, double *k_Pairs_med_out, int **Distribution_Grado_Pairs_out) {
    // Allocate memory for the degree distribution array
    int *Distribution_Grado_Pairs = calloc(k_Pairs_max + 1, sizeof(int));
    double k_Pairs_med = 0.0;
    int ii;

    // Calculate the degree distribution and mean degree
    for (ii = 0; ii < N; ii++) {
        Distribution_Grado_Pairs[k_Pairs[ii]]++;
        k_Pairs_med += k_Pairs[ii];
    }

    k_Pairs_med /= N;

    // Write the degree distribution to a file
    FILE *Distribution;
    char fil_char[200];
    sprintf(fil_char, "Hypergraphs/Pairs_Distribution_Grado_in_N_%d_k_%d_k_delta_%d.txt", N, k_Pairs_teo, k_Delta_teo);
    Distribution = fopen(fil_char, "w");
    if (Distribution == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", fil_char);
        free(Distribution_Grado_Pairs);
        return;
    }

    for (ii = 0; ii <= k_Pairs_max; ii++) {
        fprintf(Distribution, "%d\t%d\n", ii, Distribution_Grado_Pairs[ii]);
    }

    fclose(Distribution);

    // Return the mean degree and the degree distribution array
    *k_Pairs_med_out = k_Pairs_med;
    *Distribution_Grado_Pairs_out = Distribution_Grado_Pairs;

    // Print the statistics
    printf("\nHypergraph characteristics:\nNumber of nodes: %d\nAverage 1-hyperedges degree: %lf\nMaximum 1-hyperedges degree: %d\n", N, k_Pairs_med, k_Pairs_max);
}


void Read_Triangles(int N, const char *Network_Type, int k_Pairs_teo, int k_Delta_teo, double Overlapness, int **k_Trios_out, int **T_1_out, int **T_2_out, int *k_Trios_max_out) {
    // Allocate memory for k_Trios, T_1, and T_2
    int *k_Trios = calloc(N, sizeof(int)); // Stores the degree of each node in the triangle network
    int *T_1 = calloc(N * N, sizeof(int)); // Stores the first neighbor of each triangle (outgoing from node)
    int *T_2 = calloc(N * N, sizeof(int)); // Stores the second neighbor of each triangle (outgoing from node)

    // Initialize k_Trios_max
    int k_Trios_max = 0;

    // Open the file for reading
    FILE *fil;
    char fil_char[200];
    sprintf(fil_char, "Hypergraphs/%s_Triangles_N_%d_k_%d_k_delta_%d_T_%.4lf.txt", Network_Type, N, k_Pairs_teo, k_Delta_teo, Overlapness);
    fil = fopen(fil_char, "r");
    if (fil == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", fil_char);
        free(k_Trios);
        free(T_1);
        free(T_2);
        return;
    }

    // Read the triangle information from the file and process it
    int t0, t1, t2;
    while (fscanf(fil, "%d\t%d\t%d\n", &t0, &t1, &t2) != EOF) {
        t0 -= 1; // Adjust for 1-based indexing
        t1 -= 1;
        t2 -= 1;

        // Save triangle neighbors
        T_1[(t0) * N + k_Trios[t0]] = t1;
        T_2[(t0) * N + k_Trios[t0]] = t2;
        T_1[(t1) * N + k_Trios[t1]] = t0;
        T_2[(t1) * N + k_Trios[t1]] = t2;
        T_1[(t2) * N + k_Trios[t2]] = t0;
        T_2[(t2) * N + k_Trios[t2]] = t1;

        // Update degree (out-degree)
        k_Trios[t0]++;
        k_Trios[t1]++;
        k_Trios[t2]++;

        // Update the maximum degree
        if (k_Trios[t0] > k_Trios_max) k_Trios_max = k_Trios[t0];
        if (k_Trios[t1] > k_Trios_max) k_Trios_max = k_Trios[t1];
        if (k_Trios[t2] > k_Trios_max) k_Trios_max = k_Trios[t2];
    }

    fclose(fil);

    // Return the results through output pointers
    *k_Trios_max_out = k_Trios_max;
    *k_Trios_out = k_Trios;
    *T_1_out = T_1;
    *T_2_out = T_2;
}

void Degree_Distribution_T(int N, int k_Pairs_teo, int k_Delta_teo, int *k_Trios, int k_Trios_max, double *k_Trios_med_out, int **Distribution_Grado_Trios_out) {
    int *Distribution_Grado_Trios = calloc((k_Trios_max + 1), sizeof(int));
    double k_Trios_med = 0.0;
    int i;

    // Calculate the distribution and the mean degree
    for (i = 0; i < N; i++) {
        Distribution_Grado_Trios[k_Trios[i]]++;
        k_Trios_med += k_Trios[i];
    }
    
    k_Trios_med /= N;

    // Create the output file name and open the file
    FILE *Distribution;
    char fil_char[200];
    sprintf(fil_char, "Hypergraphs/Triangles_Distribution_Grado_in_N_%d_k_%d_k_delta_%d.txt", N, k_Pairs_teo, k_Delta_teo);
    Distribution = fopen(fil_char, "w");
    if (Distribution == NULL) {
        printf("Error: Unable to open file %s\n", fil_char);
        free(Distribution_Grado_Trios); // Don't forget to free the allocated memory
        return;
    }

    // Write the degree distribution to the file
    for (i = 0; i <= k_Trios_max; i++) {
        fprintf(Distribution, "%d\t%d\n", i, Distribution_Grado_Trios[i]);
    }

    fclose(Distribution);

    // Print the statistics
    printf("Average 2-hyperedges degree: %lf\nMaximum 2-hyperedges degree: %d\n", k_Trios_med, k_Trios_max);

    *k_Trios_med_out = k_Trios_med;
    *Distribution_Grado_Trios_out = Distribution_Grado_Trios;

    // Free the dynamically allocated memory
}

void Initialize_vectors(int N, int Iter, double **natural_w_out, double **w_eff_out, double **theta_mem_out) {
    // Allocate memory for the vectors
    double *natural_w = calloc(N, sizeof(double));  // Cast the void pointer to double pointer
    double *w_eff = calloc(N, sizeof(double));      // Cast the void pointer to double pointer
    double *theta_mem = calloc(N * Iter, sizeof(double));  // Cast the void pointer to double pointer


    // Return the allocated memory through output pointers
    *natural_w_out = natural_w;
    *w_eff_out = w_eff;
    *theta_mem_out = theta_mem;

    // Optional: Initialize the vectors (you can add any initializations if needed)
    // Example: for(int i = 0; i < N; i++) { natural_w[i] = 0.0; }
}

void Free_vectors(double *natural_w, double *w_eff, double *theta_mem) {
    // Free the allocated memory for the vectors
    free(natural_w);
    free(w_eff);
    free(theta_mem);
}

void Initial_Conditions(int N, double *natural_w, double *theta_mem, double freq_factor, double phase_factor) {
    // Natural frequencies within the interval [-1/2, 1/2]
    // Initial phases in the range [-pi, pi]. This is done effectively by setting between [-1, 1] and then multiplying it by pi (M_PI)

    int i;
    for(i = 0; i < N; i++) {
        natural_w[i] = freq_factor * 2 * M_PI * (Random() - 0.5);  // k_in[i];
        theta_mem[i] = 2 * phase_factor * Random() - phase_factor;
    }
    // Note that we do not change the value of the effective frequencies, since they are initialized to zero by calloc
}





void Force_theta(int i, double *aux, double *ki, double *natural_w, int *T_1, int *T_2, int *k_Pairs, int *k_Trios, double sigma_P, double sigma_T, int N, int k_Pairs_med, int k_Trios_med, int *V) {
    double force_P, force_T;
    int j, k;
    
    force_P = force_T = 0;
    
    // Force from pairs
    for(j = 0; j < k_Pairs[i]; j++) {
        force_P += sin(M_PI * (aux[V[N * i + j]] - aux[i]));
    }
    force_P *= sigma_P / k_Pairs_med;
    
    // Force from trios
    for(k = 0; k < k_Trios[i]; k++) {
        force_T += sin(M_PI * (aux[T_1[i * N + k]] + aux[T_2[i * N + k]] - 2 * aux[i]));
    }
    force_T *= sigma_T / k_Trios_med;
    
    // Update ki[i] with the natural frequency and forces
    ki[i] = natural_w[i] + force_P + force_T;
}


void Evolution(int N, int t, double h, double *theta_mem, double *w_eff, double *natural_w, int *T_1, int *T_2, int *k_Pairs, int *k_Trios, int *V, double sigma_P, double sigma_T, int k_Pairs_med, int k_Trios_med) {
    int i, j, k;
    double k1[N], k2[N], k3[N], k4[N], aux[N], c_aux[N];
    
    // Set the copies and update the differences
    for(i = 0; i < N; i++) {
        aux[i] = theta_mem[t * N + i];
    }


    // RK First step
    for(i = 0; i < N; i++) {
        Force_theta(i, aux, k1, natural_w, T_1, T_2, k_Pairs, k_Trios, sigma_P, sigma_T, N, k_Pairs_med, k_Trios_med, V);
        c_aux[i] = theta_mem[t * N + i] + h * k1[i] / 2;
    }

    // RK Second step
    for(i = 0; i < N; i++) {
        Force_theta(i, c_aux, k2, natural_w, T_1, T_2, k_Pairs, k_Trios, sigma_P, sigma_T, N, k_Pairs_med, k_Trios_med, V);
        aux[i] = theta_mem[t * N + i] + h * k2[i] / 2;
    }

    // RK Third step
    for(i = 0; i < N; i++) {
        Force_theta(i, aux, k3, natural_w, T_1, T_2, k_Pairs, k_Trios, sigma_P, sigma_T, N, k_Pairs_med, k_Trios_med, V);
        c_aux[i] = theta_mem[t * N + i] + h * k3[i];
    }

    // RK Fourth step
    for(i = 0; i < N; i++) {
        Force_theta(i, c_aux, k4, natural_w, T_1, T_2, k_Pairs, k_Trios, sigma_P, sigma_T, N, k_Pairs_med, k_Trios_med, V);
        theta_mem[(t + 1) * N + i] = theta_mem[t * N + i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);

        // Ensure theta values are within [-1, 1]
        while(theta_mem[(t + 1) * N + i] > 1)
            theta_mem[(t + 1) * N + i] -= 2;
        while(theta_mem[(t + 1) * N + i] < -1)
            theta_mem[(t + 1) * N + i] += 2;
        
        w_eff[i] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
}


double r_order(int N, int t, double *theta_mem) {
    int i;
    double r, Real = 0, Im = 0, Phi = 0;
    
    // Sum phases and calculate real and imaginary parts
    for(i = 0; i < N; i++) {
        Phi += theta_mem[t * N + i];
        Real += cos(M_PI * theta_mem[t * N + i]);
        Im += sin(M_PI * theta_mem[t * N + i]);
    }

    // Calculate the mean phase
    Phi /= N;
    Phi *= M_PI;

    // Calculate the order parameter
    r = sqrt((Real * Real + Im * Im) / (cos(Phi) * cos(Phi) + sin(Phi) * sin(Phi))) / N;
    
    return r;
}


double r_link(int N, int Iter, int Iter_burn, double *theta_mem, int *k_Pairs, int *k_Trios, int *V, int *T_1, int *T_2) {
    int i, j, k, tt;
    double r, dif, rij = 0, Real = 0, Im = 0, norm_p = 0, norm_t = 0;

    // Loop over all oscillators (nodes)
    for(i = 0; i < N; i++) {
        // Loop over pairs of nodes
        for(j = 0; j < k_Pairs[i]; j++) {
            Real = Im = 0;
            // Loop over time steps
            for(tt = Iter_burn + (int)((Iter - Iter_burn) / 2); tt < Iter; tt++) {
                dif = M_PI * (theta_mem[tt * N + i] - theta_mem[tt * N + V[i * N + j]]);
                Real += cos(dif);
                Im += sin(dif);
            }
            rij += sqrt(Real * Real + Im * Im) / (Iter - (Iter_burn + (int)((Iter - Iter_burn) / 2))); // Average over time
            norm_p += 1;
        }

        // Loop over triads of nodes
        for(k = 0; k < k_Trios[i]; k++) {
            Real = Im = 0;
            // Loop over time steps
            for(tt = Iter_burn + (int)((Iter - Iter_burn) / 2); tt < Iter; tt++) {
                dif = M_PI * (theta_mem[tt * N + T_1[i * N + k]] + theta_mem[tt * N + T_2[i * N + k]] - 2 * theta_mem[tt * N + i]);
                Real += cos(dif);
                Im += sin(dif);
            }
            rij += sqrt(Real * Real + Im * Im) / (Iter - (Iter_burn + (int)((Iter - Iter_burn) / 2))); // Average over time
            norm_t += 1;
        }
    }

    // Compute the final r_link value as the average of both types of connections
    r = rij / (norm_p + norm_t);

    return r; // Return the computed r_link value
}




































