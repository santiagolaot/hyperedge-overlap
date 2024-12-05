//This code corresponds to the article: 
//   Malizia, F., Lamata-Otín S., Frasca, M., Latora, V. and Gómez-Gardeñes, J.
//   'Hyperedge overlap drives explosive transitions in systems with higher-order interactions' 
//   Nature Communications (2024)

//This code allows for the integration of the dynamics of the higher-order Kuramoto model in a network with hyperedges of order 2 and 3, and computes the order parameters

//The code takes as imputs:
//  Parameters red from the command line:
//  Hypergraphs contained in a Folder called "Hypergraphs"

//The code returns in the "Results folder" the following files:
//  Stationary_lP_%.4lf_lT_%.4lf_phase_%.4lf_freq_%.4lf_%s_N_%d_k_%d_kT_%d_T_%.4lf_Iter_burn_%d_Iter_%d_h_%.4lf.txt
//       containing the stationary values of the order parameters r (Eq. (5)) and r_loc (Eq. (6))
//  w_eff_lP_%.4lf_lT_%.4lf_phase_%.4lf_freq_%.4lf_%s_N_%d_k_%d_kT_%d_T_%.4lf_Iter_burn_%d_Iter_%d_h_%.4lf.txt  
//       containing the effective frequencies of the nodes (Eq. (7)) at the end of the simulation
//  Evolution_lP_%.4lf_lT_%.4lf_%.4lf_phase_%.4lf_freq_%.4lf_%s_N_%d_k_%d_kT_%d_T_%.4lf_Iter_burn_%d_Iter_%d_h_%.4lf.txt
//       containing the evolution of the order parameter r (Eq. (5)) during the simulation


//The utils file contains the functions that are used in the main code
#include "HO-Kuramoto-utils.h"

//The main takes the following arguments:
//   Network_Type: Type of network (e.g., ER, SF, BA, ...)
//   N: Number of nodes
//   k_Pairs_teo: Average degree of the 1-hyperedges
//   k_Delta_teo: Average degree of the 2-hyperedges
//   Overlapness: Intra-order hyperedge overlap of the 2-hyperedges
//   Iter_burn: Number of iterations for the burn-in time
//   Iter: Number of iterations for the observable computation
//   h: Integration step
//   sigma_T: Coupling strength of the 2-hyperedges 
//   sigma_P: Coupling strength of the 1-hyperedges

int main(int argc, char *argv[]){

//We initialize the random generator
//ini_ran(123456789); //Fixed initialization of the random generator
ini_ran(time(NULL)); //Variable initialization of the random generator

//We initialize the variables read from the command line
char Network_Type[200];
sprintf(Network_Type,"%s",argv[1]);

int N=atoi(argv[2]);
int k_Pairs_teo=atoi(argv[3]);
int k_Delta_teo=atoi(argv[4]);
double Overlapness=strtod(argv[5], NULL);
int Iter_burn=atoi(argv[6]);
int Iter=atoi(argv[7]);
double h=strtod(argv[8], NULL);
double sigma_T=strtod(argv[9], NULL);
double sigma_P=strtod(argv[10], NULL);
double phase_factor=strtod(argv[11], NULL);
double freq_factor=strtod(argv[12], NULL);

printf("\nDynamical parameters:\n");
printf("1-hyperedges strength=%.4lf\n",sigma_P);
printf("2-hyperedges strength=%.4lf\n",sigma_T);
printf("Phase factor=%.4lf\n\n",phase_factor);

printf("\nSimulation parameters:\n");
printf("Iterations=%d\n",Iter);
printf("Integration step=%.4lf\n",h);
printf("Burn-in iterations=%d\n\n",Iter_burn);


//Create the files to store the results
FILE *r_stationary_File,*r_evo_File, *w_eff_File;
char r_stationary_char[400], r_evo_char[400], w_eff_char[400];
sprintf(r_stationary_char,"Results/Stationary_lP_%.4lf_lT_%.4lf_phase_%.4lf_freq_%.4lf_%s_N_%d_k_%d_kT_%d_T_%.4lf_Iter_burn_%d_Iter_%d_h_%.4lf.txt",sigma_P,sigma_T,phase_factor,freq_factor,Network_Type,N,k_Pairs_teo,k_Delta_teo,Overlapness,Iter_burn,Iter,h);
r_stationary_File=fopen(r_stationary_char,"w");
sprintf(w_eff_char,"Results/w_eff_lP_%.4lf_lT_%.4lf_phase_%.4lf_freq_%.4lf_%s_N_%d_k_%d_kT_%d_T_%.4lf_Iter_burn_%d_Iter_%d_h_%.4lf.txt",sigma_P,sigma_T,phase_factor,freq_factor,Network_Type,N,k_Pairs_teo,k_Delta_teo,Overlapness,Iter_burn,Iter,h);
w_eff_File=fopen(w_eff_char,"w");
sprintf(r_evo_char,"Results/Evolution_lP_%.4lf_lT_%.4lf_%.4lf_phase_%.4lf_freq_%.4lf_%s_N_%d_k_%d_kT_%d_T_%.4lf_Iter_burn_%d_Iter_%d_h_%.4lf.txt",sigma_P,sigma_T,phase_factor,freq_factor,Network_Type,N,k_Pairs_teo,k_Delta_teo,Overlapness,Iter_burn,Iter,h);
r_evo_File=fopen(r_evo_char,"w");


//We initialize the vectors that will store the network characteristics
//  k_Pairs: Degree of the 1-hyperedges
//  k_Pairs_med: Mean degree of the 1-hyperedges
//  k_Pairs_max: Maximum degree of the 1-hyperedges
//  Degree_Distribution_Pairs: Degree distribution of the 1-hyperedges
//  V: Neighbourlist of the 1-hyperedges
//  k_Trios: Degree of the 2-hyperedges
//  k_Trios_med: Mean degree of the 2-hyperedges
//  k_Trios_max: Maximum degree of the 2-hyperedges
//  Degree_Distribution_Trios: Degree distribution of the 2-hyperedges
//  T_1: First neighbour of the 2-hyperedges
//  T_2: Second neighbour of the 2-hyperedges

int *k_Pairs, *Degree_Distribution_Pairs, *V, *k_Trios, *Degree_Distribution_Trios, *T_1, *T_2;
double k_Pairs_med, k_Trios_med;
int k_Pairs_max, k_Trios_max;

//We read the hyeprgraph
Read_Links(N, Network_Type, k_Pairs_teo, k_Delta_teo, &k_Pairs, &V, &k_Pairs_max);
Degree_Distribution(N, k_Pairs_max, k_Pairs, k_Pairs_teo, k_Delta_teo, &k_Pairs_med, &Degree_Distribution_Pairs);
Read_Triangles(N, Network_Type, k_Pairs_teo, k_Delta_teo, Overlapness, &k_Trios, &T_1, &T_2, &k_Trios_max);
Degree_Distribution_T(N, k_Pairs_teo, k_Delta_teo, k_Trios, k_Trios_max, &k_Trios_med, &Degree_Distribution_Trios);

//We initialize the dynamical variables
double *natural_w, *w_eff;
double *theta_mem;
Initialize_vectors(N, Iter, &natural_w, &w_eff, &theta_mem);
Initial_Conditions(N, natural_w, theta_mem, freq_factor, phase_factor);

//We integrate the dynamics
int t, iii;
double raux=0, rlink=0, ror=0;

//Burn-in time
for(t=0;t<Iter_burn;t++){
    Evolution(N, t, h, theta_mem, w_eff, natural_w, T_1, T_2, k_Pairs, k_Trios, V, sigma_P, sigma_T, k_Pairs_med, k_Trios_med);
}

//Observable computation
for(t=Iter_burn;t<Iter-1;t++){
    Evolution(N, t, h, theta_mem, w_eff, natural_w, T_1, T_2, k_Pairs, k_Trios, V, sigma_P, sigma_T, k_Pairs_med, k_Trios_med);
    ror=r_order(N, t, theta_mem);
    raux+=ror;
    fprintf(r_evo_File,"%d\t%.4lf\n",t,ror);
}
raux/=(Iter-Iter_burn-1);
rlink=r_link(N, Iter, Iter_burn, theta_mem, k_Pairs, k_Trios, V, T_1, T_2);

//We store the results
fprintf(r_stationary_File,"%.4lf\t%.4lf\t%.4lf\n",sigma_P,raux,rlink);
for(iii=0;iii<N;iii++){
    fprintf(w_eff_File,"%.4lf\t%.4lf\n",sigma_P,w_eff[iii]/((Iter)*h));
}

//We free the memory
fclose(r_evo_File);
fclose(r_stationary_File);
fclose(w_eff_File);
Free_vectors(natural_w, w_eff, theta_mem);

printf("\n\nEND SIMULATION\n");
return 0;
}
