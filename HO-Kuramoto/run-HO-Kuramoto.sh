#!/bin/bash

# logfile to follow the run
exec >logfile_run-HO-Kuramoto.txt 2>&1

# We compile the C code
gcc -o HO-Kuramoto HO-Kuramoto.c -O3 -march=native -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed. Check the logfile for details."
    exit 1
fi

# We define variables
Network_Type="Regular_Hg"
N=1000
k_Pairs_teo=5
k_Delta_teo=6
Intra_order_hyperedge_overlap=1
Iter_burn=8000
Iter=10000
h=0.01
sigma_T=3
sigma_P=8
phase_factor=1
freq_factor=1

# We print parameters of the run
echo "($(date '+%Y-%m-%d %H:%M:%S')) Parameters:"
echo "Network_Type=$Network_Type"
echo "N=$N, k_Pairs_teo=$k_Pairs_teo, k_Delta_teo=$k_Delta_teo, Intra_order_hyperedge_overlap=$Intra_order_hyperedge_overlap"
echo "Iter_burn=$Iter_burn, Iter=$Iter, h=$h, sigma_T=$sigma_T, sigma_P=$sigma_P"
echo "phase_factor=$phase_factor, freq_factor=$freq_factor"

# We run the program with arguments
./HO-Kuramoto "$Network_Type" "$N" "$k_Pairs_teo" "$k_Delta_teo" \
"$Intra_order_hyperedge_overlap" "$Iter_burn" "$Iter" "$h" "$sigma_T" "$sigma_P" \
"$phase_factor" "$freq_factor" &

# Inform the user
if [ $? -eq 0 ]; then
    echo "Program launched successfully."
else
    echo "Program failed to launch. Check the logfile for details."
    exit 1
fi
