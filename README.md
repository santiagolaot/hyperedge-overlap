# Hyperedge overlap drives explosive transitions in systems with higher-order interactions

This code files contains the core code to perform the numerical simulations in the following papers:

- F. Malizia, S. Lamata-Otín, M. Frasca, V. Latora & J. Gómez-Gardeñes. [Hyperedge overlap drives explosive transitions in systems with higher-order interactions](https://www.nature.com/articles/s41467-024-55506-1), *Nature Communications* **16**, 555 (2025)

- S. Lamata-Otín, F. Malizia, V. Latora, M. Frasca & J. Gómez-Gardeñes. Hyperedge overlap drives synchronizability of systems with higher-order interactions, *Physical Review E* (accepted, 2025)

  
# Abstract

#### *Nature Communications* **16**, 555 (2025)

Recent studies have shown that novel collective behaviors emerge in complex systems due to the presence of higher-order interactions. However, how the collective behavior of a system is influenced by the microscopic organization of its higher-order interactions is not fully understood. In this work,  we introduce a way to quantify the overlap among the hyperedges of a higher-order network, and we show that real-world systems exhibit  different levels of intra-order hyperedge overlap. We then study two types of dynamical processes on higher-order networks, namely complex contagion and synchronization, finding that intra-order hyperedge overlap plays a universal role in determining the collective behavior in a variety of systems. Our results demonstrate that the presence of higher-order interactions alone does not guarantee abrupt transitions. Rather, explosivity and bistability require a microscopic organization of the structure with a low value of intra-order hyperedge overlap.

#### *Physical Review E* (accepted, 2025)

The microscopic organization of dynamical systems coupled via higher-order interactions plays a pivotal role in understanding their collective behavior. In this paper, we introduce a framework for systematically investigating the impact of the interaction structure on dynamical processes. Specifically, we develop an hyperedge overlap matrix whose elements characterize the two main aspects of the microscopic organization of higher-order interactions: the inter-order hyperedge overlap (non-diagonal matrix elements) and the intra-order hyperedge overlap (encapsulated in the diagonal elements). This way, the first set of terms quantifies the extent of superposition of nodes among hyperedges of different orders, while the second focuses on the number of nodes in common between hyperedges of the same order. Our findings indicate that large values of both types of hyperedge overlap hinder synchronization stability, and that the larger is the order of interactions involved, the more important is their role. Our findings also indicate that the two types of overlap have qualitatively distinct effects on the dynamics of coupled chaotic oscillators. In particular, large values of intra-order hyperedge overlap hamper synchronization by favouring the presence of disconnected sets of hyperedges, while large values of inter-order hyperedge overlap hinder synchronization by increasing the number of shared nodes between groups converging on different trajectories, without necessarily causing disconnected sets of hyperedges.

# Contents of the repository

#### Hyperedge overlap

This folder contains the code to compute the metric and generate the structures appearing in *Nature Communications* **16**, 555 (2025).

#### HO-SIS

This folder contains the code to perform the numerical simulations of higher-order SIS dynamics appearing in *Nature Communications* **16**, 555 (2025).

#### HO-Kuramoto 

This folder contains the code to integrate the higher-order Kuramoto dynamics appearing in *Nature Communications* **16**, 555 (2025).

#### Hyperedge overlap matrix

This folder contains the code to compute the overlap matrix introduced in *Physical Review E* (Accepted)

#### Structure generation

This folder contains the code to generate structures with maximum overlap and then minimize the value to cover the whole spectrum of structures utilized in *Physical Review E* (Accepted)

#### HO-Chaotic-Oscillators

This folder contains the code to integrate the chaotic oscillator dynamics and to perform the stability analysis appearing in *Physical Review E* (Accepted)



