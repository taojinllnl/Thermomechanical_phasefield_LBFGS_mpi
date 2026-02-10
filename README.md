# Thermomechanical_phasefield_LBFGS_mpi

The MPI version of the L-BFGS monolithic solver for phasefield crack modeling under thermomechanically coupled loading. 


## Purpose

Phase-field fracture models are computationally expensive because they require extremely fine meshes to resolve the regularized crack topology.  
This project aims to improve the efficiency and scalability of a monolithic thermomechanical solver for large-scale thermomechanically coupled phase-field simulations on both single machines and High-Performance Computing (HPC) clusters.

The main features include:

- Configurable execution modes: selectable via a parameter (`.prm`) configuration file, supporting both serial execution and Message Passing Interface (MPI) parallelization with non-overlapping mesh decomposition over ranks.
- Adaptive mesh refinement (AMR) with optional repartitioning: in MPI mode, dynamic repartitioning can be conducted based on a user-defined imbalance threshold (ratio between the maximum and minimum number of cells per compute rank) to maintain load balance.
- Selectable linear algebra backends: the solver infrastructure supports both *PETSc* and *Trilinos* backends for distributed vectors, matrices, and solvers through a unified wrapper interface.
- Selectable linear solvers: the *Direct sparse* or *iterative* solvers (e.g., Conjugate Gradient (CG)) can be selected for the linear algebra system.
- Quasi-Newton algorithm: Serial and MPI-enabled monolithic limited-memory BFGS (L-BFGS) method for thermomechanically coupled phase-field fracture problems.
- Historical variable: quadrature-point history field storing the maximum positive strain energy to enforce irreversibility.
- Dimension-independent implementation: the code works for both 2D and 3D simulations. 


## How to Build

This project is implemented using the deal.II finite element library and supports both serial and MPI-enabled monolithic L-BFGS finite element simulations.

### Requirements

- tested with deal.II v9.6.0 (last verified: 2026-02-10)
- C++17 compatible compiler
- deal.II configured with:
  - MPI
  - Trilinos / PETSc
  - BLAS / LAPACK
  - TBB (Threading Building Blocks)
  - UMFPACK

### Build

An out-of-source CMake build is recommended:

```bash
cmake -S . -B build
cmake --build build 
```
The executable will be generated inside the `build/` directory.
