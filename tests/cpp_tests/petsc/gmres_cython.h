static char help[] = "Solves 2D Poisson equation using multigrid.\n\n";
#include <iostream>
#include <complex>
#include <numeric>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>

using namespace std;

template <typename T>
T* gmres_py(T* A, T* b, T* x, int dim, int restart = 30, double tol = 1e-10){
    KSP ksp; /* linear solver context */
    Mat Amat; /* matrix */
    Vec xsol, bmat; /* solution, RHS vectors */
    int n, its; /* problem dimension, number of iterations */
    MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,&A);
    MatSetFromOptions(A);
    /* (code to assemble matrix not shown) */
    VecCreate(PETSC_COMM_WORLD,&x);
    VecSetSizes(x,PETSC_DECIDE, n);
    VecSetFromOptions(x);
    VecDuplicate(x,&b);
    /* (code to assemble RHS vector not shown)*/
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A, DIFFERENT_NONZERO_PATTERN);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, b, x);
    KSPDestroy(ksp);
}
