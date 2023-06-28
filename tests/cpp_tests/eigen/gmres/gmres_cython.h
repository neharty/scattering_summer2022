#include <iostream>
#include <complex>
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <complex>

//TODO: change to complex data

using Eigen::SparseMatrix;
using Eigen::MatrixXcd;
using Eigen::Matrix;
using Eigen::Vector;
using Eigen::Dynamic;
using Eigen::VectorXcd;
using Eigen::GMRES;
using Eigen::Map;

/*
 algorithm is as follows:
    *take in C arrays for matrix and rhs vector
    *conver to Eigen types using Map
    *apply GMRES
    *return as C arrays
*/
template<typename T>
Vector<T, Dynamic> gmres_eigen(Matrix<T, Dynamic, Dynamic> A, Vector<T, Dynamic> b, int restart = 30, double tol = 1e-10, int maxiter = 0){
    // set up gmres
    GMRES<Matrix<T, Dynamic, Dynamic> > gmres;
    gmres.set_restart(restart);
    gmres.setTolerance(tol);
    gmres.setMaxIterations(maxiter);
    gmres.compute(A);
    Vector<T, Dynamic> x = gmres.solve(b);
    //std::cout << A << std::endl;
    //std::cout << x << std::endl;
    std::cout << "exit info:   "<< gmres.info() << std::endl;
    std::cout << "#iterations:     " << gmres.iterations() << std::endl;
    std::cout << "estimated error: " << gmres.error()      << std::endl;
    //std::cout << x << std::endl;
    return x;
}

template <typename T>
T* gmres_py(T* A, T* b, T* x, int dim, int restart = 30, double tol = 1e-10){
    // convert C arrays to matrices and vectors
    Matrix<T, Dynamic, Dynamic> Amat = Map<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor> >(A, dim, dim);
    Vector<T, Dynamic> bvect = Map<Vector<T, Dynamic> >(b, dim);
    //solve using gmres
    Vector<T, Dynamic> xsol = gmres_eigen(Amat, bvect, restart, tol, dim);
    //copy the data to the original array
    for(int i=0; i < dim; i++){
        x[i] = xsol[i];
    }
    return x;
}

