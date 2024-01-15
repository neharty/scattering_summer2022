#include <iostream>
#include <complex>
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <complex>

//TODO: change to complex data

using namespace std;
using namespace Eigen;

/*
 this algorithm is as follows:
    *take in C arrays for matrix and rhs vector
    *conver to Eigen types using Map
    *apply GMRES
    *return as C arrays
*/
/*
template<typename T>
Vector<T, Dynamic> mvp(Vector<T, Dynamic>& x, Matrix<T, Dynamic, Dynamic>& ST, Matrix<T, Dynamic, Dynamic>& JHM, int nmode, int ncyl){
    int len = ncyl*(2*nmode + 1);
    x = np.array(x, dtype=np.complex128)

    #split up the x coeffs
    xs = [
        np.array(
            x[self.lengths[i-1]:self.lengths[i]], dtype=np.complex128
            ) for i in range(1, len(self.lengths))
        ]
    ST = self.ST

    mulscat = [1j*np.zeros(len(xn)) for xn in xs]

    #computes the block MVPs and stores in a multidimensional list
    SCmvp = [
        [
            convolve(
                np.array(ST[p][q], dtype=np.complex128),
                np.array(xs[q], dtype=np.complex128),
                mode='valid'
            ) for q in self.labels]
        for p in self.labels]
    
    for(int p=0; p<ncyl; p++){
        for (int q=0; q<ncyl; q++){
            // convolve using FFTW
            if(p != q){
                for(int i=0; i<len; i++){
                    // SCmvp[p,q] *= JHM[p][i]
                }
            // mulscat[p] += SCmvp[p][q]
            }
        }
    }
}
*/

template<typename T>
Vector<T, Dynamic> gmres_eigen(Matrix<T, Dynamic, Dynamic>& A, Vector<T, Dynamic>& b, int restart = 30, double tol = 1e-10, int maxiter = 0){
    // set up gmres
    GMRES<Matrix<T, Dynamic, Dynamic>, IdentityPreconditioner> gmres;
    gmres.set_restart(restart);
    gmres.setTolerance(tol);
    gmres.setMaxIterations(maxiter);
    gmres.compute(A);
    Vector<T, Dynamic> x = gmres.solve(b);
    cout << "exit info:   "<< gmres.info() << endl;
    cout << "#iterations:     " << gmres.iterations() << endl;
    cout << "estimated error: " << gmres.error()      << endl;
    //std::cout << x << std::endl;
    return x;
}

template <typename T>
T* gmres_py(T* A, T* b, T* x, int dim, int restart = 30, double tol = 1e-10){
    // convert C arrays to matrices and vectors
    Matrix<T, Dynamic, Dynamic> Amat = Map<Matrix<T, Dynamic, Dynamic, RowMajor> >(A, dim, dim);
    Vector<T, Dynamic> bvect = Map<Vector<T, Dynamic> >(b, dim);
    //std::cout << Amat << '\n' << bvect << '\n' << std::endl;
    //solve using gmres
    Vector<T, Dynamic> xsol = gmres_eigen(Amat, bvect, restart, tol, dim);
    //std::cout << xsol << std::endl;
    //copy the data to the original array
    for(int i=0; i < dim; i++){
        x[i] = xsol[i];
        //std::cout << xsol[i] << std::endl;
    }
    
    return x;
}

