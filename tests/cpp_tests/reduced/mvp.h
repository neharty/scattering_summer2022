#include <iomanip>
#include <iostream>
#include <complex>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <fftw3.h>

using namespace std;
using namespace Eigen;

class MatReplacement;

namespace Eigen{
    namespace internal{
        template<>
        struct traits<MatReplacement> :  public Eigen::internal::traits<SparseMatrix<complex<double>>> {};
    }
}

class MatReplacement : public Eigen::EigenBase<MatReplacement> {
    public:
        vector<vector<complex<double>>> STin;
        vector<vector<vector<complex<double>>>> ST; 
        vector<vector<complex<double>>> JM; 
        vector<vector<complex<double>>> HM; 

        fftw_plan fdp; 
        fftw_plan fodp;
        fftw_plan bwdp;
        fftw_plan bwodp;
        int nmode=0; 
        int ncyl=0;
        int dim=0;
        
        MatReplacement() {} // default constructor
        
        MatReplacement(int nm, int nc){
            nmode = nm; 
            ncyl = nc; 
            dim = nc*((2*nm)+1);
        }

        int Nd;
        int Nod;
        int N;
        vector<complex<double>> xddum; // dummy arrays for fftw plan
        vector<complex<double>> yddum;
        vector<complex<double>> xoddum;
        vector<complex<double>> yoddum;
        
        // gmres parameters
        int maxiter=0;
        int restart=30;
        double tol=1e-10;

        // Required typedefs, constants, and method:
        typedef double Scalar;
        typedef double RealScalar;
        typedef int StorageIndex;
        enum {
            ColsAtCompileTime = Dynamic,
            MaxColsAtCompileTime = Dynamic,
            IsRowMajor = false
        };

        Index rows() const { return dim; }
        Index cols() const { return dim; }
        
        //setters
        void set_STin(complex<double> *st){
            vector<complex<double>> tmp(st, st + 4*nmode+1);
            this->STin.push_back(tmp);
        };

        void set_ST(){
            vector<vector<complex<double>>> tmp;
            for (int p(0); p<ncyl; p++){
                for (int q(0); q<ncyl; q++){
                    tmp.push_back(this->STin[p*ncyl+q]);
                }
                this->ST.push_back(tmp);
                tmp.clear();
            }
        };
        
        void set_JM(const complex<double> *jm){
            vector<complex<double>> tmp(jm, jm+2*nmode+1);
            this->JM.push_back(tmp);
        };
        

        void set_HM(const complex<double> *hm){
            vector<complex<double>> tmp(hm, hm+2*nmode+1);
            this->HM.push_back(tmp);
        };

        void set_maxiter(const int mit){this->maxiter=mit;};
        void set_restart(const int rsrt){this->restart=rsrt;};
        void set_tol(const double tl){this->tol=tl;};

        //getters
        int get_dim() const { return dim; }
        int get_maxiter() const { return maxiter; }
        int get_restart() const { return restart; }
        double get_tol() const { return tol; }

        void mvp_setup(){
            // makes fftw plans
            N = 4*nmode+1;
            xddum.resize(N); // dummy arrays for plan
            yddum.resize(N);
            xoddum.resize(N);
            yoddum.resize(N);

            // forward fft diagonal plan
            fdp = fftw_plan_dft_1d(N, 
                            reinterpret_cast<fftw_complex*>(xddum.data()), 
                            reinterpret_cast<fftw_complex*>(yddum.data()), 
                            FFTW_FORWARD, 
                            FFTW_MEASURE);          

            //  forward off diagonal plan
            fodp = fftw_plan_dft_1d(N, 
                            reinterpret_cast<fftw_complex*>(xoddum.data()), //assumes more than 1 circle 
                            reinterpret_cast<fftw_complex*>(yoddum.data()), 
                            FFTW_FORWARD, 
                            FFTW_MEASURE);

            // backwards diagonal plan
            bwdp = fftw_plan_dft_1d(N, 
                            reinterpret_cast<fftw_complex*>(xddum.data()), 
                            reinterpret_cast<fftw_complex*>(yddum.data()), 
                            FFTW_BACKWARD, 
                            FFTW_MEASURE);

           //backward odd diagonal plan
           bwodp = fftw_plan_dft_1d(N, 
                            reinterpret_cast<fftw_complex*>(xoddum.data()), //assumes more than 1 circle 
                            reinterpret_cast<fftw_complex*>(yoddum.data()), 
                            FFTW_BACKWARD, 
                            FFTW_MEASURE);
           cout << "mvp initialized" << endl;
        }
        
        void conv_ptr(const complex<double>* fptr, const int flen, const complex<double>* gptr, const int glen, complex<double>* xout){
            // convolution algorithm with pointers
            vector<complex<double>> fv(fptr, fptr+flen);
            vector<complex<double>> gv(gptr, gptr+glen);
            vector<complex<double>> xv(max(flen, glen) - min(flen, glen) + 1);
            xv = convolve(fv, gv, fdp, bwdp);
            for(int i(0); i<dim; i++){
                xout[i] = xv[i];
            }
        }

        void mvp_ptr(const complex<double>* xin, complex<double>* xout){
            // mvp algorithm with pointers
            vector<complex<double>> xv(xin, xin+dim);
            vector<complex<double>> xvout(dim);
            xvout = mvp(xv);
            for(int i(0); i<dim; i++){
                xout[i] = xvout[i];
            }
        }
        
        void print_ST(){
            for(int i(0); i < ncyl; i++){
                for(int j(0); j < ncyl; j++){
                    cout << "ST[" << i << "][" << j << "]" << endl;
                    for(int k(0); k < 4*nmode+1; k++){ cout << setprecision(15) << ST[i][j][k] << " "; }
                cout << endl << endl;
                }
            }
        }

        void print_JM(){  
            for(int i(0); i < ncyl; i++){
                cout << "JM[" << i << "]" << endl;
                for(int j(0); j < 2*nmode+1; j++){ cout << setprecision(15) << JM[i][j] << " "; }
                cout << endl << endl;
            }
        }
        
        void print_HM(){  
            for(int i(0); i < ncyl; i++){
                cout << "HM[" << i << "]" << endl;
                for(int j(0); j < 2*nmode+1; j++){ cout << setprecision(15) << HM[i][j] << " "; }
                cout << endl << endl;
            }
        }
        
        vector<complex<double>> mvp(vector<complex<double>>& xin) const {
            vector<vector<vector<complex<double>>>> SCmvp(ncyl, vector<vector<complex<double>>>(ncyl, vector<complex<double>>(2*nmode+1, 0)));
            vector<vector<complex<double>>> xs(ncyl,vector<complex<double>>((2*nmode+1), 0));
            vector<vector<complex<double>>> mulscat(ncyl, vector<complex<double>>((2*nmode+1), 0));
            vector<complex<double>> tmp;
            vector<complex<double>> tmp2(2*nmode+1);
            vector<complex<double>> col(4*nmode+1);
            vector<complex<double>> row(4*nmode+1);

            // split up the x coeffs
            int idx;
            for(int i=0; i<ncyl; i++){
                for(int j=0; j<(2*nmode + 1); j++){
                    idx = i*(2*nmode+1) + j;
                    xs[i][j] = xin[idx];
                }
            }
            
            //matrix multiply the toeplitz part by convolution
            for(int p=0; p<ncyl; p++){
                for (int q=0; q<ncyl; q++){
                    if(p == q){
                        for(int i=0; i<2*nmode+1; i++){ SCmvp[p][q][i] = xs[p][i]; }
                    }else{
                        for(int i=0; i<(2*nmode+1); i++){ tmp2[i] = xs[q][i]/HM[q][i]; }
                        reverse_copy(begin(ST[p][q]), begin(ST[p][q]) + (2*nmode+1), begin(col));
                        //cout << "col" << endl;

                        copy(begin(ST[p][q])+(2*nmode), end(ST[p][q]), begin(row)); 
                        //cout << "row" << endl;

                        tmp = toeplitz_mvp(row, col, tmp2, fdp, bwdp);
                        //cout << "mvp" << endl;
                        for(int i=0; i<2*nmode+1; i++){ SCmvp[p][q][i] = tmp[i]; }
                    }
                }
            }
            
            
            for(int p=0; p<ncyl; p++){
                for (int q=0; q<ncyl; q++){
                    for(int i=0; i<int(2*nmode+1); i++){
                        if(p != q){ SCmvp[p][q][i] *= JM[p][i]; }
                        mulscat[p][i] += SCmvp[p][q][i];
                    }
                }
            } 

            // resize the result vector
            vector<complex<double>> xout(dim, 0);
            idx=0;
            for(int p=0; p<ncyl; p++){
                for(int i=0; i<(2*nmode+1); i++){
                    idx = p*(2*nmode+1) + i;
                    xout[idx] = mulscat[p][i];
                }
            }
             
            return xout;
        };
        
        vector<complex<double>> toeplitz_mvp(const vector<complex<double>>& c, const vector<complex<double>>& r, const vector<complex<double>>& x, const fftw_plan& fp, const fftw_plan& bp) const {
            vector<complex<double>> rnew(2*x.size());
            vector<complex<double>> xnew(2*x.size());

            for(int i=0; i<x.size(); i++){
                rnew[i] = c[i];
                xnew[i] = x[i];
            }
            
            vector<complex<double>> rflip(r.rbegin(), r.rend());
            for(int i=1; i<x.size(); i++){
                rnew[i+x.size()-1] = rflip[i];
            }

            vector<complex<double>> tmp = convolve(rnew, xnew, fp, bp);
            vector<complex<double>> retv(x.size());
            for(int i=0; i<x.size(); i++){
                retv[i] = tmp[i];
            }
            
            return retv;
        }

        vector<complex<double>> convolve(const vector<complex<double>>& f, const vector<complex<double>>& g, const fftw_plan& fp, const fftw_plan& bp) const {
            
            int n = int(f.size());
            vector<complex<double>> fin = f;
            vector<complex<double>> gin = g;

            // these should be the same size
            //int m = int(g.size());
                        
            // fft to convolve the padded arrays   
            vector<complex<double>> fout(n);
            vector<complex<double>> gout(n);
            fftw_execute_dft(fp, reinterpret_cast<fftw_complex*>(fin.data()), reinterpret_cast<fftw_complex*>(fout.data()));
            fftw_execute_dft(fp, reinterpret_cast<fftw_complex*>(gin.data()), reinterpret_cast<fftw_complex*>(gout.data()));

            vector<complex<double>> convres(n);
            for(int i=0; i < n; i++){ convres[i] = fout[i]*gout[i]; } // complex multiply the results

            //invert to get result
            vector<complex<double>> resout(n);
            fftw_execute_dft(bp, reinterpret_cast<fftw_complex*>(convres.data()), reinterpret_cast<fftw_complex*>(resout.data()));
            // normalize data
            for(int i = 0; i < n; i++){ resout[i] /= double(n); }
                       
            return resout;
        }

        void fftw_clean(){
            //deallocate fftw pointers
            fftw_destroy_plan(fdp);
            fftw_destroy_plan(fodp);
            fftw_destroy_plan(bwdp);
            fftw_destroy_plan(bwodp);
            fftw_cleanup();
        }
        
        template<typename Rhs>
        Eigen::Product<MatReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
            return Eigen::Product<MatReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
        }
};

//implementation of matrix-free MVP to be called by eigen
template<typename Rhs>
struct internal::generic_product_impl<MatReplacement, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
: internal::generic_product_impl_base<MatReplacement, Rhs, generic_product_impl<MatReplacement, Rhs> >{
    typedef typename Product<MatReplacement,Rhs>::Scalar Scalar;

    template<typename Dest>
    static void evalTo(Dest& dst, const MatReplacement& lhs, const Rhs& rhs)//scaleAndAddTo(Dest& dst, const MatReplacement& lhs, const Rhs& rhs, const Scalar& alpha)
    {
        //assert(alpha==Scalar(1) && "scaling is not implemented");
        //EIGEN_ONLY_USED_FOR_DEBUG(alpha);
        
        vector<complex<double>> rhsin(rhs.data(), rhs.data() + rhs.rows() * rhs.cols());
        vector<complex<double>> prodout = lhs.mvp(rhsin);
        dst = Map<Vector<complex<double>, Dynamic>>(prodout.data(), prodout.size());
    }
};

void gmres_cpp(complex<double>* xin, complex<double>* xout, const MatReplacement& M){
    //set up gmres
    GMRES<MatReplacement, IdentityPreconditioner> gmres;
    gmres.set_restart(M.get_restart());
    gmres.setTolerance(M.get_tol());
    gmres.setMaxIterations(M.get_maxiter());
    gmres.compute(M);
    int N = M.get_dim();
    
    // change data types to eigen vector
    Vector<complex<double>, Dynamic> xingmres = Map<Vector<complex<double>, Dynamic> >(xin, N);
    Vector<complex<double>, Dynamic> xoutgmres = gmres.solve(xingmres); // solve the system

    //print gmres info
    cout << "exit info:   "<< gmres.info() << endl;
    cout << "#iterations:     " << gmres.iterations() << endl;
    cout << "estimated error: " << gmres.error()      << endl;
    
    //copy the data to the original array
    for(int i=0; i < N; i++){
        xout[i] = xoutgmres[i];
        //std::cout << xsol[i] << std::endl;
    }
}
