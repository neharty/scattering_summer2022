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
        vector<vector<vector<complex<double>>>> ST; 
        vector<vector<complex<double>>> JHM; 
        fftw_plan fdp; 
        fftw_plan fodp;
        fftw_plan bwdp;
        fftw_plan bwodp;
        int nmode; 
        int ncyl;
        int dim = ncyl*(2*nmode+1);
        
        int Nd;
        int Nod;
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
        void set_ST(const vector<vector<vector<complex<double>>>>& st){ST = st;};
        void set_JHM(const vector<vector<complex<double>>>& jhm){JHM = jhm;};
        void set_maxiter(const int mit){maxiter=mit;};
        void set_restart(const int rsrt){restart=rsrt;};
        void set_tol(const double tl){tol=tl;};

        //getters
        int get_dim() const { return dim; }
        int get_maxiter() const { return maxiter; }
        int get_restart() const { return restart; }
        double get_tol() const { return tol; }

        void mvp_setup(){
            // makes fftw plans
            Nd = ST[0][0].size() + (2*nmode + 1) - 1;
            Nod = ST[0][1].size() + (2*nmode + 1) - 1;
            xddum.resize(Nd); // dummy arrays for plan
            yddum.resize(Nd);
            xoddum.resize(Nod);
            yoddum.resize(Nod);
   
            // forward fft diagonal plan
            fdp = fftw_plan_dft_1d(Nd, 
                            reinterpret_cast<fftw_complex*>(xddum.data()), 
                            reinterpret_cast<fftw_complex*>(yddum.data()), 
                            FFTW_FORWARD, 
                            FFTW_MEASURE);          

            //  forward off diagonal plan
            fodp = fftw_plan_dft_1d(Nod, 
                            reinterpret_cast<fftw_complex*>(xoddum.data()), //assumes more than 1 circle 
                            reinterpret_cast<fftw_complex*>(yoddum.data()), 
                            FFTW_FORWARD, 
                            FFTW_MEASURE);

            // backwards diagonal plan
            bwdp = fftw_plan_dft_1d(Nd, 
                            reinterpret_cast<fftw_complex*>(xddum.data()), 
                            reinterpret_cast<fftw_complex*>(yddum.data()), 
                            FFTW_BACKWARD, 
                            FFTW_MEASURE);

           //backward odd diagonal plan
           bwodp = fftw_plan_dft_1d(Nod, 
                            reinterpret_cast<fftw_complex*>(xoddum.data()), //assumes more than 1 circle 
                            reinterpret_cast<fftw_complex*>(yoddum.data()), 
                            FFTW_BACKWARD, 
                            FFTW_MEASURE);
           cout << "mvp initialized" << endl;
        }
        
        complex<double>* conv_ptr(complex<double>* fptr, int flen, complex<double>* gptr, int glen, complex<double>* xout){
            cout << dim << endl;
            vector<complex<double>> fv(fptr, fptr+flen);
            vector<complex<double>> gv(gptr, gptr+glen);
            vector<complex<double>> xv(max(flen, glen) - min(flen, glen) + 1);
            xv = convolve(fv, gv, fdp, bwdp);
            cout << "convolved" << endl;
            for(int i(0); i<dim; i++){
                xout[i] = xv[i];
            }
            return xout;
        }

        complex<double>* mvp_ptr(complex<double>* xin, complex<double>* xout){
            cout << dim << endl;
            vector<complex<double>> xv(xin, xin+dim);
            xv = mvp(xv);
            for(int i(0); i<dim; i++){
                xout[i] = xv[i];
            }
            return xout;
        }

        vector<complex<double>> mvp(vector<complex<double>>& xin){
            //int len = ncyl*(2*nmode + 1);
            //cout << "start mvp" << endl;
            vector<vector<vector<complex<double>>>> SCmvp(ncyl, vector<vector<complex<double>>>(ncyl, vector<complex<double>>(2*nmode+1)));
            vector<vector<complex<double>>> xs(ncyl,vector<complex<double>>((2*nmode+1)));
            vector<vector<complex<double>>> mulscat(ncyl, vector<complex<double>>((2*nmode+1)));
            
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
                    if(p==q){ SCmvp[p][q] = convolve(ST[p][q], xs[q], fdp, bwdp); }
                    else{ SCmvp[p][q] = convolve(ST[p][q], xs[q], fodp, bwodp); }
                }
            }
            
            for(int p=0; p<ncyl; p++){
                for (int q=0; q<ncyl; q++){
                     for(int i=0; i<SCmvp[p][q].size(); i++){ cout << SCmvp[p][q][i] << " ";}
                     cout << endl;
                }
                cout << endl;
            }
            
            cout << endl;
            for(int p=0; p<ncyl; p++){
                for (int q=0; q<ncyl; q++){
                    if(p != q){
                        for(int i=0; i<SCmvp[p][q].size(); i++){ SCmvp[p][q][i] *= JHM[p][i]; }
                    }
                    
                    for(int i=0; i<(2*nmode + 1); i++){ mulscat[p][i] += SCmvp[p][q][i]; }
                }
            }

            // resize the result vector
            idx=0;
            for(int p=0; p<ncyl; p++){
                for(int i=0; i<(2*nmode+1); i++){
                    idx = p*(2*nmode+1) + i;
                    xin[idx] = mulscat[p][i];
                }
            }
            return xin;
        };
 
        vector<complex<double>> convolve(vector<complex<double>>& f, vector<complex<double>>& g, const fftw_plan fp, const fftw_plan bp){
            int n = int(f.size());
            int m = int(g.size());
            const int N = n+m-1;
            // zero pad the arrays
            vector<complex<double>> fpad(N);
            vector<complex<double>> gpad(N);
            
            for(int i=0;i<n;i++){
                fpad[i] = f[i];
            }
            for(int i=0; i<m; i++){
                gpad[i] = g[i];
            }
            
            // fft to convolve the padded arrays   
            vector<complex<double>> fout(N);
            vector<complex<double>> gout(N);
            fftw_execute_dft(fp, reinterpret_cast<fftw_complex*>(fpad.data()), reinterpret_cast<fftw_complex*>(fout.data()));
            fftw_execute_dft(fp, reinterpret_cast<fftw_complex*>(gpad.data()), reinterpret_cast<fftw_complex*>(gout.data()));

            vector<complex<double>> convres(N);
            
            for(int i=0; i < N; i++){
                // complex multiply the results
                convres[i] = fout[i]*gout[i];
            }
             
            //invert to get result
            vector<complex<double>> resout(N);
            fftw_execute_dft(bp, reinterpret_cast<fftw_complex*>(convres.data()), reinterpret_cast<fftw_complex*>(resout.data()));
            // normalize data
            for(int i = 0; i < N; i++){
                resout[i] = resout[i]/double(N);
            }
            
            //select the "valid" components in the convolution
            int minlen = min(m, n);
            int maxlen = max(m, n);
            int outlen = maxlen-minlen + 1;

            vector<complex<double>> outvect(outlen);
            for(int i=0; i<outlen; i++){
                outvect[i] = resout[i+minlen-1];
            }
            
            return outvect;
            
            /*
            //from stack overflow
            //https://stackoverflow.com/questions/24518989/how-to-perform-1-dimensional-valid-convolution
            //cout << "takin it slow" << endl;
            int const nf = f.size();
            int const ng = g.size();
            vector<complex<double>> const &min_v = (nf < ng)? f : g; //min len vector
            vector<complex<double>> const &max_v = (nf < ng)? g : f; //max len vector
            int const n  = max(nf, ng) - min(nf, ng) + 1; //output vector
            vector<complex<double>> out(n);
            
            for(auto i(0); i < n; ++i) {
                for(int j(min_v.size() - 1), k(i); j >= 0; --j) {
                    out[i] += min_v[j] * max_v[k];
                    ++k;
                }
            }
            return out;
            */
        }

        void fftw_clean(){ 
            // to be implemented
        }
        
        template<typename Rhs>
        Eigen::Product<MatReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
            //cout << "product defn" << endl;
            return Eigen::Product<MatReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
        }
};

template<typename Rhs>
struct internal::generic_product_impl<MatReplacement, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
: internal::generic_product_impl_base<MatReplacement, Rhs, generic_product_impl<MatReplacement, Rhs> >{
    typedef typename Product<MatReplacement,Rhs>::Scalar Scalar;

    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const MatReplacement& lhs, const Rhs& rhs, const Scalar& alpha)
    {
        //assert(alpha==Scalar(1) && "scaling is not implemented");
        //EIGEN_ONLY_USED_FOR_DEBUG(alpha);
        
        vector<complex<double>> rhsin(rhs.data(), rhs.data() + rhs.rows() * rhs.cols());        
        MatReplacement tmp = lhs; //this line is wrong
        vector<complex<double>> prodout = tmp.mvp(rhsin);
        dst += Map<Vector<complex<double>, Dynamic>>(prodout.data(), prodout.size());
    }
};

complex<double>* gmres_cpp(complex<double>* xin, complex<double>* xout, const MatReplacement& M){
    GMRES<MatReplacement, IdentityPreconditioner> gmres;
    gmres.set_restart(M.get_restart());
    gmres.setTolerance(M.get_tol());
    gmres.setMaxIterations(M.get_maxiter());
    gmres.compute(M);
    int N = M.get_dim(); 
    Vector<complex<double>, Dynamic> xingmres = Map<Vector<complex<double>, Dynamic> >(xin, N);
    Vector<complex<double>, Dynamic> xoutgmres = gmres.solve(xingmres);
    cout << "exit info:   "<< gmres.info() << endl;
    cout << "#iterations:     " << gmres.iterations() << endl;
    cout << "estimated error: " << gmres.error()      << endl;
    //vector<complex<double>> xout(&xoutgmres[0], xoutgmres.data()+xoutgmres.cols()*xoutgmres.rows());
    
    //copy the data to the original array
    for(int i=0; i < N; i++){
        xout[i] = xoutgmres[i];
        //std::cout << xsol[i] << std::endl;
    }

    return xout;
}
