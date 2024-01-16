#include <iostream>
#include <complex>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <fftw3.h>
#include <random>

using namespace std;
using namespace Eigen;

template<typename T>
class mvp_class{
    public:
        vector<vector<vector<T>>> ST; 
        vector<vector<T>> JHM; 
        int nmode; 
        int ncyl;
         
        void mvp_class(vector<vector<vector<T>>>& ST, vector<vector<T>>& JHM){
            this->ST = ST;
            this->JHM = JHM;
        };
        
        Vector<T, Dynamic> mvp(Vector<T, Dynamic>& x){
            int len = ncyl*(2*nmode + 1);
            x = np.array(x, dtype=np.complex128)
            vector<vector<vector<T>>> SCmvp(ncyl, ncyl, (2*nmode+1));
            vecot<vector<T>> xs(ncyl,vector<int>((2*nmode+1)));
            vector<vector<T>> mulscat(
            vector<T> retvect(len);
            #split up the x coeffs
            xs = [
                np.array(
                    x[self.lengths[i-1]:self.lengths[i]], dtype=np.complex128
                    ) for i in range(1, len(self.lengths))
                ]
            ST = self.ST

            mulscat = [1j*np.zeros(len(xn)) for xn in xs]
            
            /*
            #computes the block MVPs and stores in a multidimensional list
            SCmvp = [
                [
                    convolve(
                        np.array(ST[p][q], dtype=np.complex128),
                        np.array(xs[q], dtype=np.complex128),
                        mode='valid'
                    ) for q in self.labels]
                for p in self.labels]
            */

            for(int p=0; p<ncyl; p++){
                for (int q=0; q<ncyl; q++){
                    this->convolve(this->ST[p][q], xs[q]);
                    if(p != q){
                        for(int i=0; i<len; i++){
                            SCmvp[p][q][i] *= this->JHM[p][i];
                        }
                    // mulscat[p] += SCmvp[p][q]
                    }
                }
            }
            
            int idx;
            for(int p=0; p<ncyl; p++){
                for(int i=0; i<(2*nmode+1); i++){
                    idx = p*(2*nmode+1) + i;
                    retvect[idx] = mulscat[p][i];
                }
            }
            return retvect;
        };
 
        vector<complex<double>> convolve(vector<T>& f, vector<T>& g){
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
            fftw_plan fp = fftw_plan_dft_1d(N, 
                            reinterpret_cast<fftw_complex*>(fpad.data()), 
                            reinterpret_cast<fftw_complex*>(fout.data()), 
                            FFTW_FORWARD, 
                            FFTW_ESTIMATE);
                            
            fftw_plan gp = fftw_plan_dft_1d(N, 
                            reinterpret_cast<fftw_complex*>(gpad.data()), 
                            reinterpret_cast<fftw_complex*>(gout.data()), 
                            FFTW_FORWARD, 
                            FFTW_ESTIMATE);
            
            fftw_execute(fp);
            fftw_execute(gp);

            vector<complex<double>> convres(N);
            
            for(int i=0; i < N; i++){
                // complex multiply the results
                convres[i] = fout[i]*gout[i];
            }
            
            fftw_destroy_plan(fp);
            fftw_destroy_plan(gp);
            
            //invert to get result
            vector<complex<double>> resout(N);
            fftw_plan p = fftw_plan_dft_1d(N, 
                            reinterpret_cast<fftw_complex*>(convres.data()), 
                            reinterpret_cast<fftw_complex*>(resout.data()), 
                            FFTW_BACKWARD, 
                            FFTW_ESTIMATE);
            fftw_execute(p);
            
            // normalize data
            for(int i = 0; i < N; i++){
                resout[i] = resout[i]/double(N);
            }
            
            //select the "valid" components in the convolution
            int minlen = min(m, n);
            int maxlen = max(m, n);
            
            vector<complex<double>> outvect(maxlen-minlen + 1);
            for(int i=minlen; i<maxlen+1; i++){
                outvect[i-minlen] = resout[i-1];
            }
            
            return outvect;
        }
};
