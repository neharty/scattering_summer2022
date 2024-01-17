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

class mvp_class{
    public:
        vector<vector<vector<complex<double>>>> ST; 
        vector<vector<complex<double>>> JHM; 
        int nmode; 
        int ncyl;
         
        
        /*
        mvp_class(vector<vector<vector<complex<double>>>>& ST, vector<vector<complex<double>>>& JHM, int nmode, int ncyl){
            this->ST = ST;
            this->JHM = JHM;
            this->nmode = nmode;
            this->ncyl = ncyl;
        };
        */
        
        complex<double>* mvp(complex<double>* xin){
            int len = ncyl*(2*nmode + 1);
            vector<complex<double>> x(xin, xin+len);
            vector<vector<vector<complex<double>>>> SCmvp(ncyl, vector<vector<complex<double>>>(ncyl, vector<complex<double>>(2*nmode+1)));
            vector<vector<complex<double>>> xs(ncyl,vector<complex<double>>((2*nmode+1)));
            vector<vector<complex<double>>> mulscat(ncyl, vector<complex<double>>((2*nmode+1)));
            vector<complex<double>> retvect(len);
            

            // split up the x coeffs
            int idx;
            for(int i=0; i<ncyl; i++){
                for(int j=0; j<(2*nmode + 1); j++){
                    idx = i*(2*nmode+1) + j;
                    xs[i][j] = x[idx];
                }
            }
            
            // do the multiplication
            for(int p=0; p<ncyl; p++){
                for (int q=0; q<ncyl; q++){
                    
                    this->convolve(this->ST[p][q], xs[q]);
                    
                    if(p != q){
                        for(int i=0; i<len; i++){
                            SCmvp[p][q][i] *= this->JHM[p][i];
                        }
                    }
                    
                    for(int i=0; i<(2*nmode + 1); i++){
                        mulscat[p][i] += SCmvp[p][q][i];
                    }
                }
            }
            
            // resize the result vector
            idx=0;
            for(int p=0; p<ncyl; p++){
                for(int i=0; i<(2*nmode+1); i++){
                    idx = p*(2*nmode+1) + i;
                    retvect[idx] = mulscat[p][i];
                }
            }

            return retvect.data();
        };
 
        vector<complex<double>> convolve(vector<complex<double>>& f, vector<complex<double>>& g){
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
