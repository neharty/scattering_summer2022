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
        fftw_plan fdp; 
        fftw_plan fodp;
        fftw_plan bwdp;
        fftw_plan bwodp;
        int nmode; 
        int ncyl;
        
        int Nd;
        int Nod;
        vector<complex<double>> xddum; // dummy arrays for fftw plan
        vector<complex<double>> yddum;
        vector<complex<double>> xoddum;
        vector<complex<double>> yoddum;
        bool fast_conv = true;

        void set_ST(const vector<vector<vector<complex<double>>>>& st){ST = st; cout<<ST[1][0][0]<<endl;};
        void set_JHM(const vector<vector<complex<double>>>& jhm){JHM = jhm;};
        void set_fast_conv(const bool fc){fast_conv = fc;cout << fast_conv << fc << endl;};

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
                            FFTW_ESTIMATE);          

            //  forward off diagonal plan
            fodp = fftw_plan_dft_1d(Nod, 
                            reinterpret_cast<fftw_complex*>(xoddum.data()), //assumes more than 1 circle 
                            reinterpret_cast<fftw_complex*>(yoddum.data()), 
                            FFTW_FORWARD, 
                            FFTW_ESTIMATE);

            // backwards diagonal plan
            bwdp = fftw_plan_dft_1d(Nd, 
                            reinterpret_cast<fftw_complex*>(xddum.data()), 
                            reinterpret_cast<fftw_complex*>(yddum.data()), 
                            FFTW_BACKWARD, 
                            FFTW_ESTIMATE);

           //backward odd diagonal plan
           bwodp = fftw_plan_dft_1d(Nod, 
                            reinterpret_cast<fftw_complex*>(xoddum.data()), //assumes more than 1 circle 
                            reinterpret_cast<fftw_complex*>(yoddum.data()), 
                            FFTW_BACKWARD, 
                            FFTW_ESTIMATE);
        }

        complex<double>* mvp(complex<double>* xin){
            //int len = ncyl*(2*nmode + 1);
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
            
            for(int p=0; p<ncyl; p++){
                for (int q=0; q<ncyl; q++){
                    if(fast_conv){
                        if(p==q){ SCmvp[p][q] = convolve(ST[p][q], xs[q], fdp, bwdp);}
                        else{ SCmvp[p][q] = convolve(ST[p][q], xs[q], fodp, bwodp);}
                    }
                    else{
                        if(p==q){ SCmvp[p][q] = slow_convolve(ST[p][q], xs[q]);}
                        else{ SCmvp[p][q] = slow_convolve(ST[p][q], xs[q]);}
                    }
                }
            }
            
            for(int p=0; p<ncyl; p++){
                for (int q=0; q<ncyl; q++){
                    if(p != q){
                        for(int i=0; i<SCmvp[p][q].size(); i++){
                            SCmvp[p][q][i] *= JHM[p][i];
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
            
            vector<complex<double>> outvect(maxlen-minlen + 1);
            for(int i=minlen; i<maxlen+1; i++){
                outvect[i-minlen] = resout[i];
            }
            
            return outvect;
        }
         
        vector<complex<double>> slow_convolve(vector<complex<double>> const &f, vector<complex<double>> const &g) {
            //from stack overflow
            //https://stackoverflow.com/questions/24518989/how-to-perform-1-dimensional-valid-convolution
            cout << "takin it slow" << endl;
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
        }
};
