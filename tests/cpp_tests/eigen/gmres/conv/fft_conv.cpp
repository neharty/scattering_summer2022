#include <iostream>
#include <complex>
#include <numeric>
#include <vector>
#include <fftw3.h>
#include <random>

using namespace std;

vector<complex<double>> convolve_slow(vector<complex<double>> const &f, vector<complex<double>> const &g) {
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

complex<double>* conv_slow(complex<double>* f, complex<double>* g, int fn, int gn, complex<double>* xout){
    vector<complex<double>> fv(f, f+fn);
    vector<complex<double>> gv(g, g+gn);
    vector<complex<double>> out = convolve_slow(fv, gv);
    for (int i(0); i<out.size(); i++){
        //cout << out[i] << " ";
        xout[i] = out[i];
    }
    //cout << endl << endl;
    cout << "slow return" << endl;
    return xout; 
}


vector<complex<double>> convolve_fft(vector<complex<double>>& f, vector<complex<double>>& g){
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
    int outlen = maxlen-minlen + 1;

    vector<complex<double>> outvect(outlen);
    for(int i=0; i<outlen; i++){
        outvect[i] = resout[i+minlen-1];
    }
    
    return outvect;
}

complex<double>* conv_fft(complex<double>* f, complex<double>* g, int fn, int gn, complex<double>* xout){
    vector<complex<double>> fv(f, f+fn);
    vector<complex<double>> gv(g, g+gn);
    vector<complex<double>> out = convolve_fft(fv, gv);
    for (int i(0); i<out.size(); i++){
        //cout << out[i] << " ";
        xout[i] = out[i];
    }
    //cout << endl << endl;
    cout << "fft return" << endl;
    return xout;
}
/*
    int main(){
        int m = 10;
        int n = 20;
        vector<double> f(m);
        vector<double> g(n);
        
        for (int i=0; i<f.size(); i++){ f[i] = rand();};
        for (int i=0; i<g.size(); i++){ g[i] = rand();};
        
        cout << "[";
        for(int i=0; i<f.size()-1; i++){
            cout << f[i] << ", ";
        }
        cout << f[f.size()-1] << "]" << endl;
        cout << "[";
        for(int i=0; i<g.size()-1; i++){
            cout << g[i] << ", ";
        }
        cout << g[g.size()-1] << "]" << endl;
        
        vector<complex<double>> outp = convolve(f, g);
        cout << "[";
        for(int i=0; i<outp.size()-1; i++){
            cout << outp[i] << ", ";
        }
        cout << outp[outp.size()-1] << "]" << endl;
    }
*/
