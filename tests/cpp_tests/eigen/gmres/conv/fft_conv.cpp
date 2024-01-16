#include <iostream>
#include <complex>
#include <numeric>
#include <vector>
#include <fftw3.h>
#include <random>

using namespace std;

template<typename T> 
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
