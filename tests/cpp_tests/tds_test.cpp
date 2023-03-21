#include <iostream>
#include <vector>
#include <numeric>
#include <complex>
#include <cmath>
#include <Python>
#include <Eigen/Dense>
#include <boost/math/special_functions/hankel.hpp>
#include <boost/math/special_functions/bessel.hpp>

//using namespace std;
using Eigen::MatrixXcd;
using boost::math::cyl_hankel_1; 
using boost::math::cyl_bessel_j;

int main()
{
  MatrixXcd m(2,2);
  m(0,0) = cyl_hankel_1(0.0, 1.0L);
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;

  std::vector<int> a(10);
  std::iota(a.begin(), a.end(), -3);
  std::cout<<a.size()<< ' ';
  for (int i=0; i < a.size(); i++){
    std::cout << a[i] << std::endl;
  }
  return 0;
}

std::vector<double> cart_to_polar(std::vector<double> v){
    std::vector<double> vret(std::sqrt((std::pow(v[0], 2) + std::pow(v[1], 2))), std::atan2(v[1], v[0]));   
    return vret;
}

std::vector<double> polar_to_cart(std::vector<double> v){
    std::vector<double> vret{v[0]*cos(v[1]), v[0]*sin(v[1])};
    return vret;
}

std::complex<double> phi(int n, std::vector<double> v){
    // v is in cartesian coords
    std::vector<double> vpol = cart_to_polar(v);
    double r = vpol[0];
    double theta = vpol[1];
    std::complex<double> arg(0, n*theta);
    // need to add wavenumber
    return cyl_hankel_1(n, r)*exp(arg);
}

std::complex<double> phihat(int n, std::vector<double> v){
    // v is in cartesian coords
    std::vector<double> vpol = cart_to_polar(v);
    double r = vpol[0];
    double theta = vpol[1];
    std::complex<double> arg(0, n*theta);
    //need to add wavenumber
    return cyl_bessel_j(n, r)*exp(arg);
}

std::complex<double> S(int m, int n, std::vector<double> v){
    // v is in cartesian coords
    return phi(m-n, v);
}
/*
std::vector<complex<double>> make_root_vect_1cyl(PyObject * cyl_i, PyObject * cyl_sc){
    // using indices instead of cyls here
    b = cyl_i.pos - cyl_sc.pos;
    idxa_p = np.arange(-cyl_i.sum_index, cyl_i.sum_index + 1);
    idxd_q = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index + 1)[::-1]; 
    return np.hstack(([self.S(n, idxa_p[0], b) for n in idxd_q], [self.S(idxd_q[-1], n, b) for n in idxa_p[1:]]));
}


def make_root_matrix():
    if self.ST is None:
        self.ST = np.array([[self.make_root_vect_1cyl(self.cyls[p], self.cyls[q]) for q in self.labels] for p in self.labels], dtype=object)
    return self.ST
*/

 make_mul_scattering_block_root(self, cyl_i, cyl_sc):
    rv = self.make_root_vect_1cyl(cyl_i, cyl_sc)
    Nq = cyl_sc.sum_index
    Np = cyl_i.sum_index
    col = np.flip(rv[:2*Nq+1])
    row = rv[2*Nq:]
    S = toeplitz(row, col)

    bc = cyl_i.bc
    radius = cyl_i.radius
    b = cyl_i.pos - cyl_sc.pos
    k = self.k
    lam = cyl_i.impedance

    idxa_i = np.arange(-cyl_i.sum_index, cyl_i.sum_index + 1)
    if bc == 'd':
        D = np.diag(jv(idxa_i, k*radius)/hankel1(idxa_i, k*radius))
    elif bc == 'n':
        D = np.diag(jvp(idxa_i, k*radius)/h1vp(idxa_i, k*radius))
    elif bc == 'i':
        D = np.diag((k*jvp(idxa_i, k*radius) + lam*jv(idxa_i, k*radius))/(k*h1vp(idxa_i, k*radius) + lam*hankel1(idxa_i, k*radius)))
    else:
        print('invalid bc')
    
    return D @ S

