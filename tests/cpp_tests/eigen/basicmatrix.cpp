#include <iostream>
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include <boost/math/special_functions/hankel.hpp>

using Eigen::MatrixXcd;
using boost::math::cyl_hankel_1; 

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
