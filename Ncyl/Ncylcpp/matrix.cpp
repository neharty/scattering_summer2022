#include <iostream>
#include <vector>
#include <random>
#include <ctime>
using namespace std;

vector<double> makevect(int n){
	vector<double> v(n);
	
	uniform_real_distribution<double> u(1, 2);
	random_device re;

	for(int i = 0; i<n; i++){
		v[i] = u(re);
	};

	return v;
}

vector<vector<double>> makematrix(int n, int m){
	vector<vector<double>> a(m);

        for(int i = 0; i < m; i++){
               a[i] = makevect(n);
        };

        return a;
}

vector<double> mul_matrix_vect(vector<vector<double>> mat, vector<double> vect){
	int n = mat.size(); // assume the dimensions work out, mat = n x m
	int m = vect.size();// vect = m x 1
	
	vector<double> res(n); //res = n x 1

	for(int i = 0; i < n; i++){
		double sum = 0.;
		for(int j = 0; j < m; j++){
			sum += mat[i][j]*vect[j];
		}
		res[i] = sum;
	}
	return res;
}

vector<vector<double>> mul_matrix_matrix(vector<vector<double>> mat1, vector<vector<double>> mat2){
	int n = mat1.size(); // assume the dimensions work out, mat = n x m
	int m = mat2[0].size();// vect = m x k
	int k = mat2.size();
	
	vector<vector<double>> res(n, vector<double> (k, 0.)); //res = n x k

	for(int i = 0; i < n; i++){
		for(int l=0; l < k; l++){
			double sum = 0.;
			for(int j = 0; j < m; j++){
				sum += mat1[i][j]*mat2[j][l];
			}
			res[i][l] = sum;
		}
	}
	return res;
}

int main(){
	cout << "matrix/vector multiplication" << endl;
	
	int n;
	cout << "1st dimension of matrix" << endl;
	cin >> n;
	
	int m;
	cout << "2nd dimension of matrix" << endl;
	cin >> m;

	vector<double> v = makevect(n);
	vector<vector<double>> a = makematrix(m, n);

	for(int i = 0; i < n; i++){
		cout << v[i] << endl;
	};

	for(int i = 0; i < n; i++){
                for(int j = 0; j < m; j++){
			cout << a[i][j] << "  ";
		}
		cout << endl;
        };
	
	vector<double> b = mul_matrix_vect(a, v);
		
	for(int i = 0; i < n; i++){
                cout << b[i] << endl;
        };

	cout << "matrix/matrix multiplication" << endl;
	
	int n;
	cout << "1st dimension of matrix" << endl;
	cin >> n;
	
	int m;
	cout << "2nd dimension of matrix" << endl;
	cin >> m;
	
	int k;
	cout << "3rd dimension of matrix" << endl;
	cin >> k;

	vector<vector<double>> a1 = makevect(n, m);
	vector<vector<double>> a2 = makematrix(m, k);

	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			cout << a1[i][j] << "  ";
		};
		cout << end;
	};

	for(int i = 0; i < m; i++){
                for(int j = 0; j < k; j++){
			cout << a2[i][j] << "  ";
		}
		cout << endl;
        };
	
	vector<vector<double>> b = mul_matrix_matrix(a1, a2);
		
	for(int i = 0; i < n; i++){
                cout << b[i] << endl;
        };

	return 0;
}


