#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

template<typename T>
T* vect_to_carr(std::vector<T> v){
    T* ptr = &v[0];
    return ptr;
}

template<typename T>
std::vector<T> carr_to_vect(T* arr, int l){
    std::vector<T> v(arr, arr + l);
    return v;
}

//an inefficient way to build a circulant matrix
template <typename T> 
std::vector<std::vector<T>> circulant(std::vector<T> v){
    int n = v.size();
    T vtmp;
    std::vector<std::vector<T>> mat;
    for(int i=0; i < n; i++){
        mat.push_back(v);
        vtmp = v[n-1];
        v.pop_back();
        v.insert(v.begin(), vtmp);
    }
    return mat;
}

template<typename T>
void print(std::vector<T> v){
    int sz = v.size();
    for(int i=0; i < sz; i++){
        for(int j=0; j < sz; j++){
            std::cout << v[i][j] << ' ';
        }
        std::cout << std::endl;
    } 
}

int main(){
    // vector version
    std::vector<int> v = {1,2,3,4,5};
    std::vector<std::vector<int>> m = circulant(v);
    print(m);
    
    std::cout << std::endl;

    // start with array, then convert to vector
    int* arr = &v[0];
    std::vector<int> vout = carr_to_vect(arr, 5); 
    print(circulant(vout));
    
    //change a vector to an array
    int * ptr[5];
    for(int i = 0; i < 5; i++){
        ptr[i] = &m[i][0];
    }
    
    std::cout << std::endl;
   
    for(int i=0; i < 5; i++){
        int* p = ptr[i];
        for(int j=0; j < 5; j++){
            std::cout << *p << ' ';
            p = p + 1;
        }
        std::cout << std::endl;
    } 


    return 0;
}
