#include <iostream>

int* get_arr(int* arr, int n){    
    for(int i = 1; i < n + 1; i++){
        arr[i-1] = i;
    }
    return arr;
}

