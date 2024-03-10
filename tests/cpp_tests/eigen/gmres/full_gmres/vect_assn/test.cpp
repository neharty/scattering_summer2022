#include <iomanip>                                                                                                                                                                                                                            
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

int len = 5;
int height = 3;

vector<int> mulby2(vector<int>& in){
    vector<int> out(len);
    for(int i=0; i<len; i++){ out[i] = 2*in[i]; }
    return out;
}

int main(){
    // create a 2d vector, multiply the entries by 2, then assigns the output to output vects

    //define output vectors
    vector<vector<int>> outvect(height, vector<int>(len));
    vector<vector<int>> outvect2(len, vector<int>(len));

    // initialize the input vectors
    vector<vector<int>> invect(height, vector<int>(len));
    vector<vector<int>> invect2(height, vector<int>(len));

    int iter = 0;
    for(int i=0; i<height;i++){
        for(int j=0; j<len; j++){
            invect[i][j] = iter;
            invect2[i][j] = iter++;
        }
    }
    
    //use a temp vector and external function
    vector<int> tmp;
    for(int j=0; j<height; j++){
        tmp = mulby2(invect[j]);
        for(int i=0; i<len; i++){
            outvect[j][i] = tmp[i];
        }
    }

    //do the computation directly
    for(int i = 0; i<height; i++){
        for(int j=0; j<len; j++){
            outvect2[i][j] = 2*invect2[i][j];
         }
     }
    

    //print the results
    for(int i = 0; i<height; i++){
        for(int j=0; j<len; j++){
            cout << outvect[i][j] << " ";
         }
         cout << endl;
     }
    
    for(int i = 0; i<height; i++){
        for(int j=0; j<len; j++){
            cout << outvect2[i][j] << " ";
         }
         cout << endl;
     }


    return 0;
}

