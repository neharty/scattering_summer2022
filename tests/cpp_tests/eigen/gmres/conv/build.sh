clear
g++ fft_conv.cpp -I/usr/local/include -L/usr/local/lib -lfftw3 -lm -o conv_test
chmod a+x conv_test
./conv_test
