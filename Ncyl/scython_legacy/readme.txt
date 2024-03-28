TO BUILD: python3 setup.py build_ext --inplace

It will probably throw some type error warnings, these are ok.

There are a variety of options we can set for the compiler inside of setup.py. Right now I have -O3 as the optimization flag. There are ways to compile with icc, i++ ect. but I have not played around with that.
