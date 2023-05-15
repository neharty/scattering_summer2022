This folder contains output data related to ringing tests

largetime - varied the computation time multiplier (see included png)

k_cutoff - varied the cutoff frequency error, which will increase the maximum frequency computed
    1e-10 -> f_c ~ 3.9
    1e-11 -> f_c ~ 4.0
    1e-12 -> f_c ~ 4.2
    1e-13 -> f_c ~ 50.7

sampnum - varied the sampling number on the interval [0, T], call it N. The total number of samples for the full time interval should be 2MN (see png) where M is some multiplier. In the examples we talked about M=4.

dist - kept computation parameters constant while varying the center-to-center spacing of the circles.
