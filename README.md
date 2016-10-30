CUDA implementation of RGB to grayscale.
Roughly 5x to 30x faster than OpenCV's implementation,
depending on your card.

You can use equal weighting by calling the templated
function with weight set to 'false', or you
can specify custom weights in CUDARGB2Y.h.

The default weights match OpenCV's default.

All functionality is contained in CUDARGB2Y.h and CUDARGB2Y.cu.
'main.cpp' is a demo and test harness.