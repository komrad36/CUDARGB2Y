/*******************************************************************
*   CUDARGB2Y.h
*   CUDARGB2Y
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Oct 29, 2016
*******************************************************************/
//
// CUDA implementation of RGB to grayscale.
// Roughly 5x to 30x faster than OpenCV's implementation,
// depending on your card.
//
// Converts an RGB color image to grayscale.
//
// You can use equal weighting by calling the templated
// function with weight set to 'false', or you
// can specify custom weights in CUDARGB2Y.h.
//
// The default weights match OpenCV's default.
// 
// All functionality is contained in CUDARGB2Y.h and CUDARGB2Y.cu.
// 'main.cpp' is a demo and test harness.
//

#pragma once

#include <cstdint>

#include "cuda_runtime.h"

#ifdef __INTELLISENSE__
#define asm(x)
#define min(x) 0
#define fmaf(x) 0
#include "device_launch_parameters.h"
#define __CUDACC__
#include "device_functions.h"
#undef __CUDACC__
#endif

void CUDARGB2Y(bool weight, const cudaTextureObject_t tex_img, const int pixels, uint8_t* const __restrict d_newimg);
