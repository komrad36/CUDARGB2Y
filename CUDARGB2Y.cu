/*******************************************************************
*   CUDARGB2Y.cu
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

#include "CUDARGB2Y.h"

// Set your weights here.
constexpr double B_WEIGHT = 0.114;
constexpr double G_WEIGHT = 0.587;
constexpr double R_WEIGHT = 0.299;

// Internal; do NOT modify
constexpr int B_WT = static_cast<int>(64.0 * B_WEIGHT + 0.5);
constexpr int G_WT = static_cast<int>(64.0 * G_WEIGHT + 0.5);
constexpr int R_WT = static_cast<int>(64.0 * R_WEIGHT + 0.5);

template<bool weight>
__global__ void CUDARGB2Y_kernel(const cudaTextureObject_t tex_img, const int pixels, uint8_t* const __restrict d_newimg) {
	const unsigned int x = (blockIdx.x << 8) + threadIdx.x;
	const uint8_t res = weight ? min(255, (B_WT*tex1Dfetch<int>(tex_img, 3 * x) + G_WT*tex1Dfetch<int>(tex_img, 3 * x + 1) + R_WT*tex1Dfetch<int>(tex_img, 3 * x + 2)) >> 6)
		: (tex1Dfetch<int>(tex_img, 3 * x) + tex1Dfetch<int>(tex_img, 3 * x + 1) + tex1Dfetch<int>(tex_img, 3 * x + 2)) / 3;
	if (x < pixels) d_newimg[x] = res;
}

void CUDARGB2Y(bool weight, const cudaTextureObject_t tex_img, const int pixels, uint8_t* const __restrict d_newimg) {
	(weight ? CUDARGB2Y_kernel<true> : CUDARGB2Y_kernel<false>)<<<((pixels - 1) >> 8) + 1, 256>>>(tex_img, pixels, d_newimg);
	cudaDeviceSynchronize();
}
