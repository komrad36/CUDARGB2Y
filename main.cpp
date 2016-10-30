/*******************************************************************
*   main.cpp
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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

void printReport(const std::string& name, const nanoseconds& dur, const nanoseconds& comp = nanoseconds(0)) {
	std::cout << std::left << std::setprecision(6) << std::setw(10) << name << " took " << std::setw(7) << static_cast<double>(dur.count()) * 1e-3 << " us";
	if (comp.count() && comp < dur) {
		std::cout << " (" << std::setprecision(4) << std::setw(4) << static_cast<double>(dur.count()) / comp.count() << "x slower than RGB2Y)." << std::endl;
	}
	else {
		std::cout << '.' << std::endl;
	}
}

void RGB2Y_ref(const uint8_t* __restrict const data, const int32_t cols, const int32_t rows, const int32_t stride, uint8_t* const __restrict out) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			const auto idx = 3 * (i*stride + j);
			out[i*stride + j] = (static_cast<uint32_t>(data[idx]) + static_cast<uint32_t>(data[idx + 1]) + static_cast<uint32_t>(data[idx + 2])) / 3;
		}
	}
}

int main() {
	// ------------- Configuration ------------
	constexpr bool display_image = true;
	constexpr auto warmups = 100;
	constexpr auto runs = 2000;
	constexpr bool weighted_averaging = true;
	constexpr char name[] = "test.jpg";
	// --------------------------------


	// ------------- Image Read ------------
	cv::Mat image = cv::imread(name);
	if (!image.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	// --------------------------------


	// ------------- Ref ------------
	uint8_t* refresult = new uint8_t[image.cols*image.rows];
	RGB2Y_ref(image.data, image.cols, image.rows, image.cols, refresult);
	// --------------------------------


	// setting cache and shared modes
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	// allocating and transferring image as texture object
	uint8_t* d_img;
	cudaMalloc(&d_img, 3*image.rows*image.cols);
	cudaMemcpy(d_img, image.data, 3*image.rows*image.cols, cudaMemcpyHostToDevice);
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = d_img;
	resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
	resDesc.res.linear.desc.x = 8;
	resDesc.res.linear.sizeInBytes = 3 * image.rows * image.cols;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] = texDesc.addressMode[3] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t tex_img = 0;
	cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, nullptr);

	// allocating space for new image
	uint8_t* d_newimg;
	cudaMalloc(&d_newimg, image.rows*image.cols);

	std::cout << (weighted_averaging ? "Weighted" : "Equal") << " averaging mode." << std::endl;

	std::cout << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) CUDARGB2Y(weighted_averaging, tex_img, image.cols*image.rows, d_newimg);
	std::cout << "Testing..." << std::endl;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) CUDARGB2Y(weighted_averaging, tex_img, image.cols*image.rows, d_newimg);
	high_resolution_clock::time_point end = high_resolution_clock::now();
	nanoseconds RGB2Y_ns = (end - start) / runs;
	// --------------------------------

	// transferring new image back to host
	uint8_t* h_newimg = reinterpret_cast<uint8_t*>(malloc(image.rows*image.cols));
	cudaMemcpy(h_newimg, d_newimg, image.rows*image.cols, cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	std::cout << "CUDA reports " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	// ------------- OpenCV ------------
	nanoseconds CV_ns;
	cv::Mat newimage_cv;
	std::cout << "------------ OpenCV ------------" << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) cv::cvtColor(image, newimage_cv, CV_BGR2GRAY);
	std::cout << "Testing..." << std::endl;
	{
		start = high_resolution_clock::now();
		for (int32_t i = 0; i < runs; ++i) {
			cv::cvtColor(image, newimage_cv, CV_BGR2GRAY);
		}
		end = high_resolution_clock::now();
		CV_ns = (end - start) / runs;
	}
	// --------------------------------


	// ------------- Verification ------------
	int i = 0;
	for (; i < image.cols*image.rows; ++i) {
		if (abs(h_newimg[i] - (weighted_averaging ? newimage_cv.data[i] : refresult[i])) > 1) {
			std::cerr << "ERROR! One or more pixels disagree!" << std::endl;
			std::cerr << i << ": got " << +h_newimg[i] << ", should be " << (weighted_averaging ? +newimage_cv.data[i] : +refresult[i]) << std::endl;
			break;
		}
	}
	if (i == image.cols*image.rows) std::cout << std::endl << "All pixels agree! Test valid." << std::endl << std::endl;
	// --------------------------------


	// ------------- Output ------------
	printReport("CUDARGB2Y", RGB2Y_ns);
	printReport("OpenCV", CV_ns, RGB2Y_ns);
	std::cout << std::endl;
	if (display_image) {
		cv::namedWindow("CUDARGB2Y", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::imshow("CUDARGB2Y", cv::Mat(image.rows, image.cols, CV_8U, h_newimg));
		cv::namedWindow("OpenCV", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::imshow("OpenCV", newimage_cv);
		cv::waitKey(0);
	}
	// --------------------------------
}
