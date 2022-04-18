#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>

/* input initial width and height.*/
__global__ void ScaleAndGray(unsigned char* orig, unsigned* gray, unsigned width, unsigned height, int scaleFactor);

__global__ void Zncc(unsigned* leftPixels, unsigned* rightPixels, unsigned* disparityMap,
	unsigned width, unsigned height, int minDisp, int maxDisp, int windowWidth, int windowHeight);

/* use integral map and shared memory.*/
__global__ void Zncc_int(unsigned* leftPixels, unsigned* rightPixels, unsigned* disparityMap,
	unsigned width, unsigned height, int minDisp, int maxDisp, int windowWidth, int windowHeight);

__global__ void CrossCheck(unsigned* leftDisp, unsigned* rightDisp, unsigned* result, unsigned imSize, int crossCheckingThreshold);

__global__ void OcclusionFill(unsigned* map, unsigned* result, unsigned width, unsigned height, int occlusionNeighbours);
