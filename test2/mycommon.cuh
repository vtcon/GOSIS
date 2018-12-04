#pragma once

#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <iostream>

#include <stdio.h>

#define something

#define MYZERO (double)0.000000001
#define MYINFINITY (double)999999999
#define MYFLOATTYPE double

#define NULLVECTOR vec3<MYFLOATTYPE>(0,0,0)

static const int bundlesize = 32;


//#define _DEBUGMODE1
#define _DEBUGMODE2

//this macro prints out cuda API call errors
#define CUDARUN(cudacall) {cudacall;\
cudaError_t cudaStatus = cudaGetLastError();\
if (cudaStatus != cudaSuccess) {\
	fprintf(stderr, "Error at file %s line %d, ",__FILE__,__LINE__);\
	fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));\
}}

//this macro prints out miscellaneous
#ifdef _DEBUGMODE1
#define LOG1(x) printf("%s\n",x);
#else
#define LOG1(x)
#endif

#ifdef _DEBUGMODE2
#define LOG2(x) std::cout << x << "\n";
#else
#define LOG2(x)
#endif