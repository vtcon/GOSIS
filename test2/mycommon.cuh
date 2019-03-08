#pragma once

#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <iostream>

#include <stdio.h>

#include "../ConsoleApplication/src/SolutionGlobalInclude.h"

#define something

//compile time type resolution
//#define MYFLOATTYPE float
#define sqrt(x) sqrt((MYFLOATTYPE)(x))

// compile-time constants
#define MYZERO (double)0.000000001
#define MYINFINITY (double)999999999
#define MYPI (double)3.14159265358979323846264338327950288419716939937510582097494 
#define MYEPSILONBIG (MYFLOATTYPE)0.0001
#define MYEPSILONMEDIUM (MYFLOATTYPE)0.0000001
#define MYEPSILONSMALL (MYFLOATTYPE)0.0000000001

#define NULLVECTOR vec3<MYFLOATTYPE>(0,0,0)

//static const int bundlesize = 32;

//#define _MYDEBUGMODE

#ifdef _MYDEBUGMODE
//#define _DEBUGMODE1
#define _DEBUGMODE2
#endif

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

//pointer swap by reference
template<typename pT>
__host__ __device__ inline void swap(pT& a, pT& b)
{
	pT temp = a;
	a = b;
	b = temp;
}

/*
template<typename T = float>
__device__ T operator/(T lhs, T rhs)
{
	return (rhs == 0) ? (T)MYINFINITY : lhs / rhs;
}
*/ //IEEE 754 to the rescue!

#define ASSERT(x) if (!(x)) __debugbreak();

#ifdef __CUDACC__
#define forhost __host__
#define fordevice __device
#else
#define forhost
#define fordevice
#endif