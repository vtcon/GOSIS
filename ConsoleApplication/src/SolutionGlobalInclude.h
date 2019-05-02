#pragma once

#define _SINGLE_PRECISION 0
#define _DOUBLE_PRECISION 1

//define the precision mode here! After that please recompile the whole solution
#define _PRECISION_MODE _DOUBLE_PRECISION

//define the required compute capabilities here, then change in the Properties/CUDA C++/Device, and recompile whole solution
#define CUDA_CC_MAJOR 7
#define CUDA_CC_MINOR 5




#if _PRECISION_MODE == _SINGLE_PRECISION
#define MYFLOATTYPE float
#else
#define MYFLOATTYPE double
#endif 


