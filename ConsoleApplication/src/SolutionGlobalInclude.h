#pragma once

#define _SINGLE_PRECISION 0
#define _DOUBLE_PRECISION 1

//define the precision mode here! After that please recompile the whole solution
#define _PRECISION_MODE _DOUBLE_PRECISION

#if _PRECISION_MODE == _SINGLE_PRECISION
#define MYFLOATTYPE float
#else
#define MYFLOATTYPE double
#endif 


