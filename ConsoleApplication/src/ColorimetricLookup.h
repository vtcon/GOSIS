#pragma once

#include "SolutionGlobalInclude.h"
#include "pch.h"
#include "CppCommon.h"

//enum is too cumbersome, #define makes a cleaner code
#define CLU_LMS_S 1
#define CLU_LMS_M 2
#define CLU_LMS_L 3
#define CLU_XYZ_X 4
#define CLU_XYZ_Y 5
#define CLU_XYZ_Z 6
/* // not now
#define CLU_V_photopic 7
#define CLU_V_scotopic 8
*/

namespace ColorimetricLookup
{
	MYFLOATTYPE lookup(MYFLOATTYPE wavelength, unsigned short int colorMatchingFunction);
}


