#pragma once

#include "SolutionGlobalInclude.h"
#include <string>

void testopencv();

template<typename T>
void quickDisplay(T* rawData, int rows, int columns, int longEdge = 500);

template<typename T>
void quickSave(T* rawData, int rows, int columns, std::string filename, std::string path = "" );

void initiateCV();