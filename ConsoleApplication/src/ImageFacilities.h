#pragma once

#include "SolutionGlobalInclude.h"
#include <string>

#define IF_PROJECTION_NONE 0
#define IF_PROJECTION_ALONGZ 1
#define IF_PROJECTION_MECATOR 2

void testopencv();

template<typename T>
void quickDisplay(T* rawData, int rows, int columns, int longEdge = 500);

template<typename T>
void quickSave(T* rawData, int rows, int columns, std::string filename, std::string path = "" );

void initiateCV();

template<typename T>
void quickDisplayv2(T* rawData, int rows, int columns, void* map_x, void* map_y, int longEdge);

template<typename T>
void generateProjectionMap(void*& mapX, void*& mapY, int rows, int columns, unsigned int projection = IF_PROJECTION_NONE, int argc = 0, T* argv = nullptr);