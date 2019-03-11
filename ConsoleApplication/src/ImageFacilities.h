#pragma once

#include "SolutionGlobalInclude.h"
#include <string>

#include "ProgramInterface.h"

#define IF_PROJECTION_NONE 0
#define IF_PROJECTION_ALONGZ 1
#define IF_PROJECTION_PLATE_CARREE 2

#define IF_SRGB 0
#define IF_ADOBERGB 1


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

void clearProjectionMap(void *& mapX, void *& mapY);

bool importImageCV(std::vector<tracer::PI_LuminousPoint>& outputvec, std::string path, float posX, float posY, float posZ, float sizeHorz, float sizeVert, float rotX, float rotY, float rotZ, float brightness);

bool importCustomApo(double*& p_customApoData, int& customApoDataSize, std::string path);