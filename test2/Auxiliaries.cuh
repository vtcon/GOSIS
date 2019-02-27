#pragma once

#include "vec3.cuh"
#include "class_hierarchy.cuh"



OpticalConfig::EntrancePupilLocation locateSimpleEntrancePupil(OpticalConfig* currentConfig, int sweepDepth = 6, int scanDepth = 5, MYFLOATTYPE scanPitch = 5.0);
//WARNING: only call this function after the surfaces of the optical config HAVE BEEN COPIED to GPU side

template<typename T>
void init_1D_fan(raybundle<T>* bundle, T z_position, T startTheta, T endTheta, T phi = 0.0, int insize = 32);
//please only send initialized bundles and optical configs to this function

template<typename T>
void init_2D_dualpolar(raybundle<T>* bundle, vec3<T> originpos, T min_horz, T max_horz, T min_vert, T max_vert, T step);
//please only send initialized bundles and optical configs to this function

template<typename T>
void init_2D_dualpolar_v2(raybundle<T>* bundle, OpticalConfig* thisOpticalConfig, vec3<T> origin, T step);
//please only send initialized bundles and optical configs to this function

template<typename T>
void init_2D_dualpolar_v3(raybundle<T>* bundle, OpticalConfig* thisOpticalConfig, LuminousPoint point);
//please only send initialized bundles and optical configs to this function