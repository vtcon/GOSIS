#pragma once
#include "vec3.cuh"
#include "class_hierarchy.cuh"

class KernelLaunchParams
{
public:
private:
};

class QuadricTracerKernelLaunchParams :public KernelLaunchParams
{
public:
	raybundle<MYFLOATTYPE>** d_inbundles = nullptr;
	raybundle<MYFLOATTYPE>** d_outbundles = nullptr;
	quadricsurface<MYFLOATTYPE>* pquad = nullptr;
	int otherparams[7];
};

struct PerKernelRenderingInput
{
	vec3<MYFLOATTYPE> vtx1;
	vec3<MYFLOATTYPE> vtx2;
	vec3<MYFLOATTYPE> vtx3;
	vec3<MYFLOATTYPE> dir1;
	vec3<MYFLOATTYPE> dir2;
	vec3<MYFLOATTYPE> dir3;
	float intensity1;
	float intensity2;
	float intensity3;
	//more could be intensity, OPL etc.
};

class RendererKernelLaunchParams:public KernelLaunchParams
{
public:
	SimpleRetinaDescriptor retinaDescriptorIn;
	RetinaImageChannel* dp_rawChannel;
	PerKernelRenderingInput* dp_triangles;
	int otherparams[7]; //[0] is the triangle count
};

int OpticalConfigManager(int argc = 0, char** argv = nullptr);

int KernelLauncher(int argc = 0, char** argv = nullptr);

int KernelLauncher2(int argc = 0, char** argv = nullptr);

int ColumnCreator(int argc = 0, char** argv = nullptr);

void testbenchGPU();

#define MF_CONVEX 0
#define MF_CONCAVE 1
#define MF_FLAT 2
#define MF_REFRACTIVE 5
#define MF_IMAGE 6
#define MF_STOP 7

bool constructSurface(mysurface<MYFLOATTYPE>*& p_surface,
	unsigned short int surfaceType,
	vec3<MYFLOATTYPE> vertexPositionInMM,
	MYFLOATTYPE curvatureR,
	MYFLOATTYPE diameterInMM,
	unsigned short int side,
	MYFLOATTYPE n1 = 1,
	MYFLOATTYPE n2 = 1,
	MYFLOATTYPE conicConstantK = 0,
	unsigned short int apodization = APD_UNIFORM,
	std::string customApoPath = "",
	point2D<MYFLOATTYPE> primaryAxisTiltThetaAndPhiInDegree = { 0,0 });

int ColumnCreator3();
int ColumnCreator4();
int KernelLauncher(int argc, char** argv);
int KernelLauncher2(int argc, char** argv);