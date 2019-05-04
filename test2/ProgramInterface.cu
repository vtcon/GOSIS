#include "ProgramInterface.h"
#include "StorageManager.cuh"
#include "ManagerFunctions.cuh"
#include "Auxiliaries.cuh"
#include "../ConsoleApplication/src/ImageFacilities.h"
#include "../ConsoleApplication/src/OutputImage.h"
#include "../ConsoleApplication/src/GLDrawFacilities.h"
#include "PerformanceTester.cuh"

#include <list>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cstdlib> //for the "rand" function
#include <chrono>
#include <thread>

//for the console:
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <windows.h>

//for the hardware info
#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "user32.lib")

using namespace tracer;

//internal global variables, please think VERY carefully before modifying these lines
std::list<StorageManager> repoList;
StorageManager mainStorageManager;
float activeWavelength = 555.0; //pls don't break this line
std::unordered_map<int, OutputImage> outputImages;
bool maximizeContrast = true; //should be set to true if inputs are points, and false if input is an image

//external global variables
int PI_ThreadsPerKernelLaunch = 32;
int PI_linearRayDensity = 25;//20 is ok
unsigned int PI_rgbStandard = IF_ADOBERGB;
int PI_traceJobSize = 10;
int PI_renderJobSize = 10;
unsigned short int PI_rawFormat = OIC_XYZ; //OIC_LMS
unsigned int PI_projectionMethod = IF_PROJECTION_PLATE_CARREE;
int PI_displayWindowSize = 800;
float PI_primaryWavelengthR = 620;
float PI_primaryWavelengthG = 530;
float PI_primaryWavelengthB = 465;
int PI_maxTextureDimension = 2048;
int PI_maxParallelThread = 10;
int PI_performanceTestRepetition = 100000;

//internally used global variables, but still should be put on preferences
int PI_refractiveSurfaceArms = 20;
int PI_refractiveSurfaceRings = 20;
int PI_imageSurfaceArms = 60;
int PI_imageSurfaceRings = 60;

//these variables serves the progress count
float PI_traceProgress; //from 0.0 to 1.0
float PI_renderProgress; //from 0.0 to 1.0
long toTraceCount = 0;
long tracedCount = 0;
long toRenderCount = 0;
long renderedCount = 0;
void updateProgressTrace();
void updateProgressRender();

//these variables controls the execution
bool PI_cancelTraceRender = false;
bool PI_running = false;

//function definitions
bool PI_LuminousPoint::operator==(const PI_LuminousPoint & rhs) const
{
	if (x == rhs.x &&y == rhs.y &&z == rhs.z &&wavelength == rhs.wavelength &&intensity == rhs.intensity)
		return true;
	else
		return false;
}

extern bool runTestOpenCV;
extern bool runTestOpenGL;
extern bool runTestGLDrawFacilities;
extern void GLtest();
extern int GLInfoPrint();
bool runPerformanceTest = true;
bool normalTraceRenderTest = false;

PI_Message tracer::initialization()
{
	//do necessary program initialization here
	PI_maxParallelThread = std::thread::hardware_concurrency();

	//checking system info, suppressed in debug mode
#ifndef _DEBUG
	SYSTEM_INFO siSysInfo;
	GetSystemInfo(&siSysInfo);

	printf("--- General hardware information: ---\n");
	printf(" OEM ID: %u\n", siSysInfo.dwOemId);
	printf(" Number of processors: %u\n",
		siSysInfo.dwNumberOfProcessors);
	printf(" Page size: %u\n", siSysInfo.dwPageSize);
	printf(" Processor type: %u\n", siSysInfo.dwProcessorType);
	printf(" Minimum application address: %lu\n",
		siSysInfo.lpMinimumApplicationAddress);
	printf(" Maximum application address: %lu\n",
		siSysInfo.lpMaximumApplicationAddress);
	printf(" Active processor mask: %u\n",
		siSysInfo.dwActiveProcessorMask);

	//checking ram info
	MEMORYSTATUSEX memInfo;
	memInfo.dwLength = sizeof(MEMORYSTATUSEX);
	GlobalMemoryStatusEx(&memInfo);
	DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;
	DWORDLONG totalPhysMem = memInfo.ullTotalPhys;
	printf(" Total physical memory: %lu\n", totalPhysMem);
	printf(" Total virtual memory: %lu\n\n", totalVirtualMem);
	
	//checking CUDA devices
	printf("--- CUDA Device(s) information: ---\n");
	int count = 0;
	CUDARUN(cudaGetDeviceCount(&count));

	if (count == 0)
	{
		printf("--- No CUDA device detected! ---\n");
		return { PI_SYSTEM_REQUIREMENT_ERROR , "No CUDA device detected!\n" };
	}

	cudaDeviceProp* prop = new cudaDeviceProp[count];

	for (int i = 0; i < count; i++) {
		CUDARUN(cudaGetDeviceProperties(&prop[i], i));
		printf("General Information for device %d\n", i);
		printf("  Name: %s\n", prop[i].name);
		printf("  Compute capability: %d.%d\n", prop[i].major, prop[i].minor);
		printf("  Clock rate: %d\n", prop[i].clockRate);
		printf("  Device copy overlap: ");
		if (prop[i].deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("  Kernel execution timeout : ");
		if (prop[i].kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Memory Information for device %d\n", i);
		printf("  Total global mem: %lu\n", prop[i].totalGlobalMem);
		printf("  Total constant Mem: %ld\n", prop[i].totalConstMem);
		printf("  Max mem pitch: %ld\n", prop[i].memPitch);
		printf("  Texture Alignment: %ld\n", prop[i].textureAlignment);
		printf("MP Information for device %d\n", i);
		printf("  Multiprocessor count: %d\n",
			prop[i].multiProcessorCount);
		printf("  Shared mem per mp: %ld\n", prop[i].sharedMemPerBlock);
		printf("  Registers per mp: %d\n", prop[i].regsPerBlock);
		printf("  Threads in warp: %d\n", prop[i].warpSize);
		printf("  Max threads per block: %d\n",
			prop[i].maxThreadsPerBlock);
		printf("  Max thread dimensions: (%d, %d, %d)\n",
			prop[i].maxThreadsDim[0], prop[i].maxThreadsDim[1],
			prop[i].maxThreadsDim[2]);
		printf("  Max grid dimensions: (%d, %d, %d)\n",
			prop[i].maxGridSize[0], prop[i].maxGridSize[1],
			prop[i].maxGridSize[2]);
		printf("\n");
	}

	//choose cuda device with highest SM and CC
	int min_major = CUDA_CC_MAJOR;
	int min_minor = CUDA_CC_MINOR;
	bool exist = false;
	for (int i = 0; i < count; i++)
	{
		if (prop[i].major >= min_major && prop[i].minor >= min_minor)
		{
			min_major = prop[i].major;
			min_minor = prop[i].minor;
			exist = true;
		}
	}
	if (!exist)
	{
		printf("Error: CUDA device is required to have compute capabilities %d.%d or higher! \n", min_major, min_minor);
		return { PI_SYSTEM_REQUIREMENT_ERROR , "No CUDA device matches the compute capabilities requirement!\n" };
	}

	cudaDeviceProp selectProp;
	int selectDev;
	memset(&selectProp, 0, sizeof(cudaDeviceProp));
	selectProp.major = min_major;
	selectProp.minor = min_minor;
	CUDARUN(cudaChooseDevice(&selectDev, &selectProp));
	CUDARUN(cudaSetDevice(selectDev));
	printf("CUDA device with ID = %d and compute capabilities %d.%d was selected.\n", selectDev, selectProp.major, selectProp.minor);
	printf("Minimum required compute capabilities: %d.%d was selected.\n", CUDA_CC_MAJOR, CUDA_CC_MINOR);

	delete[] prop;


	printf("\n--- OpenGL information ---\n");
	
	int ret = GLInfoPrint();
	if (ret != 0)
	{
		std::cout << "Cannot initialize OpenGL!\n";
		return { PI_SYSTEM_REQUIREMENT_ERROR , "Cannot initialize OpenGL!\n" };
	}
#endif 

	return { PI_OK, "Successful!" };
}

PI_Message tracer::test()
{
	if (runPerformanceTest)
	{
		std::cout << "Adjust the Preference and Enter the number of repetitions:\n";
		int repetition = PI_performanceTestRepetition;
		{
			std::cout << "Running GPU tracing test with repetition = " << repetition << " times!\n";

			//start clock
			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

			testTracingGPU(repetition);

			//stop clock
			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
			std::cout << "GPU tracing completed after " << duration << " ms\n";
		}
		{
			std::cout << "Running CPU tracing test with repetition = " << repetition << " times!\n";

			//start clock
			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

			testTracingCPU(repetition);

			//stop clock
			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
			std::cout << "CPU tracing completed after " << duration << " ms\n";
		}
	}
	if (runTestOpenCV)
	{
		maximizeContrast = false;
		testopencv();
		maximizeContrast = true;
	}
	if (runTestOpenGL)
	{
		GLtest();
	}
	if (runTestGLDrawFacilities)
	{
		testGLDrawFacilities();
	}
	if (normalTraceRenderTest)
	{
		//tracer::importImage("C:/testinput.bmp", 20, 20, 30, 25, 25, 45, 90, 45, 700, 550, 400, 1.0);
	/*
	int count = 0;
	std::cout << "Please enter the number of surfaces\n";
	std::cin >> count;

	PI_Surface* surfaces = new PI_Surface[count];
	float angularResol;
	//float angularExtend;

	for (int i = 0; i < count - 1; i++)
	{
		std::cout << "For surface " << i + 1 << ":\n";
		std::cout << "Please enter vertex position\n";
		std::cin >> surfaces[i].z;
		std::cout << "Please enter diameter\n";
		std::cin >> surfaces[i].diameter;
		std::cout << "Please enter curvature radius\n";
		std::cin >> surfaces[i].radius;
		std::cout << "Please enter refractive index\n";
		std::cin >> surfaces[i].refractiveIndex;
	}
	{
		int i = count - 1;
		std::cout << "For image surface:\n";
		std::cout << "Please enter diameter\n";
		std::cin >> surfaces[i].diameter;
		std::cout << "Please enter curvature radius\n";
		std::cin >> surfaces[i].radius;
	}
	{
		std::cout << "For the retina: \n";
		std::cout << "Please angular resolution\n";
		std::cin >> angularResol;
		//std::cout << "Please angular extend\n";
		//std::cin >> angularExtend;
	}
	*/


		{
			int count = 2;
			PI_Surface* surfaces = new PI_Surface[count];
			surfaces[0].z = 40.0; surfaces[0].diameter = 40.0; surfaces[0].radius = 40.0;
			surfaces[0].refractiveIndex = 1.5168f; surfaces[0].asphericity = 0.95f;
			surfaces[0].apodization = PI_APD_CUSTOM; surfaces[0].customApoPath = "C:/testcustomapo.jpg";

			surfaces[1].diameter = 40.0; surfaces[1].radius = -60.0;
			float angularResol = 0.16f;//0.16 is OK

			float angularExtend = 90.0;

			addOpticalConfigAt(555.0, count, surfaces, angularResol, angularExtend);
			delete[] surfaces;
		}
		/*
		{
			int count = 4;
			PI_Surface* surfaces = new PI_Surface[count];
			surfaces[0].z = 40.0; surfaces[0].diameter = 40.0; surfaces[0].radius = 40.0; surfaces[0].refractiveIndex = 1.5168; surfaces[0].asphericity = -0.5;
			surfaces[1].z = 25.0; surfaces[1].diameter = 40.0; surfaces[1].radius = 40.0; surfaces[1].refractiveIndex = 1.7;
			surfaces[2].z = 15.0; surfaces[2].diameter = 40.0; surfaces[2].radius = 40.0; surfaces[2].refractiveIndex = 2.0;
			surfaces[3].diameter = 40.0; surfaces[3].radius = -60.0;
			float angularResol = 0.16;//0.16 is OK

			float angularExtend = 90.0;


			addOpticalConfigAt(555.0, count, surfaces, angularResol, angularExtend);
			delete[] surfaces;
		}
		{
			int count = 4;
			PI_Surface* surfaces = new PI_Surface[count];
			surfaces[0].z = 40.0; surfaces[0].diameter = 40.0; surfaces[0].radius = 40.0; surfaces[0].refractiveIndex = 1.5168;
			surfaces[1].z = 25.0; surfaces[1].diameter = 40.0; surfaces[1].radius = 40.0; surfaces[1].refractiveIndex = 1.7;
			surfaces[2].z = 15.0; surfaces[2].diameter = 40.0; surfaces[2].radius = 40.0; surfaces[2].refractiveIndex = 2.0;
			surfaces[3].diameter = 40.0; surfaces[3].radius = -60.0;
			float angularResol = 0.16;//0.16 is OK

			float angularExtend = 90.0;


			addOpticalConfigAt(400.0, count, surfaces, angularResol, angularExtend);
			delete[] surfaces;
		}
		{
			int count = 4;
			PI_Surface* surfaces = new PI_Surface[count];
			surfaces[0].z = 40.0; surfaces[0].diameter = 40.0; surfaces[0].radius = 40.0; surfaces[0].refractiveIndex = 1.5168;
			surfaces[1].z = 25.0; surfaces[1].diameter = 40.0; surfaces[1].radius = 40.0; surfaces[1].refractiveIndex = 1.7;
			surfaces[2].z = 15.0; surfaces[2].diameter = 40.0; surfaces[2].radius = 40.0; surfaces[2].refractiveIndex = 2.0;
			surfaces[3].diameter = 40.0; surfaces[3].radius = -60.0;
			float angularResol = 0.16;//0.16 is OK

			float angularExtend = 90.0;


			addOpticalConfigAt(650.0, count, surfaces, angularResol, angularExtend);
			delete[] surfaces;
		}
		*/
		/*
		std::cout << "Please enter the number of points\n";
		int pcount = 0;
		std::cin >> pcount;

		for (int i = 0; i < pcount; i++)
		{
			PI_LuminousPoint point;
			std::cout << "For point " << i + 1 << ":\n";
			std::cout << "Please enter X\n";
			std::cin >> point.x;
			std::cout << "Please enter Y\n";
			std::cin >> point.y;
			std::cout << "Please enter Z\n";
			std::cin >> point.z;
			addPoint(point);
		}
		*/

		{
			PI_LuminousPoint point;
			point.x = 0;	point.y = 0;	point.z = 250;	point.wavelength = 555.0;
			addPoint(point);
			/*
			point.x = -20;	point.y = -30;	point.z = 180;	point.wavelength = 400.0;	point.intensity = 5.0;
			addPoint(point);
			point.x = 30;	point.y = -30;	point.z = 180;	point.wavelength = 650.0;	point.intensity = 5.0;
			addPoint(point);
			point.x = -20;	point.y = -20;	point.z = 160;	point.wavelength = 555.0;
			addPoint(point);
			point.x = 0;	point.y = 0;	point.z = 160;	point.wavelength = 400.0;	point.intensity = 5.0;
			addPoint(point);
			point.x = 20;	point.y = 0;	point.z = 200;	point.wavelength = 400.0;	point.intensity = 5.0;
			addPoint(point);
			point.x = 0;	point.y = -30;	point.z = 180;	point.wavelength = 400.0;	point.intensity = 5.0;
			addPoint(point);
			point.x = -30;	point.y = 0;	point.z = 160;	point.wavelength = 650.0;	point.intensity = 5.0;
			addPoint(point);
			point.x = 40;	point.y = 0;	point.z = 200;	point.wavelength = 650.0;	point.intensity = 5.0;
			addPoint(point);
			point.x = -40;	point.y = -30;	point.z = 180;	point.wavelength = 650.0;	point.intensity = 5.0;
			addPoint(point);
			*/
		}

		/*
		int rayDensity = 20;
		std::cout << "Enter desired linear ray generation density: \n";
		std::cin >> rayDensity;
		*/

		std::cout << "Starting...\n";

		checkData();
		trace();
		render();
		/*
		{
			float wavelengths[3] = { 400.0,555.0, 650.0 };
			int imageID = 0;
			createOutputImage(3, wavelengths, imageID);
		}
		*/
		{
			float wavelengths[1] = { 555.0 };
			int imageID = 0;
			createOutputImage(1, wavelengths, imageID);
		}
		clearStorage();
	}
	
	return {PI_OK, "Test OK"};
}

PI_Message tracer::addPoint(PI_LuminousPoint & toAdd)
{
	LuminousPoint point;
	point.x = toAdd.x;
	point.y = toAdd.y;
	point.z = toAdd.z;
	point.intensity = toAdd.intensity;
	point.wavelength = toAdd.wavelength;
	
	//why this long initialization when we could have done point(toAdd)??
	//because create this constructor LuminousPoint(tracer::PI_LuminousPoint) would leed to cyclical inclusion
	//... of the header files class_hierarchy.h and programinterface.h
	//... and would create hard to track bugs later

	bool result = mainStorageManager.add(point);
	if (result)
	{
		toTraceCount++;
		toRenderCount++;

		return { PI_OK, "Point added successfully!\n" };
	}
	else
	{
		std::stringstream errormsg;
		errormsg << "Could not add point (" << toAdd.x << ", " << toAdd.y << " ," << toAdd.z << "), " << toAdd.intensity << " at " << toAdd.wavelength << " nm!\n";
		return { PI_INPUT_ERROR, errormsg.str().c_str() };
	}
}

PI_Message tracer::addOpticalConfigAt(float wavelength, int count, PI_Surface *& toAdd, float angularResolution, float angularExtend)
{
	int numofsurfaces = count;
	float wavelength1 = wavelength;

	//check out an opticalconfig as output
	std::cout << "Checking out configuration at " << wavelength1 << " nm \n";
	OpticalConfig* newConfig = nullptr;
	mainStorageManager.jobCheckOut(newConfig, numofsurfaces, wavelength1);

	float previousN = 1.0;
	//for all the refractive surfaces
	for (int i = 0; i < count-1; i++)
	{
		//todo: data integrity check here

		auto curveSign = (toAdd[i].radius > 0) ? MF_CONVEX : MF_CONCAVE;
		toAdd[i].diameter = (abs(toAdd[i].diameter) < 2.0f*abs(toAdd[i].radius)) ? abs(toAdd[i].diameter) : 2.0f*abs(toAdd[i].radius);
		bool output = constructSurface((newConfig->surfaces)[i], MF_REFRACTIVE, vec3<MYFLOATTYPE>(toAdd[i].x, toAdd[i].y, toAdd[i].z), abs(toAdd[i].radius), abs(toAdd[i].diameter), curveSign, previousN, toAdd[i].refractiveIndex, toAdd[i].asphericity, toAdd[i].apodization, toAdd[i].customApoPath);
		if (!output) return { PI_UNKNOWN_ERROR,"Error adding surface" };
		previousN = toAdd[i].refractiveIndex;
	}

	//test apo function
	//(newConfig->surfaces)[0]->apodizationType = APD_BARTLETT;

	//for the final image surface
	//todo: data integrity check here
	toAdd[numofsurfaces - 1].diameter = (abs(toAdd[numofsurfaces - 1].diameter) < 2.0f*abs(toAdd[numofsurfaces - 1].radius)) ? abs(toAdd[numofsurfaces - 1].diameter) : 2.0f*abs(toAdd[numofsurfaces - 1].radius);
	bool output = constructSurface((newConfig->surfaces)[numofsurfaces-1], MF_IMAGE, vec3<MYFLOATTYPE>(toAdd[count-1].x, toAdd[count - 1].y, toAdd[count - 1].z), -abs(toAdd[count - 1].radius), toAdd[count - 1].diameter, MF_CONCAVE, 0.0, FP_INFINITE, toAdd[count-1].asphericity);

	//copy to sibling surfaces on GPU side
	LOG1("[main]create sibling surfaces\n");
	newConfig->copytosiblings();

	//find entrance pupil
	locateSimpleEntrancePupil(newConfig);

	//add a retina
	PixelArrayDescriptor* p_retinaDescriptor = nullptr;
	MYFLOATTYPE R = 10; //random initial value
	if (dynamic_cast<quadricsurface<MYFLOATTYPE>*>(newConfig->surfaces[numofsurfaces - 1])->isFlat())
	{
		R = newConfig->entrance.z; //take the distance from entrance pupil to retina vertex as R
	}
	else //if it is not flat
	{
		//obsolete
		//MYFLOATTYPE Rsqr = abs(dynamic_cast<quadricsurface<MYFLOATTYPE>*>(newConfig->surfaces[numofsurfaces - 1])->param.J);
		R = abs(dynamic_cast<quadricsurface<MYFLOATTYPE>*>(newConfig->surfaces[numofsurfaces - 1])->param.I)/(MYFLOATTYPE)2.0;
		//R = sqrt(Rsqr);
	}

	//angular extend actually depends on diameter and curvature radius!
	//my choice: if angular extend is smaller than the diameter allows, take the angular
	//if not, take the from-the-diameter-resulting extend
	
	MYFLOATTYPE maxAngularExtend = asin(toAdd[numofsurfaces - 1].diameter / (2.0*R)) / MYPI * 180.0;
	angularExtend = (angularExtend < (float)maxAngularExtend) ? angularExtend : (float)maxAngularExtend;
	
	p_retinaDescriptor = new SimpleRetinaDescriptor(angularResolution, R, angularExtend, angularExtend); //doesn't need to explicitly delete this
	newConfig->createImageChannel(p_retinaDescriptor);

	return { PI_OK, "Optical Config added" };
}

PI_Message tracer::checkData()
{
#ifdef nothing
	float* currentwavelength = nullptr;
	int runCounts = 0;
	for (; bool output = mainStorageManager.takeOne(currentwavelength, StorageHolder<float>::Status::uninitialized); runCounts++)
	{
		//set active wavelength
		activeWavelength = *currentwavelength;
		std::cout << "Check data at wavelength = " << activeWavelength << "\n";

		//check if optical config even exists
		OpticalConfig* thisOpticalConfig = nullptr;
		bool b_getConfig = mainStorageManager.infoCheckOut(thisOpticalConfig, activeWavelength);
		if (!b_getConfig)
		{
			std::cout << "Cannot get optical config for this wavelength\n";
			return { PI_UNKNOWN_ERROR, "Cannot get optical config for this wavelength\n" };
		}

		//initialize all points
		while (ColumnCreator4() != -2);
		mainStorageManager.jobCheckIn(currentwavelength, StorageHolder<float>::Status::initialized);
	}

	if (runCounts == 0)
	{
		std::cout << "No wavelength is available for checking\n";
		return { PI_UNKNOWN_ERROR, "No wavelength is available for checking\n" };
	}
#endif // nothing
	std::cout << "Data Check OK!\n";

	return { PI_OK, "Check successful!\n" };
}

PI_Message tracer::trace()
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	float* currentwavelength = nullptr;
	int runCounts = 0;
	for (; bool output = mainStorageManager.takeOne(currentwavelength, StorageHolder<float>::Status::uninitialized); runCounts++)
	{
		//trace all points
		activeWavelength = *currentwavelength;
		std::cout << "Tracing at wavelength = " << activeWavelength << "\n";

		PI_running = true;
		while (ColumnCreator4() != -2)
		{
			while (KernelLauncher(0, nullptr) != -2)
			{
				tracedCount = tracedCount + PI_traceJobSize;
				updateProgressTrace();
				if (PI_cancelTraceRender)
				{
					PI_running = false;
					return { PI_OK, "Trace cancelled!\n" };
				}
			}
		}
		PI_running = false;
		mainStorageManager.jobCheckIn(currentwavelength, StorageHolder<float>::Status::completed1);
	}
	
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	
	if (runCounts == 0)
	{
		std::cout << "No wavelength is available for tracing\n";
		return { PI_UNKNOWN_ERROR, "No wavelength is available for tracing\n" };
	}

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "Tracing completed after " << duration << " ms\n";

	return { PI_OK, "Trace successful!\n" };
}

PI_Message tracer::render()
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	float* currentwavelength = nullptr;
	int runCount = 0;
	for (; mainStorageManager.takeOne(currentwavelength, StorageHolder<float>::Status::completed1); runCount++)
	{
		//set the current wavelength
		activeWavelength = *currentwavelength;

		//get the corresponding optical config
		OpticalConfig* thisOpticalConfig = nullptr;
		bool b_getConfig = mainStorageManager.infoCheckOut(thisOpticalConfig, activeWavelength);
		if (!b_getConfig)
		{
			std::cout << "Cannot get optical config for this wavelength\n";
			return { PI_UNKNOWN_ERROR, "Cannot get optical config for this wavelength\n" };
		}

		//create a (hopefully clear) image on device
		thisOpticalConfig->p_rawChannel->createSibling();

		//run the renderer
		PI_running = true;
		while (KernelLauncher2(0, nullptr) != -2)
		{
			renderedCount = renderedCount + PI_renderJobSize;
			updateProgressRender();
			if (PI_cancelTraceRender)
			{
				thisOpticalConfig->p_rawChannel->copyFromSibling();
				thisOpticalConfig->p_rawChannel->deleteSibling();
				PI_running = false;
				return { PI_OK, "Render cancelled!\n" };
			}
		}
		PI_running = false;
		//deal with the output
		//data copy out
		thisOpticalConfig->p_rawChannel->copyFromSibling();

		//testing: draw all retina
		//thisOpticalConfig->p_rawChannel->setToValue(1.0, *(thisOpticalConfig->p_retinaDescriptor));
		/*
		//display data with scaling
		void* mapX = nullptr;
		void* mapY = nullptr;
		SimpleRetinaDescriptor* tempDescriptor = dynamic_cast<SimpleRetinaDescriptor*>(thisOpticalConfig->p_retinaDescriptor);

		MYFLOATTYPE scalingargs[4] = { tempDescriptor->m_thetaR, tempDescriptor->m_R0, tempDescriptor->m_maxTheta, tempDescriptor->m_maxPhi };
		generateProjectionMap(mapX, mapY, thisOpticalConfig->p_rawChannel->m_dimension.y,
			thisOpticalConfig->p_rawChannel->m_dimension.x, IF_PROJECTION_ALONGZ, 4, scalingargs);

		quickDisplayv2(thisOpticalConfig->p_rawChannel->hp_raw, thisOpticalConfig->p_rawChannel->m_dimension.y,
			thisOpticalConfig->p_rawChannel->m_dimension.x, mapX, mapY, 700);
		*/

		//performing clean up
		thisOpticalConfig->p_rawChannel->deleteSibling();

		//TODO: move this line elsewhere
		//thisOpticalConfig->p_rawChannel->deleteHostImage();

		//tell the storage manager that the job is done
		mainStorageManager.jobCheckIn(currentwavelength, StorageHolder<float>::Status::completed2);
	}

	//if output == false, what to do???
	if (runCount == 0)
	{
		std::cout << "No wavelength is available for rendering\n";
		return { PI_UNKNOWN_ERROR, "No wavelength is available for rendering\n" };
	}

	/*
	OutputImage image1;
	image1.pushNewChannel(thisOpticalConfig->p_rawChannel->hp_raw, 555.0, thisOpticalConfig->p_rawChannel->m_dimension.y, thisOpticalConfig->p_rawChannel->m_dimension.x, 0, 0);
	image1.createOutputImage(OIC_XYZ);
	image1.displayRGB();
	*/

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "Rendering completed after " << duration << " ms\n";

	return { PI_OK, "Rendering completed!\n" };
}

PI_Message tracer::traceAndRender()
{
	//start clock
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	float* currentwavelength = nullptr;
	int runCounts = 0;

	for (; bool output = mainStorageManager.takeOne(currentwavelength, StorageHolder<float>::Status::uninitialized); runCounts++)
	{
		//trace all points
		activeWavelength = *currentwavelength;

		//get the corresponding optical config
		OpticalConfig* thisOpticalConfig = nullptr;
		bool b_getConfig = mainStorageManager.infoCheckOut(thisOpticalConfig, activeWavelength);
		if (!b_getConfig)
		{
			std::cout << "Cannot get optical config for this wavelength\n";
			return { PI_UNKNOWN_ERROR, "Cannot get optical config for this wavelength\n" };
		}

		//create a (hopefully clear) image on device
		thisOpticalConfig->p_rawChannel->createSibling();
		std::cout << "Tracing and Rendering at wavelength = " << activeWavelength << "\n";

		//run the tracer
		PI_running = true;
		while (ColumnCreator4() != -2)
		{
			while (KernelLauncher(0, nullptr) != -2)
			{
				//run the renderer
				while (KernelLauncher2(0, nullptr) != -2)
				{
					renderedCount = renderedCount + PI_renderJobSize;
					updateProgressRender();
					if (PI_cancelTraceRender)
					{
						thisOpticalConfig->p_rawChannel->copyFromSibling();
						thisOpticalConfig->p_rawChannel->deleteSibling();
						PI_running = false;
						return { PI_OK, "Operation cancelled!\n" };
					}
				}
			}
		}
		
		PI_running = false;

		//copy data out
		thisOpticalConfig->p_rawChannel->copyFromSibling();

		//performing clean up
		thisOpticalConfig->p_rawChannel->deleteSibling();

		//tell the storage manager that the job is done
		mainStorageManager.jobCheckIn(currentwavelength, StorageHolder<float>::Status::completed2);
	}

	if (runCounts == 0)
	{
		std::cout << "No wavelength is available for tracing\n";
		return { PI_UNKNOWN_ERROR, "No wavelength is available for tracing\n" };
	}

	//stop clock
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "Rendering completed after " << duration << " ms\n";

	return { PI_OK, "Rendering completed!\n" };
}

PI_Message tracer::showRaw(float* wavelengths, int count)
{
	if (count != 0)
	{
		for (int i = 0; i < count; i++)
		{
			float currentWavelength = wavelengths[i];

			//get the corresponding optical config
			OpticalConfig* thisOpticalConfig = nullptr;
			bool b_getConfig = mainStorageManager.infoCheckOut(thisOpticalConfig, currentWavelength);
			if (!b_getConfig)
			{
				std::cout << "Cannot get optical config for wavelength at "<<currentWavelength<<" nm \n";
				return { PI_UNKNOWN_ERROR, "Cannot get optical config for this wavelength\n" };
			}

			//display data with scaling
			void* mapX = nullptr;
			void* mapY = nullptr;
			SimpleRetinaDescriptor* tempDescriptor = dynamic_cast<SimpleRetinaDescriptor*>(thisOpticalConfig->p_retinaDescriptor);

			MYFLOATTYPE scalingargs[4] = { tempDescriptor->m_thetaR, tempDescriptor->m_R0, tempDescriptor->m_maxTheta, tempDescriptor->m_maxPhi };
			generateProjectionMap(mapX, mapY, thisOpticalConfig->p_rawChannel->m_dimension.y,
				thisOpticalConfig->p_rawChannel->m_dimension.x, PI_projectionMethod, 4, scalingargs);

			quickDisplayv2(thisOpticalConfig->p_rawChannel->hp_raw, thisOpticalConfig->p_rawChannel->m_dimension.y,
				thisOpticalConfig->p_rawChannel->m_dimension.x, mapX, mapY, PI_displayWindowSize);
		}
	}
	else
	{
		std::cout << "Nothing to show!\n";
		return { PI_INPUT_ERROR, "Empty selection\n" };
	}
	return { PI_OK, "Successful!\n" };
}

PI_Message tracer::showRGB(int uniqueID)
{
	std::unordered_map<int, OutputImage>::iterator token = outputImages.find(uniqueID);

	if (token == outputImages.end())
	{
		std::cout << "Image at ID " << uniqueID << " does not exist!\n";
		return { PI_INPUT_ERROR, "Image ID does not exist!\n" };
	}
	else
	{
		//just take a random optical config, cause they're all have the same retina descriptor, FOR NOW
		OpticalConfig* thisOpticalConfig = nullptr;
		bool b_getConfig = mainStorageManager.infoCheckOut(thisOpticalConfig, activeWavelength);
		if (!b_getConfig)
		{
			std::cout << "Cannot get optical config for wavelength at " << activeWavelength << " nm \n";
			return { PI_UNKNOWN_ERROR, "Cannot get optical config for this wavelength\n" };
		}

		//obtain the projection maps
		//display data with scaling
		void* mapX = nullptr;
		void* mapY = nullptr;
		SimpleRetinaDescriptor* tempDescriptor = dynamic_cast<SimpleRetinaDescriptor*>(thisOpticalConfig->p_retinaDescriptor);

		MYFLOATTYPE scalingargs[4] = { tempDescriptor->m_thetaR, tempDescriptor->m_R0, tempDescriptor->m_maxTheta, tempDescriptor->m_maxPhi };
		generateProjectionMap(mapX, mapY, thisOpticalConfig->p_rawChannel->m_dimension.y,
			thisOpticalConfig->p_rawChannel->m_dimension.x, PI_projectionMethod, 4, scalingargs);

		outputImages[uniqueID].displayRGB(-1,-1,mapX, mapY);
		return { PI_OK, "Successful!\n" };
	}
}

PI_Message tracer::saveRaw(const char * path, int uniqueID)
{
	std::unordered_map<int, OutputImage>::iterator token = outputImages.find(uniqueID);

	if (token == outputImages.end())
	{
		std::cout << "Image at ID " << uniqueID << " does not exist!\n";
		return { PI_INPUT_ERROR, "Image ID does not exist!\n" };
	}
	else
	{
		outputImages[uniqueID].saveRaw(path);
		return { PI_OK, "Successful!\n" };
	}
}

PI_Message tracer::saveRGB(const char * path, int uniqueID)
{
	std::unordered_map<int, OutputImage>::iterator token = outputImages.find(uniqueID);

	if (token == outputImages.end())
	{
		std::cout << "Image at ID " << uniqueID << " does not exist!\n";
		return { PI_INPUT_ERROR, "Image ID does not exist!\n" };
	}
	else
	{
		std::cout << "Saving RGB...\n";

		//just take a random optical config, cause they're all have the same retina descriptor, FOR NOW
		OpticalConfig* thisOpticalConfig = nullptr;
		bool b_getConfig = mainStorageManager.infoCheckOut(thisOpticalConfig, activeWavelength);
		if (!b_getConfig)
		{
			std::cout << "Cannot get optical config for wavelength at " << activeWavelength << " nm \n";
			return { PI_UNKNOWN_ERROR, "Cannot get optical config for this wavelength\n" };
		}

		//obtain the projection maps
		//display data with scaling
		void* mapX = nullptr;
		void* mapY = nullptr;
		SimpleRetinaDescriptor* tempDescriptor = dynamic_cast<SimpleRetinaDescriptor*>(thisOpticalConfig->p_retinaDescriptor);

		MYFLOATTYPE scalingargs[4] = { tempDescriptor->m_thetaR, tempDescriptor->m_R0, tempDescriptor->m_maxTheta, tempDescriptor->m_maxPhi };
		generateProjectionMap(mapX, mapY, thisOpticalConfig->p_rawChannel->m_dimension.y,
			thisOpticalConfig->p_rawChannel->m_dimension.x, PI_projectionMethod, 4, scalingargs);

		outputImages[uniqueID].saveRGB(path, mapX, mapY);
		std::cout << "Saved to " << path << " \n";
		return { PI_OK, "Successful!\n" };
	}
}

PI_Message tracer::setLinearRayDensity(unsigned int newDensity)
{
	if (newDensity > 1)
	{
		PI_linearRayDensity = newDensity;
		std::stringstream response;
		response << "Setting linear ray generation density to " << newDensity << "successfully!\n";
		std::cout << response.str();
		return { PI_OK, response.str().c_str() };
	}
	else
	{
		std::stringstream response;
		response << "Cannot set linear ray generation density to " << newDensity << "!\n";
		std::cout << response.str();
		return { PI_INPUT_ERROR, response.str().c_str() };
	}
}

PI_Message tracer::getLinearRayDensity()
{
	std::stringstream response;
	response << "Linear ray density is " << PI_linearRayDensity << std::endl;
	return { PI_INFO_RESPONSE, response.str().c_str() };
}

PI_Message tracer::createOutputImage(int count, float* wavelengthList, int& uniqueID)
{
	if (count != 0)
	{
		std::cout << "Creating output image...\n";
		for (bool quit = false; quit == false;)
		{
			//generate a random ID
			uniqueID = rand() % 100; //max 100 output images (can be any number that fits the memory)

			//find, if random ID not exist, push new output image
			std::unordered_map<int, OutputImage>::iterator token = outputImages.find(uniqueID);

			if (token == outputImages.end())
			{
				outputImages.emplace(std::piecewise_construct,
					std::forward_as_tuple(uniqueID),  // args for key
					std::forward_as_tuple());  // args for mapped value
				quit = true;
			}
		}
		
		//check for wavelength info from the main storage
		float* wavelengthStorageList = nullptr;
		int wavelengthStorageCount = 0;
		StorageHolder<float>::Status* statusStorageList = nullptr;
		if (!mainStorageManager.infoCheckOut(wavelengthStorageList, statusStorageList, wavelengthStorageCount))
		{
			return { PI_UNKNOWN_ERROR, "No wavelength available!\n" };
		}

		for (int i = 0; i < count; i++)
		{
			//check wavelength status
			int index = 0;
			for (; index < wavelengthStorageCount; index++)
			{
				if (wavelengthList[i] == wavelengthStorageList[index] 
					&& statusStorageList[index] == StorageHolder<float>::Status::completed2)
				{
					break; //wavelength is OK
				}
			}
			if (index == wavelengthStorageCount)
			{
				continue; //wavelength not found
			}

			//if ok, get the optical config and push new image channel
			OpticalConfig* config = nullptr;
			if (!(mainStorageManager.infoCheckOut(config, wavelengthList[i])))
			{
				continue; //optical config for this wavelength not found
			}
			
			outputImages[uniqueID].pushNewChannel(config->p_rawChannel->hp_raw, wavelengthList[i], config->p_rawChannel->m_dimension.y, config->p_rawChannel->m_dimension.x, 0, 0);
		}

		//call createOutputImage
		outputImages[uniqueID].createOutputImage(PI_rawFormat);
		outputImages[uniqueID].displayRGB();

		//clean up
		delete[] wavelengthStorageList;
		delete[] statusStorageList;

		std::cout << "Output image created!\n";

		return { PI_OK, "Output image created from selected wavelengths!\n" };
	}
	else
	{
		return { PI_INPUT_ERROR, "No wavelengths selected!\n" };
	}
}

PI_Message tracer::deleteOutputImage(int uniqueID)
{
	//find
	auto token = outputImages.find(uniqueID);

	//if ID not exist return false
	if (token == outputImages.end())
	{
		return { PI_INPUT_ERROR, "Image ID does not exist!\n" };
	}
	else //else delete at the iterator
	{
		outputImages.erase(uniqueID);
		return { PI_OK, "Image deleted!\n" };
	}
}

PI_Message tracer::clearStorage()
{
	std::cout << "Clearing storage! \n";
	//clear the progress counter variables
	toTraceCount = 0;
	tracedCount = 0;
	toRenderCount = 0;
	renderedCount = 0;
	PI_traceProgress = 0;
	PI_renderProgress = 0;

	//restore old status
	maximizeContrast = true;

	//check for wavelength info from the main storage
	float* wavelengthStorageList = nullptr;
	int wavelengthStorageCount = 0;
	StorageHolder<float>::Status* statusStorageList = nullptr;
	if (!mainStorageManager.infoCheckOut(wavelengthStorageList, statusStorageList, wavelengthStorageCount))
		{
			return { PI_OK, "No wavelength to clean!\n" };
		}

	for (int i = 0; i < wavelengthStorageCount; i++)
	{
		std::cout << "Deleting configuration at " << wavelengthStorageList[i] << " nm \n";
		mainStorageManager.pleaseDelete(wavelengthStorageList[i]);
	}

	return { PI_OK, "Storage cleared!\n" };
}

void updateProgressTrace()
{
	if (toTraceCount == 0)
	{
		PI_traceProgress = 0;
	}
	else
	{
		if (tracedCount < 0)
		{
			tracedCount = 0;
		}
		if (tracedCount > toTraceCount)
		{
			tracedCount = toTraceCount;
		}
		PI_traceProgress = (float)tracedCount / (float)toTraceCount;
	}
	std::cout << "Estimated Trace Progress: " << PI_traceProgress * 100 << " %\n";
}

void updateProgressRender()
{
	if (toRenderCount == 0)
	{
		PI_renderProgress = 0;
	}
	else
	{
		if (renderedCount < 0)
		{
			renderedCount = 0;

		}
		if (renderedCount > toRenderCount)
		{
			renderedCount = toRenderCount;
		}
		PI_renderProgress = (float)renderedCount / (float)toRenderCount;
	}

	std::cout << "Estimated Render Progress: " << PI_renderProgress * 100 << " %\n";
}

PI_Message tracer::getProgress(float & traceProgress, float & renderProgress)
{
	traceProgress = PI_traceProgress;
	renderProgress = PI_renderProgress;
	return { PI_OK, "Successful!\n" };
}

PI_Message tracer::getVRAMUsageInfo(unsigned long & total, unsigned long & free)
{
	size_t totalvram = 0, freevram = 0;
	CUDARUN(cudaMemGetInfo(&freevram, &totalvram));
	total = totalvram;
	free = freevram;
	return { PI_OK, "Successful!\n" };
}

PI_Message tracer::importImage(const char * path, float posX, float posY, float posZ, float sizeHorz, float sizeVert, float rotX, float rotY, float rotZ, float brightness)
{
	//open CV is needed, so the main work is not done here
	std::vector<tracer::PI_LuminousPoint> inputPoints;
	bool result = importImageCV(inputPoints, path, posX, posY, posZ, sizeHorz, sizeVert, rotX, rotY, rotZ, brightness);
	if (result)
	{
		int newImagePoints = 0;
		for (auto point : inputPoints)
		{
			LuminousPoint toAdd;
			toAdd.x = point.x;
			toAdd.y = point.y;
			toAdd.z = point.z;
			toAdd.intensity = point.intensity;
			toAdd.wavelength = point.wavelength;
			bool result2 = mainStorageManager.add(toAdd);
			if (result2)
			{
				toTraceCount++;
				toRenderCount++;
				newImagePoints++;
			}
			else
			{
				std::cout << "Could not add point (" << toAdd.x << ", " << toAdd.y << " ," << toAdd.z << "), " << toAdd.intensity << " at " << toAdd.wavelength << " nm!\n";
			}
		}
		std::cout << newImagePoints << " points have been added from the image\n";

		//disable contrast maximization to conserve correct image brightness
		//maximizeContrast = false;

		return { PI_OK, "Image import successfully!\n" };
	}
	else
	{
		std::cout << "Cannot import image at " << path << " \n";
		return { PI_INPUT_ERROR, "Cannot import image!\n" };
	}
}

PI_Message tracer::getImagePrimaryWavelengths(float & wavelengthR, float & wavelengthG, float & wavelengthB)
{
	wavelengthR = PI_primaryWavelengthR;
	wavelengthG = PI_primaryWavelengthG;
	wavelengthB = PI_primaryWavelengthB;
	return { PI_OK, "Successful!\n" };
}

PI_Message tracer::getPreferences(PI_Preferences & prefs)
{
	prefs.ThreadsPerKernelLaunch = PI_ThreadsPerKernelLaunch;
	prefs.linearRayDensity = PI_linearRayDensity;
	switch (PI_rgbStandard)
	{
	case IF_SRGB:
		prefs.rgbStandard = PI_SRGB;
		break;
	case IF_ADOBERGB:
	default:
		prefs.rgbStandard = PI_ADOBERGB;
		break;
	}
	prefs.traceJobSize = PI_traceJobSize;
	prefs.renderJobSize = PI_renderJobSize;
	switch (PI_rawFormat)
	{
	case OIC_LMS:
		prefs.rawFormat = PI_LMS; 
		break;
	case OIC_XYZ:
	default:
		prefs.rawFormat = PI_XYZ; 
		break;
	}
	switch (PI_projectionMethod)
	{
	case PI_PROJECTION_PLATE_CARREE:
		prefs.projectionMethod = PI_PROJECTION_PLATE_CARREE;
		break;
	case PI_PROJECTION_NONE:
		prefs.projectionMethod = PI_PROJECTION_NONE;
		break;
	case PI_PROJECTION_ALONGZ:
	default:
		prefs.projectionMethod = PI_PROJECTION_ALONGZ;
		break;
	}
	prefs.displayWindowSize = PI_displayWindowSize;
	prefs.primaryWavelengthR = PI_primaryWavelengthR;
	prefs.primaryWavelengthG = PI_primaryWavelengthG;
	prefs.primaryWavelengthB = PI_primaryWavelengthB;
	prefs.maxParallelThread = PI_maxParallelThread;
	prefs.performanceTestRepetition = PI_performanceTestRepetition;

	return { PI_OK, "Successfully!\n" };
}

PI_Message tracer::setPreferences(PI_Preferences & prefs)
{
	if (prefs.ThreadsPerKernelLaunch <= 8)
	{
		PI_ThreadsPerKernelLaunch = 8;
	}
	else if (prefs.ThreadsPerKernelLaunch >= 128)
	{
		PI_ThreadsPerKernelLaunch = 128;
	}
	else
	{
		PI_ThreadsPerKernelLaunch = prefs.ThreadsPerKernelLaunch;
	}

	if (prefs.linearRayDensity <= 1)
	{
		PI_linearRayDensity = 1;
	}
	else if (prefs.linearRayDensity >= 200)
	{
		PI_linearRayDensity = 200;
	}
	else
	{
		PI_linearRayDensity = prefs.linearRayDensity;
	}
	
	switch (prefs.rgbStandard)
	{
	case PI_SRGB:
		PI_rgbStandard = IF_SRGB;
		break;
	case PI_ADOBERGB:
	default:
		PI_rgbStandard = IF_ADOBERGB;
		break;
	}
	if (prefs.traceJobSize <= 1)
	{
		PI_traceJobSize = 1;
	}
	else if (prefs.traceJobSize >= 100)
	{
		PI_traceJobSize = 100;
	}
	else
	{
		PI_traceJobSize = prefs.traceJobSize;
	}
	
	if (prefs.renderJobSize <= 1)
	{
		PI_renderJobSize = 1;
	}
	else if (prefs.renderJobSize >= 100)
	{
		PI_renderJobSize = 100;
	}
	else
	{
		PI_renderJobSize = prefs.renderJobSize;
	}
	
	switch (prefs.rawFormat)
	{
	case PI_LMS:
		PI_rawFormat = OIC_LMS;
		break;
	case PI_XYZ:
	default:
		PI_rawFormat = OIC_XYZ;
		break;
	}
	switch (prefs.projectionMethod)
	{
	case PI_PROJECTION_PLATE_CARREE:
		PI_projectionMethod = IF_PROJECTION_PLATE_CARREE;
		break;
	case PI_PROJECTION_NONE:
		PI_projectionMethod = IF_PROJECTION_NONE;
		break;
	case PI_PROJECTION_ALONGZ:
	default:
		PI_projectionMethod = IF_PROJECTION_ALONGZ;
		break;
	}
	if (prefs.displayWindowSize <= 50)
	{
		PI_displayWindowSize = 50;
	}
	else if (prefs.displayWindowSize >= 2048)
	{
		PI_displayWindowSize = 2048;
	}
	else
	{
		PI_displayWindowSize = prefs.displayWindowSize;
	}
	
	if (prefs.primaryWavelengthR <= 610)
	{
		PI_primaryWavelengthR = 610;
	}
	else if (prefs.primaryWavelengthR >= 630)
	{
		PI_primaryWavelengthR = 630;
	}
	else
	{
		PI_primaryWavelengthR = prefs.primaryWavelengthR;
	}
	
	if (prefs.primaryWavelengthG <= 520)
	{
		PI_primaryWavelengthG = 520;
	}
	else if (prefs.primaryWavelengthG >= 540)
	{
		PI_primaryWavelengthG = 540;
	}
	else
	{
		PI_primaryWavelengthG = prefs.primaryWavelengthG;
	}
	
	if (prefs.primaryWavelengthB <= 460)
	{
		PI_primaryWavelengthB = 460;
	}
	else if (prefs.primaryWavelengthB >= 475)
	{
		PI_primaryWavelengthB = 475;
	}
	else
	{
		PI_primaryWavelengthB = prefs.primaryWavelengthB;
	}
	
	int maxConcurency = std::thread::hardware_concurrency();
	PI_maxParallelThread = (((prefs.maxParallelThread <= 0) ? 1 : prefs.maxParallelThread) >= maxConcurency) ? maxConcurency : prefs.maxParallelThread;

	PI_performanceTestRepetition = (((prefs.performanceTestRepetition <= 1) ? 1 : prefs.performanceTestRepetition) >= 2000000000) ? 2000000000 : prefs.performanceTestRepetition;


	return { PI_OK, "Successfully!\n" };
}

PI_Message tracer::defaultPreference()
{
	PI_ThreadsPerKernelLaunch = 32;
	PI_linearRayDensity = 25;
	PI_rgbStandard = IF_ADOBERGB;
	PI_traceJobSize = 10;
	PI_renderJobSize = 10;
	PI_rawFormat = OIC_XYZ;
	PI_projectionMethod = IF_PROJECTION_PLATE_CARREE;
	PI_displayWindowSize = 800;
	PI_primaryWavelengthR = 620;
	PI_primaryWavelengthG = 530;
	PI_primaryWavelengthB = 465;
	PI_maxParallelThread = std::thread::hardware_concurrency();
	PI_performanceTestRepetition = 100000;

	return { PI_OK, "Successful!\n" };
}

PI_Message tracer::drawOpticalConfig(float wavelength, bool suppressRefractiveSurfaceTexture, bool suppressImageTexture)
{
	//load the optical config at wavelength
	OpticalConfig* thisOpticalConfig = nullptr;
	bool b_getConfig = mainStorageManager.infoCheckOut(thisOpticalConfig, wavelength);
	if (!b_getConfig)
	{
		std::cout << "Cannot get optical config for wavelength at " << wavelength << " nm \n";
		return { PI_UNKNOWN_ERROR, "Cannot get optical config for this wavelength\n" };
	}

	//parse the config and create draw data for surfaces
	std::vector<SurfaceDrawInfo> surfaceInfos(thisOpticalConfig->numofsurfaces);
	for (int i = 0; i < surfaceInfos.size(); i++)
	{
		SurfaceDrawInfo& currentsurface = surfaceInfos[i];
		auto p_surfacedata = dynamic_cast<quadricsurface<MYFLOATTYPE>*>(thisOpticalConfig->surfaces[i]);
		currentsurface.posX = (float)p_surfacedata->pos.x;
		currentsurface.posY = (float)p_surfacedata->pos.y;
		currentsurface.posZ = (float)p_surfacedata->pos.z;
		currentsurface.asphericity = (float)p_surfacedata->param.C - 1.0f;
		currentsurface.radius = (float)abs(p_surfacedata->param.I) / 2.0f;
		currentsurface.radiusSign = (p_surfacedata->param.I < 0.0) ? true : false;
		currentsurface.isFlat = (p_surfacedata->param.I == 0.0) ? true : false;
		currentsurface.diameter = (float)p_surfacedata->diameter;
		currentsurface.rings = (i == (surfaceInfos.size() - 1)) ? PI_imageSurfaceRings : PI_refractiveSurfaceRings;
		currentsurface.arms = (i == (surfaceInfos.size() - 1)) ? PI_imageSurfaceArms : PI_refractiveSurfaceArms;

	}

	//load texture
	std::vector<unsigned char*> textureDirectory(thisOpticalConfig->numofsurfaces);
	for (int i = 0; i < surfaceInfos.size() - 1; i++)//not including image surface
	{
		SurfaceDrawInfo& currentsurface = surfaceInfos[i];
		auto p_surfacedata = dynamic_cast<quadricsurface<MYFLOATTYPE>*>(thisOpticalConfig->surfaces[i]);

		if (p_surfacedata->data_size != 0 && p_surfacedata->p_data != nullptr && suppressRefractiveSurfaceTexture == false)
		{
			//read and parse data from mysurface class
			float* p_datareader = reinterpret_cast<float*>(p_surfacedata->p_data);
			int rows = static_cast<int>(p_datareader[0]);
			int cols = static_cast<int>(p_datareader[1]);
			float* p_datastart = &(p_datareader[2]);

			//create draw data for texture
			unsigned char* p_tempOutput = nullptr;
			bool output = generateGLDrawTexture(p_tempOutput, p_datastart, rows, cols);
			if (!output)
			{
				textureDirectory.push_back(nullptr);
			}
			else
			{
				//assign data to current surface
				currentsurface.texChannels = 1;
				currentsurface.texRows = rows;
				currentsurface.texCols = cols;
				currentsurface.p_tex = p_tempOutput;

				//save pointer to texturecache
				textureDirectory.push_back(p_tempOutput);
			}
		}
		else
		{
			textureDirectory.push_back(nullptr);
		}
	}

	//load texture for the image surface
	{
		SurfaceDrawInfo& currentsurface = surfaceInfos[surfaceInfos.size() - 1];
		if (!suppressImageTexture)
		{
			//obtain the projection maps
			//display data with scaling
			void* mapX = nullptr;
			void* mapY = nullptr;
			SimpleRetinaDescriptor* tempDescriptor = dynamic_cast<SimpleRetinaDescriptor*>(thisOpticalConfig->p_retinaDescriptor);

			MYFLOATTYPE scalingargs[4] = { tempDescriptor->m_thetaR, tempDescriptor->m_R0, tempDescriptor->m_maxTheta, tempDescriptor->m_maxPhi };
			generateProjectionMap(mapX, mapY, thisOpticalConfig->p_rawChannel->m_dimension.y,
				thisOpticalConfig->p_rawChannel->m_dimension.x, IF_PROJECTION_ALONGZ, 4, scalingargs);

			
			unsigned char* p_tempOutput = nullptr;
			int rows = thisOpticalConfig->p_rawChannel->m_dimension.y;
			int cols = thisOpticalConfig->p_rawChannel->m_dimension.x;
			bool output = generateGLDrawTextureImage(p_tempOutput, reinterpret_cast<char*>(thisOpticalConfig->p_rawChannel->hp_raw),
				rows,cols, mapX, mapY);
			if (!output)
			{
				textureDirectory.push_back(nullptr);
			}
			else
			{
				//assign data to current surface
				currentsurface.texChannels = 1;
				currentsurface.texRows = rows;
				currentsurface.texCols = cols;
				currentsurface.p_tex = p_tempOutput;

				//save pointer to texturecache
				textureDirectory.push_back(p_tempOutput);
			}
		}
		else
		{
			textureDirectory.push_back(nullptr);
		}
	}
	
	//draw call
	drawSurfaces(surfaceInfos, suppressRefractiveSurfaceTexture, suppressImageTexture);

	//clean up the data
	clearGLDrawTexture();
	surfaceInfos.clear();
	textureDirectory.clear();

	return { PI_OK, "Successful!\n" };
}

PI_Message tracer::drawImage(int uniqueID)
{
	std::unordered_map<int, OutputImage>::iterator token = outputImages.find(uniqueID);

	if (token == outputImages.end())
	{
		std::cout << "Image at ID " << uniqueID << " does not exist!\n";
		return { PI_INPUT_ERROR, "Image ID does not exist!\n" };
	}
	else
	{
		//load the optical config at wavelength
		OpticalConfig* thisOpticalConfig = nullptr;
		bool b_getConfig = mainStorageManager.infoCheckOut(thisOpticalConfig, activeWavelength);
		if (!b_getConfig)
		{
			std::cout << "Cannot get optical config for wavelength at " << activeWavelength << " nm \n";
			return { PI_UNKNOWN_ERROR, "Cannot get optical config for this wavelength\n" };
		}

		//parse the config and create draw data for surfaces
		std::vector<SurfaceDrawInfo> surfaceInfos(1);
		SurfaceDrawInfo& currentsurface = surfaceInfos[0];
		auto p_surfacedata = dynamic_cast<quadricsurface<MYFLOATTYPE>*>(thisOpticalConfig->surfaces[thisOpticalConfig->numofsurfaces - 1]);
		currentsurface.posX = (float)p_surfacedata->pos.x;
		currentsurface.posY = (float)p_surfacedata->pos.y;
		currentsurface.posZ = (float)p_surfacedata->pos.z;
		currentsurface.asphericity = (float)p_surfacedata->param.C - 1.0f;
		currentsurface.radius = (float)abs(p_surfacedata->param.I) / 2.0f;
		currentsurface.radiusSign = (p_surfacedata->param.I < 0.0) ? true : false;
		currentsurface.isFlat = (p_surfacedata->param.I == 0.0) ? true : false;
		currentsurface.diameter = (float)p_surfacedata->diameter;
		currentsurface.rings = PI_imageSurfaceRings;
		currentsurface.arms = PI_imageSurfaceArms;

		//load texture
		OutputImage& currentImage = outputImages[uniqueID];
		void* mapX = nullptr;
		void* mapY = nullptr;
		SimpleRetinaDescriptor* tempDescriptor = dynamic_cast<SimpleRetinaDescriptor*>(thisOpticalConfig->p_retinaDescriptor);

		MYFLOATTYPE scalingargs[4] = { tempDescriptor->m_thetaR, tempDescriptor->m_R0, tempDescriptor->m_maxTheta, tempDescriptor->m_maxPhi };
		generateProjectionMap(mapX, mapY, thisOpticalConfig->p_rawChannel->m_dimension.y,
			thisOpticalConfig->p_rawChannel->m_dimension.x, IF_PROJECTION_ALONGZ, 4, scalingargs);

		unsigned char* p_tempOutput = nullptr;
		int rows = thisOpticalConfig->p_rawChannel->m_dimension.y;
		int cols = thisOpticalConfig->p_rawChannel->m_dimension.x;
		bool output = currentImage.generateGLDrawTexture(p_tempOutput ,rows, cols, mapX, mapY);
		if (!output)
		{
			return { PI_UNKNOWN_ERROR, "Error generating texture from output image\n" };
		}
		else
		{
			//assign data to current surface
			currentsurface.texChannels = 1;
			currentsurface.texRows = rows;
			currentsurface.texCols = cols;
			currentsurface.p_tex = p_tempOutput;
		}

		//draw call
		drawSurfaces(surfaceInfos, false, false);

		//clean up the data
		currentImage.clearGLDrawTextureCache();
		surfaceInfos.clear();

		return { PI_OK, "Successful!\n" };
	}
}

PI_Message tracer::pauseTraceRender()
{
	std::cout << "Cancelling...\n";
	PI_cancelTraceRender = true;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	while (PI_running)
	{
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		if (duration > 10000)//10s watchdog timer
		{
			PI_cancelTraceRender = false;
			return { PI_UNKNOWN_ERROR, "API error: Cannot cancel operation!\n" };
		}
	}
	PI_cancelTraceRender = false;
	return { PI_OK, "Successful!\n" };
}
