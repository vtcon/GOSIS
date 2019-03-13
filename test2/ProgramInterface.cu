#include "ProgramInterface.h"
#include "StorageManager.cuh"
#include "ManagerFunctions.cuh"
#include "Auxiliaries.cuh"
#include "../ConsoleApplication/src/ImageFacilities.h"
#include "../ConsoleApplication/src/OutputImage.h"

#include <list>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cstdlib> //for the "rand" function

//for the console:
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <windows.h>

using namespace tracer;

//internal global variables, please think VERY carefully before modifying these lines
std::list<StorageManager> repoList;
StorageManager mainStorageManager;
float activeWavelength = 555.0; //pls don't break this line
std::unordered_map<int, OutputImage> outputImages;
bool maximizeContrast = true; //should be set to true if inputs are points, and false if input is an image

//external global variables
int PI_ThreadsPerKernelLaunch = 16;
int PI_linearRayDensity = 30;//20 is ok
unsigned int PI_rgbStandard = IF_ADOBERGB;
int PI_traceJobSize = 3;
int PI_renderJobSize = 3;
unsigned short int PI_rawFormat = OIC_XYZ; //OIC_LMS
unsigned int PI_projectionMethod = IF_PROJECTION_PLATE_CARREE;
int PI_displayWindowSize = 800;
float PI_primaryWavelengthR = 620;
float PI_primaryWavelengthG = 530;
float PI_primaryWavelengthB = 465;

//these variables serves the progress count
float PI_traceProgress; //from 0.0 to 1.0
float PI_renderProgress; //from 0.0 to 1.0
long toTraceCount = 0;
long tracedCount = 0;
long toRenderCount = 0;
long renderedCount = 0;
void updateProgressTrace();
void updateProgressRender();


//function definitions
bool PI_LuminousPoint::operator==(const PI_LuminousPoint & rhs) const
{
	if (x == rhs.x &&y == rhs.y &&z == rhs.z &&wavelength == rhs.wavelength &&intensity == rhs.intensity)
		return true;
	else
		return false;
}

extern bool runTestOpenCV;
PI_Message tracer::test()
{
	
	if (runTestOpenCV)
	{
		maximizeContrast = false;
		testopencv();
		maximizeContrast = true;
	}
	else
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
			surfaces[0].refractiveIndex = 1.5168; surfaces[0].asphericity = 0.95;
			surfaces[0].apodization = PI_APD_CUSTOM; surfaces[0].customApoPath = "C:/testcustomapo.jpg";

			surfaces[1].diameter = 40.0; surfaces[1].radius = -60.0;
			float angularResol = 0.16;//0.16 is OK

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
		toAdd[i].diameter = (abs(toAdd[i].diameter) < 2.0*abs(toAdd[i].radius)) ? abs(toAdd[i].diameter) : 2.0*abs(toAdd[i].radius);
		bool output = constructSurface((newConfig->surfaces)[i], MF_REFRACTIVE, vec3<MYFLOATTYPE>(toAdd[i].x, toAdd[i].y, toAdd[i].z), abs(toAdd[i].radius), abs(toAdd[i].diameter), curveSign, previousN, toAdd[i].refractiveIndex, toAdd[i].asphericity, toAdd[i].apodization, toAdd[i].customApoPath);
		if (!output) return { PI_UNKNOWN_ERROR,"Error adding surface" };
		previousN = toAdd[i].refractiveIndex;
	}

	//test apo function
	//(newConfig->surfaces)[0]->apodizationType = APD_BARTLETT;

	//for the final image surface
	//todo: data integrity check here
	toAdd[numofsurfaces - 1].diameter = (abs(toAdd[numofsurfaces - 1].diameter) < 2.0*abs(toAdd[numofsurfaces - 1].radius)) ? abs(toAdd[numofsurfaces - 1].diameter) : 2.0*abs(toAdd[numofsurfaces - 1].radius);
	bool output = constructSurface((newConfig->surfaces)[numofsurfaces-1], MF_IMAGE, vec3<MYFLOATTYPE>(toAdd[count-1].x, toAdd[count - 1].y, toAdd[count - 1].z), -abs(toAdd[count - 1].radius), toAdd[count - 1].diameter, MF_CONCAVE, 1.0, FP_INFINITE, toAdd[count-1].asphericity);

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
		MYFLOATTYPE Rsqr = abs(dynamic_cast<quadricsurface<MYFLOATTYPE>*>(newConfig->surfaces[numofsurfaces - 1])->param.J);
		R = sqrt(Rsqr);
	}

	//angular extend actually depends on diameter and curvature radius!
	//my choice: if angular extend is smaller than the diameter allows, take the angular
	//if not, take the from-the-diameter-resulting extend
	
	MYFLOATTYPE maxAngularExtend = asin(toAdd[numofsurfaces - 1].diameter / (2.0*R)) / MYPI * 180.0;
	angularExtend = (angularExtend < maxAngularExtend) ? angularExtend : maxAngularExtend;
	
	p_retinaDescriptor = new SimpleRetinaDescriptor(angularResolution, R, angularExtend, angularExtend); //doesn't need to explicitly delete this
	newConfig->createImageChannel(p_retinaDescriptor);

	return { PI_OK, "Optical Config added" };
}

PI_Message tracer::checkData()
{
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
		while (ColumnCreator3() != -2);
		mainStorageManager.jobCheckIn(currentwavelength, StorageHolder<float>::Status::initialized);
	}

	if (runCounts == 0)
	{
		std::cout << "No wavelength is available for checking\n";
		return { PI_UNKNOWN_ERROR, "No wavelength is available for checking\n" };
	}

	return { PI_OK, "Check successful!\n" };
}

PI_Message tracer::trace()
{
	float* currentwavelength = nullptr;
	int runCounts = 0;
	for (; bool output = mainStorageManager.takeOne(currentwavelength, StorageHolder<float>::Status::initialized); runCounts++)
	{
		//trace all points
		activeWavelength = *currentwavelength;
		std::cout << "Tracing at wavelength = " << activeWavelength << "\n";
		while (KernelLauncher(0, nullptr) != -2)
		{
			tracedCount = tracedCount + PI_traceJobSize;
			updateProgressTrace();
		}

		mainStorageManager.jobCheckIn(currentwavelength, StorageHolder<float>::Status::completed1);
	}
	
	if (runCounts == 0)
	{
		std::cout << "No wavelength is available for tracing\n";
		return { PI_UNKNOWN_ERROR, "No wavelength is available for tracing\n" };
	}

	return { PI_OK, "Trace successful!\n" };
}

PI_Message tracer::render()
{
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
		while (KernelLauncher2(0, nullptr) != -2)
		{
			renderedCount = renderedCount + PI_renderJobSize;
			updateProgressRender();
		}

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
		outputImages[uniqueID].saveRGB(path);
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

PI_Message tracer::getVRAMUsageInfo(long & total, long & free)
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

	return { PI_OK, "Successfully!\n" };
}

PI_Message tracer::setPreferences(PI_Preferences & prefs)
{
	if (prefs.ThreadsPerKernelLaunch <= 8)
	{
		PI_ThreadsPerKernelLaunch = 8;
	}
	else if (prefs.ThreadsPerKernelLaunch >= 32)
	{
		PI_ThreadsPerKernelLaunch = 32;
	}
	else
	{
		PI_ThreadsPerKernelLaunch = 16;
	}

	if (prefs.linearRayDensity <= 1)
	{
		PI_linearRayDensity = 1;
	}
	else if (prefs.linearRayDensity >= 50)
	{
		PI_linearRayDensity = 50;
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
	else if (prefs.traceJobSize >= 10)
	{
		PI_traceJobSize = 10;
	}
	else
	{
		PI_traceJobSize = prefs.traceJobSize;
	}
	
	if (prefs.renderJobSize <= 1)
	{
		PI_renderJobSize = 1;
	}
	else if (prefs.renderJobSize >= 10)
	{
		PI_renderJobSize = 10;
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
	

	return { PI_OK, "Successfully!\n" };
}

PI_Message tracer::defaultPreference()
{
	PI_ThreadsPerKernelLaunch = 16;
	PI_linearRayDensity = 30;
	PI_rgbStandard = IF_ADOBERGB;
	PI_traceJobSize = 3;
	PI_renderJobSize = 3;
	PI_rawFormat = OIC_XYZ;
	PI_projectionMethod = IF_PROJECTION_PLATE_CARREE;
	PI_displayWindowSize = 800;
	PI_primaryWavelengthR = 620;
	PI_primaryWavelengthG = 530;
	PI_primaryWavelengthB = 465;

	return { PI_OK, "Successful!\n" };
}
