#include "ProgramInterface.h"
#include "StorageManager.cuh"
#include "ManagerFunctions.cuh"
#include "Auxiliaries.cuh"
#include "../ConsoleApplication/src/ImageFacilities.h"

#include <list>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cstdlib> //for the "rand" function

using namespace tracer;

//internal global variables
std::list<StorageManager> repoList;
StorageManager mainStorageManager;
float activeWavelength = 555.0; //pls don't break this line
std::unordered_map<int, OutputImage> outputImages;

//external global variables
int PI_ThreadsPerKernelLaunch = 16;
int PI_linearRayDensity = 30;//20 is ok

//function definitions
bool PI_LuminousPoint::operator==(const PI_LuminousPoint & rhs) const
{
	if (x == rhs.x &&y == rhs.y &&z == rhs.z &&wavelength == rhs.wavelength &&intensity == rhs.intensity)
		return true;
	else
		return false;
}

PI_Message tracer::test()
{
	int count = 3;
	PI_Surface* surfaces = new PI_Surface[count];
	surfaces[0].z = 40.0; surfaces[0].diameter = 40.0; surfaces[0].radius = 40.0; surfaces[0].refractiveIndex = 2.0;
	surfaces[1].z = 10.0; surfaces[1].diameter = 40.0; surfaces[1].radius = -40.0; surfaces[1].refractiveIndex = 1.0;
	surfaces[2].diameter = 40.0; surfaces[2].radius = -60.0;
	float angularResol = 2.0;
	float angularExtend = 90.0;


	addOpticalConfigAt(555.0, count, surfaces, angularResol, angularExtend);

	int pcount = 1;
	
	PI_LuminousPoint point;
	point.x = 0.0;
	point.y = 0.0;
	point.z = 100.0;
	addPoint(point);
	
	checkData();
	trace();
	render();

	delete[] surfaces;

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
	mainStorageManager.add(point);
	return PI_Message();
}

PI_Message tracer::addOpticalConfigAt(float wavelength, int count, PI_Surface *& toAdd, float angularResolution, float angularExtend)
{
	int numofsurfaces = count;
	float wavelength1 = wavelength;

	//check out an opticalconfig as output
	OpticalConfig* newConfig = nullptr;
	mainStorageManager.jobCheckOut(newConfig, numofsurfaces, wavelength1);

	float previousN = 1.0;
	//for all the refractive surfaces
	for (int i = 0; i < count-1; i++)
	{
		//todo: data integrity check here

		auto curveSign = (toAdd[i].radius > 0) ? MF_CONVEX : MF_CONCAVE;
		toAdd[i].diameter = (abs(toAdd[i].diameter) < 2.0*abs(toAdd[i].radius)) ? abs(toAdd[i].diameter) : 2.0*abs(toAdd[i].radius);
		bool output = constructSurface((newConfig->surfaces)[i], MF_REFRACTIVE, vec3<MYFLOATTYPE>(toAdd[i].x, toAdd[i].y, toAdd[i].z), abs(toAdd[i].radius), abs(toAdd[i].diameter), curveSign, previousN, toAdd[i].refractiveIndex);
		if (!output) return { PI_UNKNOWN_ERROR,"Error adding surface" };
		previousN = toAdd[i].refractiveIndex;
	}

	//test apo function
	(newConfig->surfaces)[0]->apodizationType = APD_BARTLETT;

	//for the final image surface
	//todo: data integrity check here
	toAdd[numofsurfaces - 1].diameter = (abs(toAdd[numofsurfaces - 1].diameter) < 2.0*abs(toAdd[numofsurfaces - 1].radius)) ? abs(toAdd[numofsurfaces - 1].diameter) : 2.0*abs(toAdd[numofsurfaces - 1].radius);
	bool output = constructSurface((newConfig->surfaces)[numofsurfaces-1], MF_IMAGE, vec3<MYFLOATTYPE>(toAdd[count-1].x, toAdd[count - 1].y, toAdd[count - 1].z), -abs(toAdd[count - 1].radius), toAdd[count - 1].diameter, MF_CONCAVE, 1.0, FP_INFINITE);

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
		while (KernelLauncher(0, nullptr) != -2);
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
		while (KernelLauncher2(0, nullptr) != -2);

		//deal with the output
		//data copy out
		thisOpticalConfig->p_rawChannel->copyFromSibling();

		//testing: draw all retina
		//thisOpticalConfig->p_rawChannel->setToValue(1.0, *(thisOpticalConfig->p_retinaDescriptor));

		//display data with scaling
		void* mapX = nullptr;
		void* mapY = nullptr;
		SimpleRetinaDescriptor* tempDescriptor = dynamic_cast<SimpleRetinaDescriptor*>(thisOpticalConfig->p_retinaDescriptor);

		MYFLOATTYPE scalingargs[4] = { tempDescriptor->m_thetaR, tempDescriptor->m_R0, tempDescriptor->m_maxTheta, tempDescriptor->m_maxPhi };
		generateProjectionMap(mapX, mapY, thisOpticalConfig->p_rawChannel->m_dimension.y,
			thisOpticalConfig->p_rawChannel->m_dimension.x, IF_PROJECTION_ALONGZ, 4, scalingargs);

		quickDisplayv2(thisOpticalConfig->p_rawChannel->hp_raw, thisOpticalConfig->p_rawChannel->m_dimension.y,
			thisOpticalConfig->p_rawChannel->m_dimension.x, mapX, mapY, 700);

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
		outputImages[uniqueID].createOutputImage(OIC_XYZ);
		outputImages[uniqueID].displayRGB();

		//clean up
		delete[] wavelengthStorageList;
		delete[] statusStorageList;

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
		mainStorageManager.pleaseDelete(wavelengthStorageList[i]);
	}

	return { PI_OK, "Storage cleared!\n" };
}