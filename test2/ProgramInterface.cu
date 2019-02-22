#include "ProgramInterface.h"
#include "StorageManager.cuh"
#include "ManagerFunctions.cuh"
#include "Auxiliaries.cuh"

#include <list>

//internal global variables
std::list<StorageManager> repoList;
StorageManager mainStorageManager;
float activeWavelength = 555.0;

//external global variables
int PI_ThreadsPerKernelLaunch = 16;


//function definitions
bool PI_LuminousPoint::operator==(const PI_LuminousPoint & rhs) const
{
	if (x == rhs.x &&y == rhs.y &&z == rhs.z &&wavelength == rhs.wavelength &&intensity == rhs.intensity)
		return true;
	else
		return false;
}

PI_Message addPoint(PI_LuminousPoint & toAdd)
{
	mainStorageManager.add(toAdd);
	return PI_Message();
}

PI_Message addOpticalConfigAt(float wavelength, int count, PI_Surface *& toAdd, float angularResolution, float angularExtend)
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
		bool output = constructSurface((newConfig->surfaces)[i], MF_REFRACTIVE, vec3<MYFLOATTYPE>(toAdd[i].x, toAdd[i].y, toAdd[i].z), abs(toAdd[i].radius), abs(toAdd[i].diameter), curveSign, previousN, toAdd[i].refractiveIndex);
		if (!output) return { PI_UNKNOWN_ERROR,"Error adding surface" };
		previousN = toAdd[i].refractiveIndex;
	}

	//for the final image surface
	//todo: data integrity check here
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
	p_retinaDescriptor = new SimpleRetinaDescriptor(angularResolution, R, angularExtend, angularExtend); //doesn't need to explicitly delete this
	newConfig->createImageChannel(p_retinaDescriptor);

	return { PI_OK, "Optical Config added" };
}

PI_Message checkData()
{
	float* currentwavelength = nullptr;
	bool output = mainStorageManager.takeOne(currentwavelength, StorageHolder<float>::Status::uninitialized);

	//if output == false, what to do???

	//initialize all points
	activeWavelength = *currentwavelength;
	while (ColumnCreator3() != -2);

	mainStorageManager.jobCheckIn(currentwavelength, StorageHolder<float>::Status::initialized);

	return PI_Message();
}

PI_Message trace()
{
	float* currentwavelength = nullptr;
	bool output = mainStorageManager.takeOne(currentwavelength, StorageHolder<float>::Status::initialized);

	//if output == false, what to do???

	//initialize all points
	activeWavelength = *currentwavelength;
	while (KernelLauncher(0, nullptr) != -2);

	mainStorageManager.jobCheckIn(currentwavelength, StorageHolder<float>::Status::completed1);

	return PI_Message();
}

PI_Message render()
{
	float* currentwavelength = nullptr;
	bool output = mainStorageManager.takeOne(currentwavelength, StorageHolder<float>::Status::completed1);

	//if output == false, what to do???

	//initialize all points
	activeWavelength = *currentwavelength;
	while (KernelLauncher2(0, nullptr) != -2);

	mainStorageManager.jobCheckIn(currentwavelength, StorageHolder<float>::Status::completed2);

	return PI_Message();
}