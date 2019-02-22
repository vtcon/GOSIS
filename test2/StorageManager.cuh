#pragma once

#include "mycommon.cuh"
//#include "vec3.cuh"
#include "class_hierarchy.cuh"
#include <list>
#include <mutex>
#include "../ConsoleApplication/src/OutputImage.h"
#include "ProgramInterface.h"

template<typename T>
class StorageHolder //T should only be pointer type (because of the "delete" call in destructor)
{
public:
	T content;
	static enum Status:int { uninitialized, initialized, inUse1, completed1, inUse2, completed2, markedForDelete };
	Status status;

	StorageHolder(T newcontent, Status newstatus = uninitialized)
		:content(newcontent), status(newstatus)
	{}

	static inline Status prevStatus(const Status& currentstatus)
	{
		return currentstatus > 0 ? (Status)(currentstatus - 1) : (Status)currentstatus;
	}

	static inline Status nextStatus(const Status& currentstatus)
	{
		return (currentstatus != markedForDelete) ? (Status)(currentstatus + 1) : markedForDelete;
	}
};

//this class should be wayyyyyy below
class StorageManager
{
public:
	//create a new optical config and save a wavelength to the ledgers
	bool jobCheckOut(OpticalConfig*& job, int numofsurfaces, float _wavelength); //.. and yes, it is a reference to a pointer
	
	//return the info of the optical config at given wavelength
	bool infoCheckOut(OpticalConfig*& requiredinfo, float wavelength);

	//delete a optical config
	void pleaseDelete(OpticalConfig*& todelete);
	
	//create a new ray bundle column
	bool jobCheckOut(RayBundleColumn*& job, int numofsurfaces, float _wavelength);
	
	//change the status of that ray bundle column to the next status
	bool jobCheckIn(RayBundleColumn*& job, StorageHolder<RayBundleColumn*>::Status nextstatus); //essentially this function mark a column as "initialized"
	
	//take the first ray bundle column at given status and given wavelength
	bool takeOne(RayBundleColumn*& requiredinfo, StorageHolder<RayBundleColumn*>::Status requiredstatus, float requiredwavelength); // take one initialized column
	
	//tell the storage manager to delete a column
	void pleaseDelete(RayBundleColumn*& todelete);

	//add a point to the ledger, if it has a new wavelength, add the wavelength to ledger
	bool add(PI_LuminousPoint toAdd);

	//take a point at a given status and given wavelength
	bool takeOne(PI_LuminousPoint*& requiredinfo, StorageHolder<PI_LuminousPoint>::Status requiredstatus, float requiredwavelength);

	//mark the status of a point
	bool jobCheckIn(PI_LuminousPoint*& job, StorageHolder<PI_LuminousPoint>::Status nextstatus);

	//delete a point
	void pleaseDelete(PI_LuminousPoint todelete);

	//delete all points of a specific wavelength
	void pleaseDeleteAllPoints(float wavelength);

	//delete all points 
	void pleaseDeleteAllPoints();

	//check to see all the wavelengths available
	bool infoCheckOut(float*& wavelengths, int& count);

	//take one wavelength that is available for tracing/rendering
	bool takeOne(float*& wavelength, StorageHolder<float>::Status requiredstatus);

	//mark a wavelength to desired status
	bool jobCheckIn(float*& job, StorageHolder<float>::Status nextstatus);
	
private:
	std::mutex opticalConfigLedgerLock;
	std::list<StorageHolder<OpticalConfig*>> opticalConfigLedger;

	std::mutex rayBundleColumnLedgerLock;
	std::list<StorageHolder<RayBundleColumn*>> rayBundleColumnLedger;

	//TODO: points ledger
	std::mutex pointLedgerLock;
	std::list<StorageHolder<PI_LuminousPoint>> pointLedger;

	//TODO: image ledger
	std::mutex imageLedgerLock;
	std::list<StorageHolder<OutputImage*>> outputImageLedger;

	//TODO: wavelength ledger
	std::mutex wavelengthLedgerLock;
	std::list<StorageHolder<float>> wavelengthLedger;

	//GPUjob objects are created and destroyed on-the-fly, no need for managing them
	//std::mutex quadricTracerJobLedgerLock;
	//std::list<StorageHolder<QuadricTracerJob*>> quadricTracerJobLedger;
};

