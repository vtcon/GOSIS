#pragma once

#include "mycommon.cuh"
//#include "vec3.cuh"
#include "class_hierarchy.cuh"
#include <list>
#include <mutex>
#include "../ConsoleApplication/src/OutputImage.h"

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
	bool jobCheckOut(OpticalConfig*& job, int numofsurfaces, int _wavelength); //.. and yes, it is a reference to a pointer
	
	//return the info of the optical config at given wavelength
	bool infoCheckOut(OpticalConfig*& requiredinfo, int wavelength);
	
	//create a new ray bundle column
	bool jobCheckOut(RayBundleColumn*& job, int numofsurfaces, int _wavelength);
	
	//change the status of that ray bundle column to the next status
	bool jobCheckIn(RayBundleColumn*& job, StorageHolder<RayBundleColumn*>::Status nextstatus); //essentially this function mark a column as "initialized"
	
	//take the first ray bundle column at given status and given wavelength
	bool takeOne(RayBundleColumn*& requiredinfo, StorageHolder<RayBundleColumn*>::Status requiredstatus, int requiredwavelength); // take one initialized column
	
	//tell the storage manager to delete a column
	void pleaseDelete(RayBundleColumn* todelete);
	
private:
	std::mutex opticalConfigLedgerLock;
	std::list<StorageHolder<OpticalConfig*>> opticalConfigLedger;

	std::mutex rayBundleColumnLedgerLock;
	std::list<StorageHolder<RayBundleColumn*>> rayBundleColumnLedger;

	std::mutex wavelengthLedgerLock;
	std::list<StorageHolder<int>> wavelengthLedger;

	//TODO: points ledger

	//TODO: image ledger
	std::mutex imageLedgerLock;
	std::list<StorageHolder<OutputImage*>> outputImageLedger;

	//GPUjob objects are created and destroyed on-the-fly, no need for managing them
	//std::mutex quadricTracerJobLedgerLock;
	//std::list<StorageHolder<QuadricTracerJob*>> quadricTracerJobLedger;
};

