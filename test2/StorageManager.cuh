#pragma once

#include "mycommon.cuh"
#include "vec3.cuh"
#include "class_hierarchy.cuh"
#include <list>
#include <mutex>
#include <algorithm>

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

	bool jobCheckOut(OpticalConfig*& job, int numofsurfaces, int _wavelength) //.. and yes, it is a reference to a pointer
	{
		OpticalConfig* newconfig(new OpticalConfig(numofsurfaces, _wavelength));
		job = newconfig; //just that
		//save newconfig to the ledger
		{
			std::lock_guard<std::mutex> lock(opticalConfigLedgerLock);
			//opticalConfigLedger.emplace_back(newconfig); 
			//above: implicit conversion, below: explicit constructor for clarity (...or not)
			opticalConfigLedger.emplace_back(StorageHolder<OpticalConfig*>(newconfig));
		}

		//save new wavelength to the ledger
		{
			std::lock_guard<std::mutex> lock(wavelengthLedgerLock);
			wavelengthLedger.emplace_back(StorageHolder<int>(_wavelength));
		}

		return true;
	}

	bool infoCheckOut(OpticalConfig*& requiredinfo, int wavelength)
	{
		if (opticalConfigLedger.empty()) return false;

		//requiredinfo = opticalConfigLedger[0].content;

		auto pred = [&wavelength](const StorageHolder<OpticalConfig*>& thisholder)
		{
			bool cond = (thisholder.content)->wavelength == wavelength;
			return cond;
		};

		std::list<StorageHolder<OpticalConfig*>>::const_iterator token;
		{
			std::lock_guard<std::mutex> lock(opticalConfigLedgerLock);
			token = std::find_if(opticalConfigLedger.begin(), opticalConfigLedger.end(), pred);
		}

		if (token == opticalConfigLedger.end()) return false;

		requiredinfo = token->content;

		return true;
	}

	bool jobCheckOut(RayBundleColumn*& job, int numofsurfaces, int _wavelength)
	{
		RayBundleColumn* newcolumn(new RayBundleColumn(numofsurfaces,_wavelength));
		job = newcolumn;
		{
			std::lock_guard<std::mutex> lock(rayBundleColumnLedgerLock);
			rayBundleColumnLedger.emplace_back(newcolumn);
		}
		return true;
	}

	bool jobCheckIn(RayBundleColumn*& job, StorageHolder<RayBundleColumn*>::Status nextstatus) //essentially this function mark a column as "initialized"
	{
		//define a lambda
		auto pred = [&job,nextstatus](const StorageHolder<RayBundleColumn*>& thisholder)
		{
			bool cond1 = thisholder.content == job;
			bool cond2 = thisholder.status == StorageHolder<RayBundleColumn*>::prevStatus(nextstatus);
			return cond1 && cond2;
		};

		std::list<StorageHolder<RayBundleColumn*>>::iterator token;
		{
			std::lock_guard<std::mutex> lock(rayBundleColumnLedgerLock);
			token = std::find_if(rayBundleColumnLedger.begin(), rayBundleColumnLedger.end(), pred);
		}

		if (token == rayBundleColumnLedger.end()) return false; //if none is found

		token->status = nextstatus;

		return true;
	}

	bool takeOne(RayBundleColumn*& requiredinfo, StorageHolder<RayBundleColumn*>::Status requiredstatus, int requiredwavelength) // take one initialized column
	{
		if (rayBundleColumnLedger.empty()) return false; //if there's nothing to be taken

		//define a lambda that returns true if the status is initialized
		auto pred = [requiredstatus, requiredwavelength](const StorageHolder<RayBundleColumn*>& thisholder)
		{
			bool cond1 = thisholder.status == requiredstatus;
			bool cond2 = (thisholder.content)->wavelength == requiredwavelength;
			return cond1 && cond2;
		};

		std::list<StorageHolder<RayBundleColumn*>>::iterator token;//...ugh
		{
			std::lock_guard<std::mutex> lock(rayBundleColumnLedgerLock);
			token = std::find_if(rayBundleColumnLedger.begin(), rayBundleColumnLedger.end(), pred);
		}

		if (token == rayBundleColumnLedger.end()) return false; //if none is found

		//return the found object, set its status to inUse
		token->status = StorageHolder<RayBundleColumn*>::nextStatus(requiredstatus);
		requiredinfo = token->content;

		return true;
	}

	void pleaseDelete(RayBundleColumn* todelete)
	{
		//define a lambda
		auto pred = [&todelete](const StorageHolder<RayBundleColumn*>& thisholder)
		{
			bool cond1 = thisholder.content == todelete;
			//bool cond2 = thisholder.status == StorageHolder<RayBundleColumn*>::Status::uninitialized;
			return cond1;
		};

		std::list<StorageHolder<RayBundleColumn*>>::iterator token;
		{
			std::lock_guard<std::mutex> lock(rayBundleColumnLedgerLock);
			token = std::find_if(rayBundleColumnLedger.begin(), rayBundleColumnLedger.end(), pred);
		}

		if (token == rayBundleColumnLedger.end()) return; //if none is found

		//performs the deletion
		delete todelete;
		{
			std::lock_guard<std::mutex> lock(rayBundleColumnLedgerLock);
			rayBundleColumnLedger.erase(token);
		}

		return;
	}

private:
	std::mutex opticalConfigLedgerLock;
	std::list<StorageHolder<OpticalConfig*>> opticalConfigLedger;

	std::mutex rayBundleColumnLedgerLock;
	std::list<StorageHolder<RayBundleColumn*>> rayBundleColumnLedger;

	std::mutex wavelengthLedgerLock;
	std::list<StorageHolder<int>> wavelengthLedger;

	//GPUjob objects are created and destroyed on-the-fly, no need for managing them
	//std::mutex quadricTracerJobLedgerLock;
	//std::list<StorageHolder<QuadricTracerJob*>> quadricTracerJobLedger;
};


