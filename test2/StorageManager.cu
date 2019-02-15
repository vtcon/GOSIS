#include "StorageManager.cuh"
#include <algorithm>

bool StorageManager::jobCheckOut(OpticalConfig *& job, int numofsurfaces, int _wavelength)
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

bool StorageManager::infoCheckOut(OpticalConfig *& requiredinfo, int wavelength)
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

bool StorageManager::jobCheckOut(RayBundleColumn *& job, int numofsurfaces, int _wavelength)
{
	RayBundleColumn* newcolumn(new RayBundleColumn(numofsurfaces, _wavelength));
	job = newcolumn;
	{
		std::lock_guard<std::mutex> lock(rayBundleColumnLedgerLock);
		rayBundleColumnLedger.emplace_back(newcolumn);
	}
	return true;
}

bool StorageManager::jobCheckIn(RayBundleColumn *& job, StorageHolder<RayBundleColumn*>::Status nextstatus)
{
	//define a lambda
	auto pred = [&job, nextstatus](const StorageHolder<RayBundleColumn*>& thisholder)
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

bool StorageManager::takeOne(RayBundleColumn *& requiredinfo, StorageHolder<RayBundleColumn*>::Status requiredstatus, int requiredwavelength)
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

void StorageManager::pleaseDelete(RayBundleColumn * todelete)
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
