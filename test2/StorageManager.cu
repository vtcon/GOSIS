#include "StorageManager.cuh"
#include <algorithm>

bool StorageManager::jobCheckOut(OpticalConfig *& job, int numofsurfaces, float _wavelength)
{
	//first check if the optical config at that wavelength already exists, if yes, delete it
	OpticalConfig* oldconfig = nullptr;
	if (infoCheckOut(oldconfig, _wavelength))
		pleaseDelete(oldconfig);

	//create a new config
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
		wavelengthLedger.emplace_back(StorageHolder<float>(_wavelength));
	}

	return true;
}

bool StorageManager::infoCheckOut(OpticalConfig *& requiredinfo, float wavelength)
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

void StorageManager::pleaseDelete(OpticalConfig *& todelete)
{
	//define a lambda
	auto pred = [&todelete](const StorageHolder<OpticalConfig*>& thisholder)
	{
		return true;
	};

	std::list<StorageHolder<OpticalConfig*>>::iterator token;
	{
		std::lock_guard<std::mutex> lock(opticalConfigLedgerLock);
		token = std::find_if(opticalConfigLedger.begin(), opticalConfigLedger.end(), pred);
	}

	if (token == opticalConfigLedger.end()) return; //if none is found

	//performs the deletion
	delete todelete;
	todelete = nullptr;
	{
		std::lock_guard<std::mutex> lock(opticalConfigLedgerLock);
		opticalConfigLedger.erase(token);
	}

	return;
}

bool StorageManager::jobCheckOut(RayBundleColumn *& job, int numofsurfaces, float _wavelength)
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

bool StorageManager::takeOne(RayBundleColumn *& requiredinfo, StorageHolder<RayBundleColumn*>::Status requiredstatus, float requiredwavelength)
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

void StorageManager::pleaseDelete(RayBundleColumn*& todelete)
{
	//define a lambda
	auto pred = [&todelete](const StorageHolder<RayBundleColumn*>& thisholder)
	{
		//bool cond1 = thisholder.content == todelete;
		//bool cond2 = thisholder.status == StorageHolder<RayBundleColumn*>::Status::uninitialized;
		return true;
	};

	std::list<StorageHolder<RayBundleColumn*>>::iterator token;
	{
		std::lock_guard<std::mutex> lock(rayBundleColumnLedgerLock);
		token = std::find_if(rayBundleColumnLedger.begin(), rayBundleColumnLedger.end(), pred);
	}

	if (token == rayBundleColumnLedger.end()) return; //if none is found

	//performs the deletion
	delete todelete;
	todelete = nullptr;
	{
		std::lock_guard<std::mutex> lock(rayBundleColumnLedgerLock);
		rayBundleColumnLedger.erase(token);
	}

	return;
}

bool StorageManager::add(PI_LuminousPoint toAdd)
{
	//check if point already exists, if yes, quit, else add it
	auto pred = [toAdd](const StorageHolder<PI_LuminousPoint>& thisholder)
	{
		bool cond = thisholder.content == toAdd;
		return cond;
	};

	std::list<StorageHolder<PI_LuminousPoint>>::iterator token;//...ugh
	{
		std::lock_guard<std::mutex> lock(pointLedgerLock);
		token = std::find_if(pointLedger.begin(), pointLedger.end(), pred);
	}

	if (token != pointLedger.end()) return false; //if point already exists

	{
		std::lock_guard<std::mutex> lock(pointLedgerLock);
		pointLedger.emplace_back(StorageHolder<PI_LuminousPoint>(toAdd, StorageHolder<PI_LuminousPoint>::Status::uninitialized));
	}
	
	//check if wavelength of the point already exists, if not, add it
	auto pred2 = [toAdd](const StorageHolder<float>& thisholder)
	{
		bool cond = thisholder.content == toAdd.wavelength;
		return cond;
	};

	std::list<StorageHolder<float>>::iterator token2;//...ugh
	{
		std::lock_guard<std::mutex> lock(wavelengthLedgerLock);
		token2 = std::find_if(wavelengthLedger.begin(), wavelengthLedger.end(), pred2);
	}

	if (token2 == wavelengthLedger.end())
	{
		std::lock_guard<std::mutex> lock(wavelengthLedgerLock);
		wavelengthLedger.push_back(StorageHolder<float>(toAdd.wavelength));
	}

	return true;
}

bool StorageManager::takeOne(PI_LuminousPoint *& requiredinfo, StorageHolder<PI_LuminousPoint>::Status requiredstatus, float requiredwavelength)
{
	if (pointLedger.empty()) return false; //if there's nothing to be taken

	//define a lambda that returns true if the status is initialized
	auto pred = [requiredstatus, requiredwavelength](const StorageHolder<PI_LuminousPoint>& thisholder)
	{
		bool cond1 = thisholder.status == requiredstatus;
		bool cond2 = thisholder.content.wavelength == requiredwavelength;
		return cond1 && cond2;
	};

	std::list<StorageHolder<PI_LuminousPoint>>::iterator token;//...ugh
	{
		std::lock_guard<std::mutex> lock(pointLedgerLock);
		token = std::find_if(pointLedger.begin(), pointLedger.end(), pred);
	}

	if (token == pointLedger.end()) return false; //if none is found

	//return the found object, set its status to inUse
	token->status = StorageHolder<PI_LuminousPoint>::nextStatus(requiredstatus);
	requiredinfo = &(token->content);

	return true;
}

bool StorageManager::jobCheckIn(PI_LuminousPoint *& job, StorageHolder<PI_LuminousPoint>::Status nextstatus)
{
	//define a lambda
	auto pred = [&job, nextstatus](const StorageHolder<PI_LuminousPoint>& thisholder)
	{
		bool cond1 = thisholder.content == *job;
		//bool cond2 = thisholder.status == StorageHolder<PI_LuminousPoint>::prevStatus(nextstatus);
		return cond1;
	};

	std::list<StorageHolder<PI_LuminousPoint>>::iterator token;
	{
		std::lock_guard<std::mutex> lock(pointLedgerLock);
		token = std::find_if(pointLedger.begin(), pointLedger.end(), pred);
	}

	if (token == pointLedger.end()) return false; //if none is found

	token->status = nextstatus;

	return true;
}

bool StorageManager::takeOne(float *& wavelength, StorageHolder<float>::Status requiredstatus)
{
	if (wavelengthLedger.empty()) return false; //if there's nothing to be taken

	//define a lambda that returns true if the status is initialized
	auto pred = [requiredstatus](const StorageHolder<float>& thisholder)
	{
		bool cond1 = thisholder.status == requiredstatus;
		return cond1;
	};

	std::list<StorageHolder<float>>::iterator token;//...ugh
	{
		std::lock_guard<std::mutex> lock(wavelengthLedgerLock);
		token = std::find_if(wavelengthLedger.begin(), wavelengthLedger.end(), pred);
	}

	if (token == wavelengthLedger.end()) return false; //if none is found

	//return the found object, set its status to inUse
	token->status = StorageHolder<float>::nextStatus(requiredstatus);
	wavelength = &(token->content);

	std::cout << wavelengthLedger.size() << "\n";

	return true;
}

bool StorageManager::jobCheckIn(float *& job, StorageHolder<float>::Status nextstatus)
{
	//define a lambda
	auto pred = [&job, nextstatus](const StorageHolder<float>& thisholder)
	{
		bool cond1 = thisholder.content == *job;
		//bool cond2 = thisholder.status == StorageHolder<PI_LuminousPoint>::prevStatus(nextstatus);
		return cond1;
	};

	std::list<StorageHolder<float>>::iterator token;
	{
		std::lock_guard<std::mutex> lock(wavelengthLedgerLock);
		token = std::find_if(wavelengthLedger.begin(), wavelengthLedger.end(), pred);
	}

	if (token == wavelengthLedger.end()) return false; //if none is found

	token->status = nextstatus;

	return true;
}