#pragma once
#include "class_hierarchy.cuh"

class OpticalConfig
{
public:
	int numofsurfaces = 2;
	mysurface<MYFLOATTYPE>** surfaces = nullptr;

	OpticalConfig(int numberOfSurfaces):numofsurfaces(numberOfSurfaces)
	{
		surfaces = new mysurface<MYFLOATTYPE>*[numofsurfaces];
	}

	~OpticalConfig()
	{
		freesiblings();
		for (int i = 0; i < numofsurfaces; i++)
		{
			delete surfaces[i];
		}
		delete[] surfaces;
	}
	
	void copytosibling()
	{
		for (int i = 0; i < numofsurfaces; i++)
			surfaces[i]->copytosibling();
	}

	void freesiblings()
	{
		for (int i = 0; i < numofsurfaces; i++)
			surfaces[i]->freesibling();
	}
private:

	//disable both of them, OpticalConfig are meant to only be created and destroyed, not passing around
	OpticalConfig(const OpticalConfig& origin);
	OpticalConfig operator=(const OpticalConfig& origin);

};