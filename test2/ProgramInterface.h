#pragma once

#ifdef _WINDLL
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT __declspec(dllimport)
#endif

//defines
//defines a few message codes here
#define PI_OK 200
#define PI_UNKNOWN_ERROR 404
#define PI_INFO_RESPONSE 201
#define PI_INPUT_ERROR 300

//definition of apodization options
#define PI_APD_UNIFORM 0
#define PI_APD_BARTLETT 1

namespace tracer
{
	

	//Program classes for export
	class EXPORT PI_LuminousPoint
	{
	public:
		float x = 0.0;
		float y = 0.0;
		float z = 0.0;
		float wavelength = 555.0;
		float intensity = 1.0;

		bool operator==(const PI_LuminousPoint& rhs) const;
	};

	class EXPORT PI_Surface
	{
	public:
		float x = 0.0;
		float y = 0.0;
		float z = 0.0;
		float diameter = 0.0;
		float radius = 1.0;
		float refractiveIndex = 1.0;
		float asphericity = 1.0; //1.0 is default for spherical surfaces
		unsigned short int apodization = PI_APD_UNIFORM;
	};

	struct EXPORT PI_Message
	{
		unsigned int code;
		const char* detail;
	};

	//Program API functions
	PI_Message EXPORT test();

	PI_Message EXPORT newSession();
	PI_Message EXPORT clearSession();

	PI_Message EXPORT addPoint(PI_LuminousPoint& toAdd);
	//PI_Message EXPORT removePoint(PI_LuminousPoint& toRemove);
	//PI_Message EXPORT clearAllPoints();

	//PI_Message EXPORT getAllWavelength(float*& outputArray);

	//PI_Message EXPORT getOpticalConfigAt(float wavelength, int& count, PI_Surface*& output, float& angularResolution, float& angularExtend);
	PI_Message EXPORT addOpticalConfigAt(float wavelength, int count, PI_Surface*& toAdd, float angularResolution, float angularExtend);
	//PI_Message EXPORT modifyOpticalConfigAt(float wavelength, int count, PI_Surface*& toModify, float angularResolution, float angularExtend);


	PI_Message EXPORT checkData();
	PI_Message EXPORT trace();
	PI_Message EXPORT render();

	PI_Message EXPORT showRaw(float* wavelengths, int count);
	PI_Message EXPORT showRGB(int uniqueID);
	PI_Message EXPORT saveRaw(const char* path, int uniqueID);
	PI_Message EXPORT saveRGB(const char* path, int uniqueID);

	PI_Message EXPORT setLinearRayDensity(unsigned int newDensity);
	PI_Message EXPORT getLinearRayDensity();

	PI_Message EXPORT createOutputImage(int count, float * wavelengthList, int& uniqueID);
	PI_Message EXPORT deleteOutputImage(int uniqueID);

	PI_Message EXPORT clearStorage();

	PI_Message EXPORT getProgress(float& traceProgress, float& renderProgress);

	PI_Message EXPORT importImage(const char* path, float posX, float posY, float posZ, float sizeHorz, float sizeVert, float rotX, float rotY, float rotZ, float wavelengthR, float wavelengthG, float wavelengthB, float brightness);
}