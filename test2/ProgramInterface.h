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
#define PI_SYSTEM_REQUIREMENT_ERROR 105

//definition of apodization options
#define PI_APD_UNIFORM 0
#define PI_APD_BARTLETT 1
#define PI_APD_CUSTOM 2

//definition of rgb and raw options
#define PI_SRGB 0
#define PI_ADOBERGB 1
#define PI_LMS 3
#define PI_XYZ 4

//definition of projection options
#define PI_PROJECTION_NONE 0
#define PI_PROJECTION_ALONGZ 1
#define PI_PROJECTION_PLATE_CARREE 2


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
		float asphericity = 0.0; //0.0 is default for spherical surfaces
		unsigned short int apodization = PI_APD_UNIFORM;
		const char* customApoPath = "";
	};

	struct EXPORT PI_Message
	{
		unsigned int code;
		const char* detail;
	};

	struct EXPORT PI_Preferences
	{
		int ThreadsPerKernelLaunch = 16;
		int linearRayDensity = 30;//20 is ok
		unsigned int rgbStandard = PI_ADOBERGB;
		int traceJobSize = 10;
		int renderJobSize = 10;
		unsigned short int rawFormat = PI_XYZ; //OIC_LMS
		unsigned int projectionMethod = PI_PROJECTION_PLATE_CARREE;
		int displayWindowSize = 800;
		float primaryWavelengthR = 620;
		float primaryWavelengthG = 530;
		float primaryWavelengthB = 465;
		int maxParallelThread = 10;
	};

	//Program API functions
	PI_Message EXPORT initialization();
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
	PI_Message EXPORT traceAndRender();

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
	PI_Message EXPORT getVRAMUsageInfo(unsigned long & total, unsigned long & free);

	PI_Message EXPORT importImage(const char* path, float posX, float posY, float posZ, float sizeHorz, float sizeVert, float rotX, float rotY, float rotZ, float brightness);	
	PI_Message EXPORT getImagePrimaryWavelengths(float& wavelengthR, float& wavelengthG, float& wavelengthB);

	PI_Message EXPORT getPreferences(PI_Preferences& prefs);
	PI_Message EXPORT setPreferences(PI_Preferences& prefs);
	PI_Message EXPORT defaultPreference();

	PI_Message EXPORT drawOpticalConfig(float wavelength, bool suppressRefractiveSurfaceTexture = false, bool suppressImageTexture = false);
	PI_Message EXPORT drawImage(int uniqueID);
}