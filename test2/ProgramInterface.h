#pragma once

#define MAKEDLL

#ifdef MAKEDLL
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT __declspec(dllimport)
#endif

#include <string>

//defines
//defines a few message codes here
#define PI_OK 200
#define PI_UNKNOWN_ERROR 404

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
	float asphericity = 0.0;
};

struct EXPORT PI_Message
{
	unsigned int code;
	std::string detail;
};

//Program API functions
PI_Message EXPORT test();

PI_Message EXPORT newSession();
PI_Message EXPORT clearSession();

PI_Message EXPORT addPoint(PI_LuminousPoint& toAdd);
PI_Message EXPORT removePoint(PI_LuminousPoint& toRemove);
PI_Message EXPORT clearAllPoints();

PI_Message EXPORT getAllWavelength(float*& outputArray);

PI_Message EXPORT getOpticalConfigAt(float wavelength, int& count, PI_Surface*& output, float& angularResolution, float& angularExtend);
PI_Message EXPORT addOpticalConfigAt(float wavelength, int count, PI_Surface*& toAdd, float angularResolution, float angularExtend);
PI_Message EXPORT modifyOpticalConfigAt(float wavelength, int count, PI_Surface*& toModify, float angularResolution, float angularExtend);


PI_Message EXPORT checkData();
PI_Message EXPORT trace();
PI_Message EXPORT render();