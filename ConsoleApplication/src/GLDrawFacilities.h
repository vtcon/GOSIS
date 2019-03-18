#pragma once

#include <vector>

struct SurfaceDrawInfo
{
	float posX = 0.0;
	float posY = 0.0;
	float posZ = 0.0;
	float rotX = 0.0;
	float rotY = 0.0;
	float rotZ = 0.0;
	float asphericity = 0.0;
	float radius = 40.0;
	bool radiusSign = true; //true IS NEGATIVE (convex surface), that is Pz^2 - 2zR + x^2 + y^2 = 0, not +2zR
	bool isFlat = false;
	float diameter = 40.0;
	int rings = 3;
	int arms = 4;
	int texRows = 0;
	int texCols = 0;
	int texChannels = 1;
	unsigned char* p_tex;
};

struct TextureIndicator
{
	bool hasTexture = false;
	int rows = 0;
	int cols = 0;
	int channels = 1;
	unsigned char* p_tex = nullptr;
};

void drawSurfaces(const std::vector<SurfaceDrawInfo>& surfaceInfos, bool suppressTexture = true, bool suppressImage = true);

void testGLDrawFacilities();