#include "GLDrawFacilities.h"
#include <iostream>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <utility>

#define PI_F (float)3.14159265358979323846264338327950288419716939937510582097494 

//externals
extern int GLDrawer(const std::vector<std::pair<glm::vec3, glm::vec3>>& poArray,
	const std::vector<std::vector<float>>& vaoArray,
	const std::vector<std::vector<unsigned int>>& iboArray,
	const std::vector<TextureIndicator>& tiArray);

void drawSurfaces(const std::vector<SurfaceDrawInfo>& surfaceInfos, bool suppressTexture, bool suppressImage)
{
	std::vector<std::pair<glm::vec3, glm::vec3>> poArray(surfaceInfos.size()); //pair of position and orientation
	std::vector<std::vector<float>> vaoArray(surfaceInfos.size());
	std::vector<std::vector<unsigned int>> iboArray(surfaceInfos.size());
	std::vector<TextureIndicator> tiArray(surfaceInfos.size());

	{
		int si = 0; //surface index
		for (auto currentSurfaceInfo : surfaceInfos)
		{
			currentSurfaceInfo.arms = (currentSurfaceInfo.arms < 3) ? 3 : currentSurfaceInfo.arms;
			currentSurfaceInfo.rings = (currentSurfaceInfo.rings < 1) ? 1 : currentSurfaceInfo.rings;

			//write position and orientation array
			poArray[si] = std::make_pair(
				glm::vec3(currentSurfaceInfo.posX, currentSurfaceInfo.posY, currentSurfaceInfo.posZ), 
				glm::vec3(currentSurfaceInfo.rotX, currentSurfaceInfo.rotY, currentSurfaceInfo.rotZ));

			//write texture array
			if (currentSurfaceInfo.texRows > 0 
				&& currentSurfaceInfo.texCols > 0 
				&& currentSurfaceInfo.p_tex != nullptr 
				&& suppressTexture == false
				&& (si != surfaceInfos.size() - 1))
			{
				tiArray[si].hasTexture = true;
				tiArray[si].rows = currentSurfaceInfo.texRows;
				tiArray[si].cols = currentSurfaceInfo.texCols;
				tiArray[si].channels = currentSurfaceInfo.texChannels;
				tiArray[si].p_tex = currentSurfaceInfo.p_tex;
			}

			if (currentSurfaceInfo.texRows > 0 &&
				currentSurfaceInfo.texCols > 0 &&
				currentSurfaceInfo.p_tex != nullptr &&
				(si == surfaceInfos.size() - 1) && 
				!suppressImage)
			{
				tiArray[si].hasTexture = true;
				tiArray[si].rows = currentSurfaceInfo.texRows;
				tiArray[si].cols = currentSurfaceInfo.texCols;
				tiArray[si].channels = currentSurfaceInfo.texChannels;
				tiArray[si].p_tex = currentSurfaceInfo.p_tex;
			}

			float armStep = 2.0f * PI_F / currentSurfaceInfo.arms;
			float ringStep = ((float)currentSurfaceInfo.diameter / 2.0f) / currentSurfaceInfo.rings;

			//vao and ibo for the central point
			vaoArray[si].push_back(0.0f);//x
			vaoArray[si].push_back(0.0f);//y
			vaoArray[si].push_back(0.0f);//z
			if (tiArray[si].hasTexture)
			{
				vaoArray[si].push_back(0.5f);
				vaoArray[si].push_back(0.5f);
			}
			for (int i = 1; i < currentSurfaceInfo.arms; i++)
			{
				iboArray[si].push_back(0);
				iboArray[si].push_back(i);
				iboArray[si].push_back(i+1);
			}
			iboArray[si].push_back(0);
			iboArray[si].push_back(currentSurfaceInfo.arms);
			iboArray[si].push_back(1);

			//vao for surrounding points
			for (int i = 1; i <= currentSurfaceInfo.rings; i++)
			{
				float rho = ringStep * i;
				for (int j = 0; j < currentSurfaceInfo.arms; j++)
				{
					float phi = armStep * j;

					vaoArray[si].push_back(rho*cos(phi));
					vaoArray[si].push_back(rho*sin(phi));
					float delta = currentSurfaceInfo.radius*currentSurfaceInfo.radius - (currentSurfaceInfo.asphericity + 1.0f)*rho*rho;
					if (delta < 0)
					{
						vaoArray[si].push_back(0);
					}
					else
					{
						if (!currentSurfaceInfo.radiusSign)
							vaoArray[si].push_back((-currentSurfaceInfo.radius + sqrt(delta)) / (currentSurfaceInfo.asphericity + 1.0f));
						else
							vaoArray[si].push_back((currentSurfaceInfo.radius - sqrt(delta)) / (currentSurfaceInfo.asphericity + 1.0f));
					}
					if (tiArray[si].hasTexture)
					{
						vaoArray[si].push_back(0.5f + rho / currentSurfaceInfo.diameter*cos(phi));
						vaoArray[si].push_back(0.5f + rho / currentSurfaceInfo.diameter*sin(phi));
					}
				}
			}

			//ibo for surrounding points
			for (int i = 0; i < currentSurfaceInfo.rings - 1; i++)
			{
				int base = 1 + i * currentSurfaceInfo.arms;
				for (int j = 0; j < currentSurfaceInfo.arms - 1; j++)
				{
					iboArray[si].push_back(base + j);
					iboArray[si].push_back(base + j + currentSurfaceInfo.arms);
					iboArray[si].push_back(base + j + currentSurfaceInfo.arms + 1);

					iboArray[si].push_back(base + j);
					iboArray[si].push_back(base + j + currentSurfaceInfo.arms + 1);
					iboArray[si].push_back(base + j + 1);
				}
				iboArray[si].push_back(base + currentSurfaceInfo.arms - 1);
				iboArray[si].push_back(base + 2 * currentSurfaceInfo.arms - 1);
				iboArray[si].push_back(base + currentSurfaceInfo.arms);

				iboArray[si].push_back(base + currentSurfaceInfo.arms - 1);
				iboArray[si].push_back(base + currentSurfaceInfo.arms);
				iboArray[si].push_back(base);
			}

#ifdef nothing
			if (si == 0)
			{
				std::cout << "VAO for surface " << si << ":\n";
				int lineBreakCount = (tiArray[si].hasTexture) ? 5 : 3;
				for (int i = 0; i < vaoArray[si].size(); i++)
				{
					std::cout << vaoArray[si][i] << " ";
					if ((i % lineBreakCount) == (lineBreakCount - 1))
						std::cout << "\n";
				}
				std::cout << "\n";

				std::cout << "IBO for surface " << si << ":\n";
				for (int i = 0; i < iboArray[si].size(); i++)
				{
					std::cout << iboArray[si][i] << " ";
					if (i % 3 == 2)
						std::cout << "\n";
				}
				std::cout << "\n";
			}
#endif

			si++;
		}
	}

	GLDrawer(poArray, vaoArray, iboArray, tiArray);
}

bool runTestGLDrawFacilities = true;
void testGLDrawFacilities()
{
	std::vector<SurfaceDrawInfo> surfaceInfos;
	{
		SurfaceDrawInfo info;
		info.arms = 15;
		info.rings = 10;
		info.posZ = 50.0;
		surfaceInfos.push_back(info);
	}
	{
		SurfaceDrawInfo info;
		info.arms = 35;
		info.rings = 20;
		info.texRows = 1;
		info.texCols = 1;
		info.radiusSign = false;
		info.p_tex = new unsigned char();
		surfaceInfos.push_back(info);
	}
	drawSurfaces(surfaceInfos);
}
