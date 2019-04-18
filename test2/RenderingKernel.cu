#include "mycommon.cuh"
#include "CommonClasses.h"
#include "vec3.cuh"
#include "class_hierarchy.cuh"
#include "StorageManager.cuh"
#include "ManagerFunctions.cuh"

#include "../ConsoleApplication/src/ImageFacilities.h"

// data structure definition


//forward declaration
/*__device__ bool guessPoint (const vec3<MYFLOATTYPE>& vtx, MYFLOATTYPE thetaR, MYFLOATTYPE R0, 
	point2D<int>& pGuess);*/
/*__device__ bool find4points(int nx, int ny, MYFLOATTYPE thetaR, MYFLOATTYPE R0, vec3<MYFLOATTYPE>& p1, 
	vec3<MYFLOATTYPE>& p2, vec3<MYFLOATTYPE>& p3, vec3<MYFLOATTYPE>& p4);*/
__device__ int maptotriangle(
	const vec3<MYFLOATTYPE>& p1, const vec3<MYFLOATTYPE>& p2, const vec3<MYFLOATTYPE>& p3,
	const vec3<MYFLOATTYPE>& px, const vec3<MYFLOATTYPE>& dir, MYFLOATTYPE& alpha, MYFLOATTYPE& beta);

__device__ bool sortCCW(const point2D<MYFLOATTYPE>& c1, const point2D<MYFLOATTYPE>& c2,
	const point2D<MYFLOATTYPE>& c3, const point2D<MYFLOATTYPE>& c4,
	point2D<MYFLOATTYPE>& p1, point2D<MYFLOATTYPE>& p2, point2D<MYFLOATTYPE>& p3, point2D<MYFLOATTYPE>& p4);

__device__ MYFLOATTYPE SutherlandHogdman(const point2D<MYFLOATTYPE>& c1, const point2D<MYFLOATTYPE>& c2,
	const point2D<MYFLOATTYPE>& c3, const point2D<MYFLOATTYPE>& c4);

__device__ MYFLOATTYPE insideTriangle(
	const point2D<int>& p, const vec3<MYFLOATTYPE>& vtx1, const vec3<MYFLOATTYPE>& vtx2,
	const vec3<MYFLOATTYPE>& vtx3, const vec3<MYFLOATTYPE>& pdir, const SimpleRetinaDescriptor& retinaDescriptor);

__device__ MYFLOATTYPE insideTriangle2(
	const point2D<int>& p, const vec3<MYFLOATTYPE>& vtx1, const vec3<MYFLOATTYPE>& vtx2,
	const vec3<MYFLOATTYPE>& vtx3, const vec3<MYFLOATTYPE>& pdir, const SimpleRetinaDescriptor& retinaDescriptor,
	MYFLOATTYPE intensity1, MYFLOATTYPE intensity2, MYFLOATTYPE intensity3);

__device__ MYFLOATTYPE SutherlandHogdman2(const point2D<MYFLOATTYPE>& c1, const point2D<MYFLOATTYPE>& c2,
	const point2D<MYFLOATTYPE>& c3, const point2D<MYFLOATTYPE>& c4,
	MYFLOATTYPE intensity1, MYFLOATTYPE intensity2, MYFLOATTYPE intensity3);


__global__ void nonDiffractiveBasicRenderer(SimpleRetinaDescriptor retinaDescriptorIn, RetinaImageChannel* p_rawChannel);

__global__ void nonDiffractiveBasicRenderer(RendererKernelLaunchParams kernelLaunchParams);
//developing area

//main executing function
/*
void testRenderer()
{

	//create and start timing events
	cudaEvent_t start, stop;
	CUDARUN(cudaEventCreate(&start));
	CUDARUN(cudaEventCreate(&stop));
	CUDARUN(cudaEventRecord(start, 0));

	//data setup
	SimpleRetinaDescriptor retinaDescriptor(10, 2);
	RetinaImageChannel rawChannel(retinaDescriptor);
	rawChannel.createHostImage();
	rawChannel.createSibling();

	
	//launch
	dim3 blocksToLaunch = 1;
	dim3 threadsToLaunch = 1;
	nonDiffractiveBasicRenderer <<<blocksToLaunch, threadsToLaunch >>> (retinaDescriptor, rawChannel.dp_sibling);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__);
		fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();

	//stop timing, print out and delete events
	CUDARUN(cudaEventRecord(stop, 0));
	CUDARUN(cudaEventSynchronize(stop));
	float elapsedtime;
	CUDARUN(cudaEventElapsedTime(&elapsedtime, start, stop));
	LOG2("kernel run time: " << elapsedtime << " ms\n");
	CUDARUN(cudaEventDestroy(start));
	CUDARUN(cudaEventDestroy(stop));

	//data copy out
	rawChannel.copyFromSibling();
	quickDisplay<MYFLOATTYPE>(rawChannel.hp_raw, rawChannel.m_dimension.y, rawChannel.m_dimension.x);
	rawChannel.deleteSibling();
	rawChannel.deleteHostImage();

}
*/

//function bodies (definitions)
#ifdef nothing
__global__ void nonDiffractiveBasicRenderer(SimpleRetinaDescriptor retinaDescriptorIn, RetinaImageChannel* dp_rawChannel)
{
	//developing area
	

	//data setup for testing

	//triangle vertices, in real program should translate world coor to retina local coor
	vec3<MYFLOATTYPE> vtx2(-0.5, -0.5, 0);
	vec3<MYFLOATTYPE> vtx1(1, 0, 0);
	vec3<MYFLOATTYPE> vtx3(-1, 1.3, 0);
	
			   
	//dir of incoming light, in real program should take average of the dirs at triangle vertex
	vec3<MYFLOATTYPE> pdir(0.2, 0.2, 1);

	//MYFLOATTYPE thetaR = retinaDescriptor.m_thetaR;
	SimpleRetinaDescriptor retinaDescriptor = retinaDescriptorIn;
	/*explain above: because SimpleRetinaDescriptor is an inherited class, it must be re-created...
	...inside the kernel so that the v-table is created on the device's side and polymorphism can work*/
	//RetinaImageChannel rawChannel(*dp_rawChannel);

	MYFLOATTYPE R0 = retinaDescriptor.m_R0;

	//for testing only, in real just take the translated version of z
	vtx1.z = sqrt(R0*R0 - vtx1.x*vtx1.x - vtx1.y*vtx1.y);
	vtx2.z = sqrt(R0*R0 - vtx2.x*vtx2.x - vtx2.y*vtx2.y);
	vtx3.z = sqrt(R0*R0 - vtx3.x*vtx3.x - vtx3.y*vtx3.y);
#endif

__global__ void nonDiffractiveBasicRenderer(RendererKernelLaunchParams kernelLaunchParams)
{
	//developing area
	int ID = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (ID >= kernelLaunchParams.otherparams[0]) return;
#ifdef _MYDEBUGMODE
	int debugID = 247;
	if (ID != debugID) return;
#endif
	//ID = debugID;
	//triangle vertices, in real program should translate world coor to retina local coor
	vec3<MYFLOATTYPE> vtx1 = kernelLaunchParams.dp_triangles[ID].vtx1;
	vec3<MYFLOATTYPE> vtx2 = kernelLaunchParams.dp_triangles[ID].vtx2;
	vec3<MYFLOATTYPE> vtx3 = kernelLaunchParams.dp_triangles[ID].vtx3;


	//dir of incoming light, take average of the dirs at triangle vertex
	vec3<MYFLOATTYPE> pdir1 = kernelLaunchParams.dp_triangles[ID].dir1;
	vec3<MYFLOATTYPE> pdir2 = kernelLaunchParams.dp_triangles[ID].dir2;
	vec3<MYFLOATTYPE> pdir3 = kernelLaunchParams.dp_triangles[ID].dir3;
	vec3<MYFLOATTYPE> pdir = (pdir1 + pdir2 + pdir3) / (MYFLOATTYPE)3.0;

	//intensity of the triangle is the average intensity of three rays
	MYFLOATTYPE intensity1 = kernelLaunchParams.dp_triangles[ID].intensity1;
	MYFLOATTYPE intensity2 = kernelLaunchParams.dp_triangles[ID].intensity2;
	MYFLOATTYPE intensity3 = kernelLaunchParams.dp_triangles[ID].intensity3;
	MYFLOATTYPE triangleIntensity = (intensity1 + intensity2 + intensity3) / (MYFLOATTYPE)3.0;

	RetinaImageChannel* dp_rawChannel = kernelLaunchParams.dp_rawChannel;
	//MYFLOATTYPE thetaR = retinaDescriptor.m_thetaR;
	SimpleRetinaDescriptor retinaDescriptor = kernelLaunchParams.retinaDescriptorIn;
	/*explain above: because SimpleRetinaDescriptor is an inherited class, it must be re-created...
	...inside the kernel so that the v-table is created on the device's side and polymorphism can work*/
	//RetinaImageChannel rawChannel(*dp_rawChannel);

	MYFLOATTYPE R0 = retinaDescriptor.m_R0;
#ifdef _MYDEBUGMODE
	if (ID == debugID) printf("m_R0 = %f, R0 = %f\n", retinaDescriptor.m_R0, R0);
#endif
	//TODO: if R0 is infinity (flat retina), take R0 equals distance between system's stop and the retina

	//for testing only, in real just take the translated version of z
	//if (ID == debugID) printf("vtx1.z before = %f\n", vtx1.z);
	vtx1.z = sqrt(R0*R0 - vtx1.x*vtx1.x - vtx1.y*vtx1.y);
	//if (ID == debugID) printf("vtx1.z after = %f\n", vtx1.z);
	vtx2.z = sqrt(R0*R0 - vtx2.x*vtx2.x - vtx2.y*vtx2.y);
	vtx3.z = sqrt(R0*R0 - vtx3.x*vtx3.x - vtx3.y*vtx3.y);
	//***********************core program part************************************
	//pick a point
	//first guess is the center of three vertices
	vec3<MYFLOATTYPE> vtxCenter = (vtx1 + vtx2 + vtx3) / static_cast<MYFLOATTYPE>(3);
	vtxCenter.z = sqrt(R0*R0 - vtxCenter.x*vtxCenter.x - vtxCenter.y*vtxCenter.y);

	vec3<MYFLOATTYPE> vtxs[] = {vtxCenter, vtx1, vtx2, vtx3};
	MYFLOATTYPE outputSearch = 0.0;

	point2D<int> pCur;

	for (int i = 0; i < 4; i++)
	{
		point2D<int> pGuess;
		//bool outputGuess = guessPoint(vtxs[i], thetaR, R0, pGuess);
		bool outputGuess = retinaDescriptor.cartesian2Array(vtxs[i], pGuess);
#ifdef _MYDEBUGMODE
		printf("outputGuess = %d\n", outputGuess);
#endif

		if (outputGuess == false)
		{
			continue;
		}

		pCur = pGuess;

		//extend until found a point inside
		int searchRadius = 0; // save a bit memory

		while (outputSearch < MYEPSILONSMALL && searchRadius < 2) //numerical inaccuracy, again
		{
			for (int j = -searchRadius; j <= searchRadius; j++)
			{
				for (int k = -searchRadius; k <= searchRadius; k++)
				{
					point2D<int> pSearch = { pGuess.nx + j, pGuess.ny + k };
					outputSearch = insideTriangle(pSearch, vtx1, vtx2, vtx3, pdir, retinaDescriptor);
#ifdef _MYDEBUGMODE
					printf("outputSearch = %f\n", outputSearch);
#endif
					if (outputSearch != 0.0)
					{
						pCur = pSearch;
						break;
					}
				}
				if (outputSearch != 0.0)
					break;
				
			}
			searchRadius = searchRadius + 1;
		}
		if (outputSearch != 0.0)
			break;
	}
#ifdef _MYDEBUGMODE
	if (ID == debugID) printf("ID %d outputSearch = %f\n", ID, outputSearch);
#endif
	//if found no point, quit
	if (outputSearch == 0)
	{
		printf("ID %d output search unsuccessful!, x,y position: %f, %f\n", ID, vtxCenter.x, vtxCenter.y);
		return;
	}
		
#ifdef _MYDEBUGMODE
	if (ID == debugID) printf("ID %d pCur = %d,%d\n", ID, pCur.x, pCur.y);
#endif
	//extend from that point
	//int nxSeed = pCur.x; 
	int nySeed = pCur.y;
	int increY = +1;
	int ny = nySeed;
	bool quit = false;
	int nxNextLineSeed = pCur.x;
	bool foundTurningPointSeed = false;
	int nxTurningPointSeed = 0;
	MYFLOATTYPE IOA = 0.0;

	while (!quit) // while not quit %the y loop
	{
		bool foundNextLineSeed = false;

		int nxTurnLR = nxNextLineSeed;
		//move right
		int nx = nxTurnLR;
		while ((IOA = 
			insideTriangle({ nx, ny }, vtx1, vtx2, vtx3, pdir, retinaDescriptor)*triangleIntensity
			//insideTriangle2({ nx, ny }, vtx1, vtx2, vtx3, pdir, retinaDescriptor, intensity1, intensity2, intensity3)
			) != 0) // the x loop
		{
			//if point is found, do something(save it)
			IOA = abs(IOA);
			//dp_rawChannel->addToPixel({ nx, ny }, triangleIntensity*IOA);
			dp_rawChannel->addToPixel({ nx, ny }, IOA);
#ifdef _MYDEBUGMODE
			if (ID == debugID) printf("ID %d added to pixel %d,%d\n", ID, nx, ny);
#endif
			//scan for next seed
			if (foundNextLineSeed == false)
			{
				if (insideTriangle({ nx, ny + increY }, vtx1, vtx2, vtx3, pdir, retinaDescriptor))
				{
					foundNextLineSeed = true;
					nxNextLineSeed = nx;
				}
			}

			//if it is the first line, scan for turning point seed
			if (ny == nySeed && foundTurningPointSeed == false && increY == +1)
			{
				if (insideTriangle({ nx, ny - 1 }, vtx1, vtx2, vtx3, pdir, retinaDescriptor))
				{
					foundTurningPointSeed = true;
					nxTurningPointSeed = nx;
				}
			}

			//move to the right
			nx = nx + 1;
		}
		
		//start over from the left
		nx = nxTurnLR - 1;
		while ((IOA = 
			insideTriangle({ nx, ny }, vtx1, vtx2, vtx3, pdir, retinaDescriptor)*triangleIntensity
			//insideTriangle2({ nx, ny }, vtx1, vtx2, vtx3, pdir, retinaDescriptor, intensity1, intensity2, intensity3)
			) != 0) // the x loop
		{
			//if point is found, do something(save it)
			IOA = abs(IOA);
			//dp_rawChannel->addToPixel({ nx, ny }, triangleIntensity*IOA);
			dp_rawChannel->addToPixel({ nx, ny }, IOA);
#ifdef _MYDEBUGMODE
			if (ID == debugID) printf("ID %d added to pixel %d,%d \n", ID, nx, ny);
#endif
			//scan for next seed
			if (foundNextLineSeed == false)
			{
				if (insideTriangle({ nx, ny + increY }, vtx1, vtx2, vtx3, pdir, retinaDescriptor))
				{
					foundNextLineSeed = true;
					nxNextLineSeed = nx;
				}
			}

			//if it is the first line, scan for turning point seed
			if (ny == nySeed && foundTurningPointSeed == false && increY == +1)
			{
				if (insideTriangle({ nx, ny - 1 }, vtx1, vtx2, vtx3, pdir, retinaDescriptor))
				{
					foundTurningPointSeed = true;
					nxTurningPointSeed = nx;
				}
			}

			//move to the left
			nx = nx - 1;
		}

		//move to next line
		if (foundNextLineSeed == true)
		{
			ny = ny + increY;
		}
		else if (increY == +1 && foundTurningPointSeed == true)
		{
			ny = nySeed - 1;
			nxNextLineSeed = nxTurningPointSeed;
			increY = -1;
		}
		else
		{
			quit = true;
		}
	}
}

/*
__device__ bool guessPoint
(const vec3<MYFLOATTYPE>& vtx, SimpleRetinaDescriptor retinaDescriptor, point2D<int>& pGuess)
{
	
	bool convertSuccess = retinaDescriptor.cartesian2Array(vtx, pGuess);

	if (!convertSuccess) return false;

	vec3<MYFLOATTYPE> p1, p2, p3, p4;
	bool output0 = find4points(pGuess.nx, pGuess.ny, thetaR, R0, p1, p2, p3, p4);

	if (!output0)
		return false;

	pGuess = { nx, ny };
	return true;
}
*/

/*
__device__ bool find4points
(int nx, int ny, MYFLOATTYPE thetaR, MYFLOATTYPE R0,
	vec3<MYFLOATTYPE>& p1, vec3<MYFLOATTYPE>& p2, vec3<MYFLOATTYPE>& p3, vec3<MYFLOATTYPE>& p4)
{
	p1 = vec3<MYFLOATTYPE>(0, 0, 0);
	p2 = p1; p3 = p1; p4 = p1;

	thetaR = thetaR / 180 * MYPI;
	MYFLOATTYPE r = thetaR * R0;
	MYFLOATTYPE thetaY1 = static_cast<MYFLOATTYPE>(ny) * thetaR;
	MYFLOATTYPE thetaY2 = (static_cast<MYFLOATTYPE>(ny) + 1)*thetaR;

	//the equal prevents numerical inaccuracies
	if ((thetaY1 <= -MYPI / 2) || (thetaY2 >= MYPI / 2))
		return false;


	p1.y = R0 * sin(thetaY1);
	p2.y = p1.y;
	p3.y = R0 * sin(thetaY2);
	p4.y = p3.y;
	MYFLOATTYPE R1p = R0 * cos(thetaY1);
	MYFLOATTYPE R2p = R0 * cos(thetaY2);

	MYFLOATTYPE thetaX1;
	MYFLOATTYPE thetaX2;

	if (ny >= 0)
	{
		thetaX1 = static_cast<MYFLOATTYPE>(nx) * r / R1p;
		thetaX2 = (static_cast<MYFLOATTYPE>(nx) + 1)*r / R1p;
	}
	else if (ny < 0)
	{
		thetaX1 = static_cast<MYFLOATTYPE>(nx) * r / R2p;
		thetaX2 = (static_cast<MYFLOATTYPE>(nx) + 1)*r / R2p;
	}

	if ((thetaX1 <= -MYPI / 2) || (thetaX2 >= MYPI / 2))
		return false;

	p1.x = R1p * sin(thetaX1);
	p2.x = R1p * sin(thetaX2);
	p3.x = R2p * sin(thetaX2);
	p4.x = R2p * sin(thetaX1);

	p1.z = sqrt(R0*R0 - p1.x * p1.x - p1.y * p1.y);
	p2.z = sqrt(R0*R0 - p2.x * p2.x - p2.y * p2.y);
	p3.z = sqrt(R0*R0 - p3.x * p3.x - p3.y * p3.y);
	p4.z = sqrt(R0*R0 - p4.x * p4.x - p4.y * p4.y);

	return true;
}
*/
__device__ int maptotriangle(
	const vec3<MYFLOATTYPE>& p1, const vec3<MYFLOATTYPE>& p2, const vec3<MYFLOATTYPE>& p3,
	const vec3<MYFLOATTYPE>& px, const vec3<MYFLOATTYPE>& dir, MYFLOATTYPE& alpha, MYFLOATTYPE& beta)
{

	//some tests here
	alpha = 0.0;
	beta = 0.0;
	vec3<MYFLOATTYPE> pdir = dir;
	MYFLOATTYPE A1, A2, A3, B1, B2, B3;

	vec3<MYFLOATTYPE> crossresult = cross(p2 - p1, p3 - p1);
	//if the triangle is colinear
	if (norm(crossresult) < MYEPSILONSMALL)
		return -1;

	//if pdir and triangle coplanar
	//printf("dot(cross(p2 - p1, p3 - p1), pdir) = %f\n", dot(cross(p2 - p1, p3 - p1), pdir));
	if (abs(dot(crossresult, pdir)) < MYEPSILONSMALL)
		return -1;

	//if px and triangle coplanar
	if (abs(dot(crossresult, px - p1)) < MYEPSILONSMALL)
	{
		pdir = { 0, 0, 1 };
	}


	//if any component of pdir is zero
	unsigned short int nrZero = 0;
	if (pdir.x == 0)
	{
		nrZero = nrZero + 1;
		A1 = px.x - p1.x;
		A2 = p2.x - p1.x;
		A3 = p3.x - p1.x;
	}

	if (pdir.y == 0)
	{
		if (nrZero == 1)
		{
			B1 = A1;
			B2 = A2;
			B3 = A3;
		}
		nrZero = nrZero + 1;
		A1 = px.y - p1.y;
		A2 = p2.y - p1.y;
		A3 = p3.y - p1.y;
	}

	if (pdir.z == 0)
	{
		if (nrZero == 1)
		{
			B1 = A1;
			B2 = A2;
			B3 = A3;
		}
		else if (nrZero == 2)
		{
			alpha = 0;
			beta = 0;
			return -1;
		}
		nrZero = nrZero + 1;
		A1 = px.z - p1.z;
		A2 = p2.z - p1.z;
		A3 = p3.z - p1.z;
	}

	if (nrZero == 0)
	{
		A1 = px.x*pdir.y - p1.x*pdir.y - px.y*pdir.x + p1.y*pdir.x;
		A2 = p2.x*pdir.y - p1.x*pdir.y - p2.y*pdir.x + p1.y*pdir.x;
		A3 = p3.x*pdir.y - p1.x*pdir.y - p3.y*pdir.x + p1.y*pdir.x;
	}

	if (nrZero <= 1)
	{
		MYFLOATTYPE denoLeft, numLeft1, numLeft2, numLeft3, numLeft4,
			denoRight, numRight1, numRight2, numRight3, numRight4;
		if (nrZero == 0 || pdir.y == 0)
		{
			denoLeft = pdir.x;
			numLeft1 = px.x;
			numLeft2 = p2.x;
			numLeft3 = p3.x;
			numLeft4 = p1.x;
			denoRight = pdir.z;
			numRight1 = px.z;
			numRight2 = p2.z;
			numRight3 = p3.z;
			numRight4 = p1.z;
		}
		else if (pdir.x == 0)
		{
			denoLeft = pdir.y;
			numLeft1 = px.y;
			numLeft2 = p2.y;
			numLeft3 = p3.y;
			numLeft4 = p1.y;
			denoRight = pdir.z;
			numRight1 = px.z;
			numRight2 = p2.z;
			numRight3 = p3.z;
			numRight4 = p1.z;
		}
		else if (pdir.z == 0)
		{
			denoLeft = pdir.x;
			numLeft1 = px.x;
			numLeft2 = p2.x;
			numLeft3 = p3.x;
			numLeft4 = p1.x;
			denoRight = pdir.y;
			numRight1 = px.y;
			numRight2 = p2.y;
			numRight3 = p3.y;
			numRight4 = p1.y;
		}
		B1 = (numLeft1 - numLeft4)*denoRight - (numRight1 - numRight4)*denoLeft;
		//printf("B1 = (numLeft1 - numLeft4)*denoRight - (numRight1 - numRight4)*denoLeft = \n%f = (%f - %f)*%f - (%f - %f)*%f\n", B1, numLeft1, numLeft4, denoRight, numRight1, numRight4, denoLeft);
		B2 = (numLeft2 - numLeft4)*denoRight - (numRight2 - numRight4)*denoLeft;
		B3 = (numLeft3 - numLeft4)*denoRight - (numRight3 - numRight4)*denoLeft;
	}

	alpha = (A1*B3 - B1 * A3) / (A2*B3 - B2 * A3);
	beta = (A2*B1 - B2 * A1) / (A2*B3 - B2 * A3);
	/*
	vec3<MYFLOATTYPE> signCheckVec = px - p1 - alpha * (p2 - p1) - beta * (p3 - p1);

	if (signCheckVec.z*pdir.z > 0)
	{
		return 1;
	}
	*/
	return 0;
}


__device__ bool sortCCW(const point2D<MYFLOATTYPE>& c1, const point2D<MYFLOATTYPE>& c2,
	const point2D<MYFLOATTYPE>& c3, const point2D<MYFLOATTYPE>& c4,
	point2D<MYFLOATTYPE>& p1, point2D<MYFLOATTYPE>& p2, point2D<MYFLOATTYPE>& p3, point2D<MYFLOATTYPE>& p4)
{
	p1 = c1; p2 = c2; p3 = c3; p4 = c4;

	point2D<MYFLOATTYPE> c12 = c2 - c1;
	point2D<MYFLOATTYPE> c13 = c3 - c1;
	point2D<MYFLOATTYPE> c14 = c4 - c1;

	MYFLOATTYPE crp1 = dot(c13, point2D<MYFLOATTYPE>(c12.y, -c12.x));
	MYFLOATTYPE crp2 = dot(c14, point2D<MYFLOATTYPE>(c12.y, -c12.x));

	if (crp1 == 0)
	{
		if (crp2 == 0)
		{
			//all colinear
			return false;
		}
		else
		{
			MYFLOATTYPE dotcolinear = dot(c13, normalize(c12));
			if (dotcolinear > norm(c12))
			{
				p3 = c3;
				if (crp2 > 0)
				{
					p2 = c4; p4 = c2;
					return true;
				}
				else
				{
					p2 = c2; p4 = c4;
					return true;
				}
			}
			else if (dotcolinear > 0)
			{
				p3 = c2;
				if (crp2 > 0)
				{
					p2 = c4; p4 = c3;
					return true;
				}
				else
				{
					p2 = c3; p4 = c4;
					return true;
				}
			}
			else
			{
				p3 = c4;
				if (crp2 > 0)
				{
					p2 = c3; p4 = c3;
					return true;
				}
				else
				{
					p2 = c2; p4 = c3;
					return true;
				}
			}
		}
	}
	else if (crp2 == 0)
	{
		MYFLOATTYPE dotcolinear = dot(c14, normalize(c12));
		if (dotcolinear > norm(c12))
		{
			p3 = c4;
			if (crp1 > 0)
			{
				p2 = c3; p4 = c2;
				return true;
			}
			else
			{
				p2 = c2; p4 = c3;
				return true;
			}
		}
		else if (dotcolinear > 0)
		{
			p3 = c2;
			if (crp1 > 0)
			{
				p2 = c3; p4 = c4;
				return true;
			}
			else
			{
				p2 = c4; p4 = c3;
				return true;
			}
		}
		else
		{
			p3 = c3;
			if (crp1 > 0)
			{
				p2 = c4; p4 = c2;
				return true;
			}
			else
			{
				p2 = c2; p4 = c4;
				return true;
			}
		}
	}

	MYFLOATTYPE dot1 = dot(c12, c13) / (norm(c12)*norm(c13));
	MYFLOATTYPE dot2 = dot(c12, c14) / (norm(c12)*norm(c14));

	if (crp1 > 0)
	{
		if (crp2 > 0)
		{
			p4 = c2;
			if (dot1 > dot2)
			{
				p2 = c4; p3 = c3;
				return true;
			}
			else if (dot1 < dot2)
			{
				p2 = c3; p3 = c4;
				return true;
			}
			else
			{
				//colinear
				if (norm(c13) >= norm(c14))
				{
					p2 = c4; p3 = c3;
					return true;
				}
				else
				{
					p2 = c3; p3 = c4;
					return true;
				}
			}
		}
		else
		{
			p2 = c3; p3 = c2; p4 = c4;
			return true;
		}
	}
	else
	{
		if (crp2 > 0)
		{
			p2 = c4; p3 = c2; p4 = c3;
			return true;
		}
		else
		{
			p2 = c2;
			if (dot1 > dot2)
			{
				p3 = c3; p4 = c4;
				return true;
			}
			else if (dot1 < dot2)
			{
				p3 = c4; p4 = c3;
				return true;
			}
			else
			{
				//colinear
				if (norm(c13) >= norm(c14))
				{
					p3 = c3; p4 = c4;
					return true;
				}
				else
				{
					p3 = c4; p4 = c3;
					return true;
				}
			}
		}
	}

	//return true;
}

__device__ MYFLOATTYPE SutherlandHogdman(const point2D<MYFLOATTYPE>& c1, const point2D<MYFLOATTYPE>& c2,
	const point2D<MYFLOATTYPE>& c3, const point2D<MYFLOATTYPE>& c4)
{
	point2D<MYFLOATTYPE> p1, p2, p3, p4;
	bool outputSort = sortCCW(c1, c2, c3, c4, p1, p2, p3, p4);

	if (outputSort == false)
		return 0;

	point2D<MYFLOATTYPE> inputList[7] = { p1, p2, p3, p4, 0, 0, 0 };
	short int inputListSize = 4;
	point2D<MYFLOATTYPE> outputList[7] = { p1, p2, p3, p4, 0, 0, 0 };
	short int outputListSize = 4;

	
	point2D<MYFLOATTYPE> pS, pE, pT;
	//edge along beta(or alpha = 0)
	for (int i = 0; i < outputListSize; i++)
	{
		inputList[i] = outputList[i];
	}
	inputListSize = outputListSize;
	//outputList = 0; not really needed
	outputListSize = 0;
	pS = inputList[inputListSize - 1];
	for (int i = 0; i < inputListSize; i++)
	{
		pE = inputList[i];
		//if (E inside clipEdge)
		if (pE.x >= 0)
		{
			//if (S not inside clipEdge)
			if (pS.x < 0)
			{
				//add intersection to the output list
				//pT = { 0, 0 }; not really needed
				pT.x = 0;
				pT.y = pS.y - pS.x*(pE.y - pS.y) / (pE.x - pS.x);
				outputList[outputListSize] = pT;
				outputListSize = outputListSize + 1;
			}
			//add pE to output list
			outputList[outputListSize] = pE;
			outputListSize = outputListSize + 1;
		}
		//else if (S inside clipEdge)
		else if (pS.x >= 0)
		{
			//add intersection to the output list
			//pT = [0; 0];
			pT.x = 0;
			pT.y = pS.y - pS.x*(pE.y - pS.y) / (pE.x - pS.x);
			outputList[outputListSize] = pT;
			outputListSize = outputListSize + 1;
		}
		pS = pE;
	}
	if (outputListSize == 0)
		return 0;


	//edge along alpha(or beta = 0)
	for (int i = 0; i < outputListSize; i++)
	{
		inputList[i] = outputList[i];
	}
	inputListSize = outputListSize;
	outputListSize = 0;
	pS = inputList[inputListSize - 1];
	for (int i = 0; i < inputListSize; i++)
	{
		pE = inputList[i];
		//if (E inside clipEdge)
		if (pE.y >= 0)
		{
			//if (S not inside clipEdge)
			if (pS.y < 0)
			{
				// add intersection to the output list
				//pT = [0; 0];
				pT.x = pS.x - pS.y*(pE.x - pS.x) / (pE.y - pS.y);
				pT.y = 0;
				outputList[outputListSize] = pT;
				outputListSize = outputListSize + 1;
			}
			//add pE to output list
			outputList[outputListSize] = pE;
			outputListSize = outputListSize + 1;
		}
		//else if (S inside clipEdge)
		else if (pS.y >= 0)
		{
			//add intersection to the output list
			//pT = [0; 0];
			pT.x = pS.x - pS.y*(pE.x - pS.x) / (pE.y - pS.y);
			pT.y = 0;
			outputList[outputListSize] = pT;
			outputListSize = outputListSize + 1;
		}
		pS = pE;
	}
	if (outputListSize == 0)
		return 0.0;


	//oblique edge
	for (int i = 0; i < outputListSize; i++)
	{
		inputList[i] = outputList[i];
	}
	inputListSize = outputListSize;
	outputListSize = 0;
	pS = inputList[inputListSize - 1];
	for (int i = 0; i < inputListSize; i++)
	{
		pE = inputList[i];
		//if (E inside clipEdge)
		if ((pE.x + pE.y) <= 1)
		{
			//if (S not inside clipEdge)
			if ((pS.x + pS.y) > 1)
			{
				//add intersection to the output list
				pT.x = (pE.x*(pS.y - pE.y) + (1 - pE.y)*(pS.x - pE.x)) / ((pS.x - pE.x) + (pS.y - pE.y));
				pT.y = 1 - pT.x;
				outputList[outputListSize] = pT;
				outputListSize = outputListSize + 1;
			}
			//add pE to output list
			outputList[outputListSize] = pE;
			outputListSize = outputListSize + 1;
		}
		//else if (S inside clipEdge)
		else if ((pS.x + pS.y) <= 1)
		{
			//add intersection to the output list
			pT.x = (pE.x*(pS.y - pE.y) + (1 - pE.y)*(pS.x - pE.x)) / ((pS.x - pE.x) + (pS.y - pE.y));
			pT.y = 1 - pT.x;
			outputList[outputListSize] = pT;
			outputListSize = outputListSize + 1;
		}
		pS = pE;
	}
	if (outputListSize == 0)
		return 0;
	
	MYFLOATTYPE output = outputList[outputListSize - 1].x*outputList[0].y - outputList[0].x*outputList[outputListSize - 1].y;

	for (int i = 0; i < outputListSize - 1; i++)
		output += outputList[i].x*outputList[i + 1].y - outputList[i + 1].x*outputList[i].y;

	return output;
}

__device__ MYFLOATTYPE SutherlandHogdman2(const point2D<MYFLOATTYPE>& c1, const point2D<MYFLOATTYPE>& c2,
	const point2D<MYFLOATTYPE>& c3, const point2D<MYFLOATTYPE>& c4, MYFLOATTYPE intensity1, MYFLOATTYPE intensity2, MYFLOATTYPE intensity3)
{
	point2D<MYFLOATTYPE> p1, p2, p3, p4;
	bool outputSort = sortCCW(c1, c2, c3, c4, p1, p2, p3, p4);

	if (outputSort == false)
		return 0;

	point2D<MYFLOATTYPE> inputList[7] = { p1, p2, p3, p4, 0, 0, 0 };
	short int inputListSize = 4;
	point2D<MYFLOATTYPE> outputList[7] = { p1, p2, p3, p4, 0, 0, 0 };
	short int outputListSize = 4;


	point2D<MYFLOATTYPE> pS, pE, pT;
	//edge along beta(or alpha = 0)
	for (int i = 0; i < outputListSize; i++)
	{
		inputList[i] = outputList[i];
	}
	inputListSize = outputListSize;
	//outputList = 0; not really needed
	outputListSize = 0;
	pS = inputList[inputListSize - 1];
	for (int i = 0; i < inputListSize; i++)
	{
		pE = inputList[i];
		//if (E inside clipEdge)
		if (pE.x >= 0)
		{
			//if (S not inside clipEdge)
			if (pS.x < 0)
			{
				//add intersection to the output list
				//pT = { 0, 0 }; not really needed
				pT.x = 0;
				pT.y = pS.y - pS.x*(pE.y - pS.y) / (pE.x - pS.x);
				outputList[outputListSize] = pT;
				outputListSize = outputListSize + 1;
			}
			//add pE to output list
			outputList[outputListSize] = pE;
			outputListSize = outputListSize + 1;
		}
		//else if (S inside clipEdge)
		else if (pS.x >= 0)
		{
			//add intersection to the output list
			//pT = [0; 0];
			pT.x = 0;
			pT.y = pS.y - pS.x*(pE.y - pS.y) / (pE.x - pS.x);
			outputList[outputListSize] = pT;
			outputListSize = outputListSize + 1;
		}
		pS = pE;
	}
	if (outputListSize == 0)
		return 0;


	//edge along alpha(or beta = 0)
	for (int i = 0; i < outputListSize; i++)
	{
		inputList[i] = outputList[i];
	}
	inputListSize = outputListSize;
	outputListSize = 0;
	pS = inputList[inputListSize - 1];
	for (int i = 0; i < inputListSize; i++)
	{
		pE = inputList[i];
		//if (E inside clipEdge)
		if (pE.y >= 0)
		{
			//if (S not inside clipEdge)
			if (pS.y < 0)
			{
				// add intersection to the output list
				//pT = [0; 0];
				pT.x = pS.x - pS.y*(pE.x - pS.x) / (pE.y - pS.y);
				pT.y = 0;
				outputList[outputListSize] = pT;
				outputListSize = outputListSize + 1;
			}
			//add pE to output list
			outputList[outputListSize] = pE;
			outputListSize = outputListSize + 1;
		}
		//else if (S inside clipEdge)
		else if (pS.y >= 0)
		{
			//add intersection to the output list
			//pT = [0; 0];
			pT.x = pS.x - pS.y*(pE.x - pS.x) / (pE.y - pS.y);
			pT.y = 0;
			outputList[outputListSize] = pT;
			outputListSize = outputListSize + 1;
		}
		pS = pE;
	}
	if (outputListSize == 0)
		return 0.0;


	//oblique edge
	for (int i = 0; i < outputListSize; i++)
	{
		inputList[i] = outputList[i];
	}
	inputListSize = outputListSize;
	outputListSize = 0;
	pS = inputList[inputListSize - 1];
	for (int i = 0; i < inputListSize; i++)
	{
		pE = inputList[i];
		//if (E inside clipEdge)
		if ((pE.x + pE.y) <= 1)
		{
			//if (S not inside clipEdge)
			if ((pS.x + pS.y) > 1)
			{
				//add intersection to the output list
				pT.x = (pE.x*(pS.y - pE.y) + (1 - pE.y)*(pS.x - pE.x)) / ((pS.x - pE.x) + (pS.y - pE.y));
				pT.y = 1 - pT.x;
				outputList[outputListSize] = pT;
				outputListSize = outputListSize + 1;
			}
			//add pE to output list
			outputList[outputListSize] = pE;
			outputListSize = outputListSize + 1;
		}
		//else if (S inside clipEdge)
		else if ((pS.x + pS.y) <= 1)
		{
			//add intersection to the output list
			pT.x = (pE.x*(pS.y - pE.y) + (1 - pE.y)*(pS.x - pE.x)) / ((pS.x - pE.x) + (pS.y - pE.y));
			pT.y = 1 - pT.x;
			outputList[outputListSize] = pT;
			outputListSize = outputListSize + 1;
		}
		pS = pE;
	}
	if (outputListSize == 0)
		return 0;

	//start calculating IOA
	MYFLOATTYPE output = outputList[outputListSize - 1].x*outputList[0].y - outputList[0].x*outputList[outputListSize - 1].y;
	
	//calculating IOA and barycenter in one loop
	point2D<MYFLOATTYPE> barycenter = { (MYFLOATTYPE)0.0,(MYFLOATTYPE)0.0 };
	for (int i = 0; i < outputListSize - 1; i++)
	{
		barycenter.alpha += (outputList[i].x);
		barycenter.alpha += (outputList[i].y);
		output += outputList[i].x*outputList[i + 1].y - outputList[i + 1].x*outputList[i].y;
	}

	barycenter.alpha = barycenter.alpha / (MYFLOATTYPE(outputListSize));
	barycenter.beta = barycenter.beta / (MYFLOATTYPE(outputListSize));

	MYFLOATTYPE averageValue = (MYFLOATTYPE(1.0) - barycenter.alpha - barycenter.beta)*intensity1 
								+ barycenter.alpha*intensity2 + barycenter.beta*intensity3;

	output = averageValue * output;

	return output;
}

__device__ MYFLOATTYPE insideTriangle(
	const point2D<int>& p, const vec3<MYFLOATTYPE>& vtx1, const vec3<MYFLOATTYPE>& vtx2,
	const vec3<MYFLOATTYPE>& vtx3, const vec3<MYFLOATTYPE>& pdir, const SimpleRetinaDescriptor& retinaDescriptor)
{
	vec3<MYFLOATTYPE> p1, p2, p3, p4;
	//bool output0 = find4points(p.nx, p.ny, thetaR, R0, p1, p2, p3, p4);
	bool output0 = retinaDescriptor.array2Cartesian(p, p1, p2, p3, p4);
	if (!output0)
		return 0.0;

	//MYFLOATTYPE alpha1, beta1, alpha2, beta2, alpha3, beta3, alpha4, beta4;
	point2D<MYFLOATTYPE> bp1, bp2, bp3, bp4;
	int output1 = maptotriangle(vtx1, vtx2, vtx3, p1, pdir, bp1.alpha, bp1.beta);
	int output2 = maptotriangle(vtx1, vtx2, vtx3, p2, pdir, bp2.alpha, bp2.beta);
	int output3 = maptotriangle(vtx1, vtx2, vtx3, p3, pdir, bp3.alpha, bp3.beta);
	int output4 = maptotriangle(vtx1, vtx2, vtx3, p4, pdir, bp4.alpha, bp4.beta);

	//build a caching mechanism here

	if (output1 == -1 || output2 == -1 || output3 == -1 || output4 == -1)
		return 0;

	MYFLOATTYPE returnValue = SutherlandHogdman(bp1, bp2, bp3, bp4);

	vec3<MYFLOATTYPE> testVec1 = cross(p2 - p1, p4 - p1);
	vec3<MYFLOATTYPE> testVec2 = cross(p3 - p2, p1 - p2);
	vec3<MYFLOATTYPE> testVec3 = cross(p4 - p3, p2 - p3);
	vec3<MYFLOATTYPE> testVec4 = cross(p1 - p4, p3 - p4);

	MYFLOATTYPE testDir1 = dot(testVec1, pdir);
	MYFLOATTYPE testDir2 = dot(testVec2, pdir);
	MYFLOATTYPE testDir3 = dot(testVec3, pdir);
	MYFLOATTYPE testDir4 = dot(testVec4, pdir);

	if (testDir1 > 0 && testDir2 > 0 && testDir3 > 0 && testDir4 > 0)
	{
		return 0;
	}

	/*
	if (output1 == 1 && output2 == 1 && output3 == 1 && output4 == 1 && returnValue < 1.0)
		return 0;
	*/

	return returnValue;
}

__device__ MYFLOATTYPE insideTriangle2(
	const point2D<int>& p, const vec3<MYFLOATTYPE>& vtx1, const vec3<MYFLOATTYPE>& vtx2,
	const vec3<MYFLOATTYPE>& vtx3, const vec3<MYFLOATTYPE>& pdir, const SimpleRetinaDescriptor& retinaDescriptor,
	MYFLOATTYPE intensity1, MYFLOATTYPE intensity2, MYFLOATTYPE intensity3)
{
	vec3<MYFLOATTYPE> p1, p2, p3, p4;
	//bool output0 = find4points(p.nx, p.ny, thetaR, R0, p1, p2, p3, p4);
	bool output0 = retinaDescriptor.array2Cartesian(p, p1, p2, p3, p4);
	if (!output0)
		return 0.0;

	//MYFLOATTYPE alpha1, beta1, alpha2, beta2, alpha3, beta3, alpha4, beta4;
	point2D<MYFLOATTYPE> bp1, bp2, bp3, bp4;
	int output1 = maptotriangle(vtx1, vtx2, vtx3, p1, pdir, bp1.alpha, bp1.beta);
	int output2 = maptotriangle(vtx1, vtx2, vtx3, p2, pdir, bp2.alpha, bp2.beta);
	int output3 = maptotriangle(vtx1, vtx2, vtx3, p3, pdir, bp3.alpha, bp3.beta);
	int output4 = maptotriangle(vtx1, vtx2, vtx3, p4, pdir, bp4.alpha, bp4.beta);

	//build a caching mechanism here

	if (output1 == -1 || output2 == -1 || output3 == -1 || output4 == -1)
		return 0;

	MYFLOATTYPE returnValue = SutherlandHogdman2(bp1, bp2, bp3, bp4, intensity1, intensity2, intensity3);

	vec3<MYFLOATTYPE> testVec1 = cross(p2 - p1, p4 - p1);
	vec3<MYFLOATTYPE> testVec2 = cross(p3 - p2, p1 - p2);
	vec3<MYFLOATTYPE> testVec3 = cross(p4 - p3, p2 - p3);
	vec3<MYFLOATTYPE> testVec4 = cross(p1 - p4, p3 - p4);

	MYFLOATTYPE testDir1 = dot(testVec1, pdir);
	MYFLOATTYPE testDir2 = dot(testVec2, pdir);
	MYFLOATTYPE testDir3 = dot(testVec3, pdir);
	MYFLOATTYPE testDir4 = dot(testVec4, pdir);

	if (testDir1 > 0 && testDir2 > 0 && testDir3 > 0 && testDir4 > 0)
	{
		return 0;
	}

	/*
	if (output1 == 1 && output2 == 1 && output3 == 1 && output4 == 1 && returnValue < 1.0)
		return 0;
	*/

	return returnValue;
}