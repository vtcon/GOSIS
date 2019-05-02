#include "Auxiliaries.cuh"
#include "StorageManager.cuh"
#include "ManagerFunctions.cuh"

#include "ProgramInterface.h"

#include <vector>

//externs, forward declarations...
extern __global__ void quadrictracer(QuadricTracerKernelLaunchParams kernelparams);
bool narrowingSweep(OpticalConfig* thisOpticalConfig, MYFLOATTYPE z_position, MYFLOATTYPE& startTheta, MYFLOATTYPE& endTheta);

//external global variable
extern int PI_ThreadsPerKernelLaunch;
extern int PI_linearRayDensity;

//global variable
static MYFLOATTYPE lastStepSize = 89; //this is bad, but I have no other solution

//for testing
extern StorageManager mainStorageManager;

OpticalConfig::EntrancePupilLocation locateSimpleEntrancePupil(OpticalConfig* currentConfig, int sweepDepth, int scanDepth, MYFLOATTYPE scanPitch)
{
	
	quadricsurface<MYFLOATTYPE>* p_firstSurface = nullptr;
	if (typeid(*(currentConfig->surfaces[0])) == typeid(quadricsurface<MYFLOATTYPE>)) //RTTI
	{
		p_firstSurface = static_cast<quadricsurface<MYFLOATTYPE>*>(currentConfig->surfaces[0]);
	}
	else
	{
		std::cerr << "Error: first surface is not a quadric surfaces\n";
	}
	
	MYFLOATTYPE startZ = p_firstSurface->pos.z;
	MYFLOATTYPE surfaceR = sqrt(abs(p_firstSurface->param.J));
	startZ = (p_firstSurface->antiParallel == true) ? startZ + surfaceR : startZ;

	struct ZThetaPair
	{
		MYFLOATTYPE z;
		MYFLOATTYPE theta;
	};
	std::vector<ZThetaPair> v_locator;


	for (int currentScanDepth = 0; currentScanDepth < scanDepth; currentScanDepth++)
	{
		ZThetaPair newPair;
		newPair.z = startZ + 100.0 + currentScanDepth * scanPitch;
		newPair.theta = 0;
		MYFLOATTYPE thetaEnd = 89;
		lastStepSize = 89;//reset globals before using

		for (int currentSweepDepth = 0; currentSweepDepth < sweepDepth; currentSweepDepth++)
		{
			narrowingSweep(currentConfig, newPair.z, newPair.theta, thetaEnd);
		}
		v_locator.push_back(newPair);
		//std::cout << "Found new pair: z=" << newPair.z << " theta=" << newPair.theta << "\n";
	}

	std::vector<point2D<MYFLOATTYPE>> v_rims;

	for (int i = 0; i < v_locator.size() - 1; i++)
	{
		MYFLOATTYPE L = sin(v_locator[i + 1].theta / 180 * MYPI) / sin((v_locator[i].theta - v_locator[i + 1].theta) / 180 * MYPI)
			*(v_locator[i + 1].z - v_locator[i].z);
		//std::cout << "Found new point L=" << L << " theta=" << v_locator[i].theta << "\n";
		v_rims.push_back({ L*sin(v_locator[i].theta / 180 * MYPI),v_locator[i].z - L * cos(v_locator[i].theta / 180 * MYPI) });
	}

	point2D<MYFLOATTYPE> result = { 0.0,0.0 };
	for (point2D<MYFLOATTYPE> eachPoint : v_rims)
	{
		result = result + eachPoint;
		//std::cout << "Found new point y=" << eachPoint.a << " z=" << eachPoint.b << "\n";
	}
	result = result / v_rims.size();
	std::cout << "Entrance pupil found at y=" << result.a << " z=" << result.b << "\n";

	currentConfig->setEntrancePupil({ result.a, result.b, 0.0 });

	return { result.a, result.b, 0.0 };
}


void SingleTest()
{
	//for testing
	OpticalConfig* thisOpticalConfig = nullptr;
	mainStorageManager.infoCheckOut(thisOpticalConfig, 555);
	locateSimpleEntrancePupil(thisOpticalConfig);
}

bool narrowingSweep(OpticalConfig* thisOpticalConfig, MYFLOATTYPE z_position, MYFLOATTYPE& startTheta, MYFLOATTYPE& endTheta)
{
	int numofsurfaces = thisOpticalConfig->numofsurfaces;

	//creating an array of ray bundles: in tracing job manager, data from object and image manager 

	raybundle<MYFLOATTYPE>* bundles = new raybundle<MYFLOATTYPE>[numofsurfaces + 1];

	//initialize the bundles
	int stepCount = 32;
	//bundles[0].init_1D_fan(z_position, startTheta, endTheta, 0.0, stepCount);
	init_1D_fan(&bundles[0], z_position, startTheta, endTheta, (MYFLOATTYPE)0.0, stepCount);
	for (int i = 1; i < numofsurfaces + 1; i++)
	{
		bundles[i] = bundles[0];
	}
	int rays_per_bundle = bundles[0].size;

	//create 2 bundles to pass in and out the kernel: also in tracing job manager
	LOG1("[main]creating 2 siblings bundles\n");
	raybundle<MYFLOATTYPE> h_inbundle = bundles[0];
	raybundle<MYFLOATTYPE> h_outbundle = bundles[0];
	h_inbundle.copytosibling();
	h_outbundle.copytosibling();


	//job creation by cuda malloc: also in tracing job manager
	int job_size = 1;
	raybundle<MYFLOATTYPE>** d_injob;
	cudaMalloc((void**)&d_injob, job_size * sizeof(raybundle<MYFLOATTYPE>*));
	cudaMemcpy(d_injob, &(h_inbundle.d_sibling), sizeof(raybundle<MYFLOATTYPE>*), cudaMemcpyHostToDevice);
	raybundle<MYFLOATTYPE>** d_outjob;
	cudaMalloc((void**)&d_outjob, job_size * sizeof(raybundle<MYFLOATTYPE>*));
	cudaMemcpy(d_outjob, &(h_outbundle.d_sibling), sizeof(raybundle<MYFLOATTYPE>*), cudaMemcpyHostToDevice);

	//create the launch parameters
	QuadricTracerKernelLaunchParams thisparam;
	thisparam.d_inbundles = d_injob;
	thisparam.d_outbundles = d_outjob;

	thisparam.otherparams[0] = job_size;
	thisparam.otherparams[1] = rays_per_bundle;

	int kernelToLaunch = job_size * rays_per_bundle;

	int threadsToLaunch = PI_ThreadsPerKernelLaunch;
	int blocksToLaunch = (kernelToLaunch + threadsToLaunch - 1) / (threadsToLaunch);


	// launch kernel, copy result out, swap memory
	for (int i = 0; i < numofsurfaces; i++)
	{
		//create an object for param

		thisparam.otherparams[2] = i;
		thisparam.pquad = static_cast<quadricsurface<MYFLOATTYPE>*>(((thisOpticalConfig->surfaces)[i])->d_sibling);

		quadrictracer <<< blocksToLaunch, threadsToLaunch >>> (thisparam);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__);
			fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		}

		cudaDeviceSynchronize();
		LOG1("[main]copy sibling out");
		bundles[i + 1] = (i % 2 == 0) ? h_outbundle.copyfromsibling() : h_inbundle.copyfromsibling();

		swap(thisparam.d_inbundles, thisparam.d_outbundles);
	}


	//writing results out
	/*
	for (int i = 0; i < rays_per_bundle; i++)
	{
		LOG2("ray " << i);
		for (int j = 0; j < numofsurfaces + 1; j++)
		{
			switch ((bundles[j].prays)[i].status)
			{
			case (raysegment<MYFLOATTYPE>::Status::deactivated):
				LOG2(" deactivated")
					break;
			case (raysegment<MYFLOATTYPE>::Status::active):
				LOG2(" " << (bundles[j].prays)[i])
					break;
			case (raysegment<MYFLOATTYPE>::Status::finished):
				if ((bundles[j - 1].prays)[i].status != raysegment<MYFLOATTYPE>::Status::deactivated)
					LOG2(" " << (bundles[j].prays)[i] << " done")
					break;
			}
		}
		LOG2("\n");
	}
	*/

	//scan the penultimate raybundle of the column, as we want to find the entrance pupil and not mixing with the field stop (of the retina)
	int iStart = 0;
	for (; iStart < rays_per_bundle && (bundles[numofsurfaces-1].prays)[iStart].status == raysegment<MYFLOATTYPE>::Status::active; iStart++);
	iStart--; //we have swept one past the last active ray

	//check for inconsistency (i.e. the iStart ray could run though all the surfaces)
	bool toReturn = true;
	if ((bundles[numofsurfaces-1].prays)[iStart].status != raysegment<MYFLOATTYPE>::Status::active)
		toReturn = false;

	//narrow down
	MYFLOATTYPE stepSize = (endTheta - startTheta) / stepCount;

	if (stepSize < lastStepSize)
	{
		startTheta = startTheta + stepSize * iStart;
		endTheta = startTheta + stepSize;
		lastStepSize = stepSize;
	}

	cudaFree(d_injob);
	cudaFree(d_outjob);
	delete[] bundles;

	return toReturn;
}

template<typename T>
void init_1D_fan(raybundle<T>* bundle, T z_position, T startTheta, T endTheta, T phi, int insize)
{
	bundle->cleanObject();
	bundle->size = insize;
	bundle->prays = new raysegment<T>[bundle->size];
	bundle->samplinggrid = new point2D<int>[bundle->size];

	if (startTheta == endTheta)
		endTheta = startTheta + 1;

	startTheta = (startTheta >= T(0.0) && startTheta < T(90.0)) ? startTheta : (T)0.0;
	startTheta = startTheta / (T)180.0 * (T)MYPI;
	endTheta = (endTheta > startTheta && endTheta < (T)90.0) ? endTheta : (T)89.0;
	endTheta = endTheta / (T)180.0 * (T)MYPI;

	T step = (endTheta - startTheta) / bundle->size;
	T currentTheta = startTheta;
	
	for (int i = 0; i < bundle->size; i++)
	{
		currentTheta = startTheta + step * i;
		vec3<T> dir;
		dir.z = -cos(currentTheta);
		dir.x = sin(currentTheta)*sin(phi);
		dir.y = sin(currentTheta)*cos(phi);
		bundle->prays[i] = raysegment<T>(vec3<T>((T)0.0, (T)0.0, z_position), dir);
		bundle->prays[i].intensity = 1.0f;
		bundle->samplinggrid[i] = point2D<int>(i, 0);
		/*
		printf("i = %d, (u,v) = (%d,%d), pos = (%f,%f,%f), dir = (%f,%f,%f) \n", i
			, bundle->samplinggrid[i].u, bundle->samplinggrid[i].v
			, bundle->prays[i].pos.x, bundle->prays[i].pos.y, bundle->prays[i].pos.z
			, bundle->prays[i].dir.x, bundle->prays[i].dir.y, bundle->prays[i].dir.z);
		*/
	}
	
	return;
}

template<typename T>
void init_2D_dualpolar(raybundle<T>* bundle, vec3<T> originpos, T min_horz, T max_horz, T min_vert, T max_vert, T step)
{
	//clamping the limits to pi/2
	min_horz = (min_horz < -(T)MYPI / (T)2.0) ? -(T)MYPI / (T)2.0 : min_horz;
	min_vert = (min_vert < -(T)MYPI / (T)2.0) ? -(T)MYPI / (T)2.0 : min_vert;
	max_horz = (max_horz > (T)MYPI / (T)2.0) ? (T)MYPI / (T)2.0 : max_horz;
	max_vert = (max_vert > (T)MYPI / (T)2.0) ? (T)MYPI / (T)2.0 : max_vert;

	//checking the max and min limits, they must be at least one step apart
	min_horz = (min_horz > max_horz - step) ? (max_horz - step) : min_horz;
	min_vert = (min_vert > max_vert - step) ? (max_vert - step) : min_vert;


	int temp_size = static_cast<int>((max_horz / step - min_horz / step + 1)*
		(max_vert / step - min_vert / step + 1));

	//for safety, reclean the object before initialization
	bundle->cleanObject();
	/*
	if (d_sibling != nullptr) freesibling();
	delete[] prays;
	delete[] samplinggrid;
	size = 0;
	*/

	//assign temporary memory
	raysegment<T>* temp_prays = new raysegment<T>[temp_size];
	point2D<int>* temp_samplinggrid = new point2D<int>[temp_size];

	//declaration
	T angle_horz;
	T angle_vert;
	T semi_axis_horz;
	T semi_axis_vert;
	T x, y, z;

	for (int i = static_cast<int>(min_horz / step); i < (max_horz / step) + 1; i++)
	{
		for (int j = static_cast<int>(min_vert / step); j < (max_vert / step) + 1; j++)
		{
			//if the sampling point is within ellipse-bound and smaller than pi/2
			angle_horz = i * step;
			angle_vert = j * step;
			semi_axis_horz = (angle_horz < 0) ? min_horz : max_horz;
			semi_axis_vert = (angle_vert < 0) ? min_vert : max_vert;
			if (((angle_horz / semi_axis_horz)*(angle_horz / semi_axis_horz) +
				(angle_vert / semi_axis_vert)*(angle_vert / semi_axis_vert)
				<= (T)1.0)
				&& (angle_horz < (T)MYPI / (T)2.0 && angle_vert < (T)MYPI / 2.0)
				&& (sin(angle_horz)*sin(angle_horz) + sin(angle_vert)*sin(angle_vert) <= (T)1.0)
				)
			{
				//register
				temp_samplinggrid[bundle->size] = point2D<int>(i, j);
				/*
				z = -1 / sqrt(1 + tan(angle_horz)*tan(angle_horz) + tan(angle_vert)*tan(angle_vert));
				x = -z * tan(angle_horz);
				y = -z * tan(angle_vert);
				*/
				x = sin(angle_horz);
				y = sin(angle_vert);
				z = -sqrt((T)1.0 - x * x - y * y);
				temp_prays[bundle->size] = raysegment<T>(originpos, vec3<T>(x, y, z));
				bundle->size += 1;
			}
		}
	}

	//copy temporary arrays to final arrays
	bundle->prays = new raysegment<T>[bundle->size];
	bundle->samplinggrid = new point2D<int>[bundle->size];
	memcpy(bundle->prays, temp_prays, bundle->size * sizeof(raysegment<T>));
	memcpy(bundle->samplinggrid, temp_samplinggrid, bundle->size * sizeof(point2D<int>));
	delete[] temp_prays;
	delete[] temp_samplinggrid;

	//debugging: printout trace
#ifdef _MYDEBUGMODE
#ifdef _DEBUGMODE2
	if (bundle->samplinggrid != nullptr && bundle->prays != nullptr)
	{
		for (int i = 0; i < bundle->size; i++)
		{
			printf("i = %d\t (u,v) = (%d,%d)\t at (%f,%f,%f)\t pointing (%f,%f,%f)\n", i,
				bundle->samplinggrid[i].u, bundle->samplinggrid[i].v,
				bundle->prays[i].pos.x, bundle->prays[i].pos.y, bundle->prays[i].pos.z,
				bundle->prays[i].dir.x, bundle->prays[i].dir.y, bundle->prays[i].dir.z);
		}
	}
	else printf("error: null ptr detected");
#endif
#endif

	return;
}

template<typename T>
void init_2D_dualpolar_v2(raybundle<T>* bundle, OpticalConfig* thisOpticalConfig, vec3<T> origin, T step)
{
	T epr = static_cast<T>(thisOpticalConfig->entrance.y);
	T epz = static_cast<T>(thisOpticalConfig->entrance.z);

	//find the vertical extends
	T deltaz = abs(origin.z - epz);
	T rho = sqrt((T)origin.x*(T)origin.x + (T)origin.y*(T)origin.y);
	if ((T)origin.y > (T)0.0) rho = -rho;
	vec3<T> center = vec3<T>(deltaz, rho, 0);
	vec3<T> up = center + vec3<T>(0, epr, 0);
	vec3<T> down = center + vec3<T>(0, -epr, 0);
	T thetaup = asin(norm(cross(center, up)) / (norm(center)*norm(up)));
	T thetadown = asin(norm(cross(center, down)) / (norm(center)*norm(down)));

	//find the horizontal extends
	T d = sqrt(rho*rho + deltaz * deltaz);
	T phiMax = atan(epr / d);
	T phiMin = atan(-epr / d);

	//clamping the step size to the min of any of the four extends
	if (step <= 0.0)
	{
		step = phiMax;
		std::cerr << "WARNING: size of step is too small!\n";
	}
	if (phiMax < step) step = phiMax;
	if (abs(phiMin) < step) step = abs(phiMin);
	if (thetaup < step) step = thetaup;
	if (thetadown < step) step = thetadown;
	

	//TODO: fix this
	T min_horz = phiMin;
	T min_vert = -thetadown;
	T max_horz = phiMax;
	T max_vert = thetaup;

	//find the transformation matrix
	vec3<T> kp((T)origin.x, (T)origin.y, (T)origin.z - epz);
	vec3<T> jp((T)0.0, (T)1.0, (T)0.0);
	vec3<T> ip((T)1.0, (T)0.0, (T)0.0);
	if ((T)origin.x == (T)0.0 && (T)origin.y == (T)0.0)
	{
		kp = vec3<T>((T)0.0, (T)0.0, (T)1.0);
	}
	else
	{
		jp = vec3<T>((T)abs(origin.x), (T)abs(origin.y), (T)0.0);
		jp.z = (-jp.x*kp.x - jp.y*kp.y) / (kp.z);
		ip = cross(jp, kp);
		ip = normalize(ip);
		jp = normalize(jp);
		kp = normalize(kp);
	}
	vec3<T> transformmat[3] = { vec3<T>(ip.x,jp.x,kp.x),vec3<T>(ip.y,jp.y,kp.y) ,vec3<T>(ip.z,jp.z,kp.z) };

	//calculate a guess for array size
	int temp_size = static_cast<int>((max_horz / step - min_horz / step + 1)*
		(max_vert / step - min_vert / step + 1));

	//for safety, reclean the object before initialization
	bundle->cleanObject();
	
	//assign temporary memory
	raysegment<T>* temp_prays = new raysegment<T>[temp_size];
	point2D<int>* temp_samplinggrid = new point2D<int>[temp_size];

	//declaration
	T angle_horz;
	T angle_vert;
	T semi_axis_horz;
	T semi_axis_vert;
	T x, y, z;

#define QUADRAND_ALL
#ifdef QUADRAND_ALL
	for (int i = static_cast<int>(min_horz / step); i < (max_horz / step) + 1; i++)
	{
		for (int j = static_cast<int>(min_vert / step); j < (max_vert / step) + 1; j++)
		{
#endif
#ifdef QUADRAND_TOPLEFT
	for (int i = static_cast<int>(min_horz / step); i <= 0; i++)
	{
		for (int j = 0; j < (max_vert / step) + 1; j++)
		{
#endif
#ifdef QUADRAND_TOPRIGHT
	for (int i = 0; i < (max_horz / step) + 1; i++)
	{
		for (int j = 0; j < (max_vert / step) + 1; j++)
		{
#endif
#ifdef QUADRAND_BOTTOMRIGHT
	for (int i = 0; i < (max_horz / step) + 1; i++)
	{
		for (int j = static_cast<int>(min_vert / step); j <= 0; j++)
		{
#endif
#ifdef QUADRAND_BOTTOMLEFT
	for (int i = static_cast<int>(min_horz / step); i <= 0; i++)
	{
		for (int j = static_cast<int>(min_vert / step); j <= 0; j++)
		{
#endif
			//if the sampling point is within ellipse-bound and smaller than pi/2
			angle_horz = i * step;
			angle_vert = j * step;
			semi_axis_horz = (angle_horz < 0) ? min_horz : max_horz;
			semi_axis_vert = (angle_vert < 0) ? min_vert : max_vert;
			if (((angle_horz / semi_axis_horz)*(angle_horz / semi_axis_horz) +
				(angle_vert / semi_axis_vert)*(angle_vert / semi_axis_vert)
				<= 1)
				&& (angle_horz < MYPI / 2 && angle_vert < MYPI / 2)
				&& (sin(angle_horz)*sin(angle_horz) + sin(angle_vert)*sin(angle_vert) <= 1)
				)
			{
				//register
				temp_samplinggrid[bundle->size] = point2D<int>(i, j);
				/*
				z = -1 / sqrt(1 + tan(angle_horz)*tan(angle_horz) + tan(angle_vert)*tan(angle_vert));
				x = -z * tan(angle_horz);
				y = -z * tan(angle_vert);
				*/
				x = sin(angle_horz);
				y = sin(angle_vert);
				z = -sqrt(1 - x * x - y * y);
				vec3<T> pretransformed(x, y, z);
				vec3<T> transformed(dot(pretransformed, transformmat[0]), dot(pretransformed, transformmat[1]), dot(pretransformed, transformmat[2]));
				temp_prays[bundle->size] = raysegment<T>(origin, transformed);
				bundle->size += 1;
			}
		}
	}

	//copy temporary arrays to final arrays
	bundle->prays = new raysegment<T>[bundle->size];
	bundle->samplinggrid = new point2D<int>[bundle->size];
	memcpy(bundle->prays, temp_prays, bundle->size * sizeof(raysegment<T>));
	memcpy(bundle->samplinggrid, temp_samplinggrid, bundle->size * sizeof(point2D<int>));
	delete[] temp_prays;
	delete[] temp_samplinggrid;

	//debugging: printout trace
#ifdef _MYDEBUGMODE
#ifdef _DEBUGMODE2
	if (bundle->samplinggrid != nullptr && bundle->prays != nullptr)
	{
		for (int i = 0; i < bundle->size; i++)
		{
			printf("i = %d\t (u,v) = (%d,%d)\t at (%f,%f,%f)\t pointing (%f,%f,%f)\n", i,
				bundle->samplinggrid[i].u, bundle->samplinggrid[i].v,
				bundle->prays[i].pos.x, bundle->prays[i].pos.y, bundle->prays[i].pos.z,
				bundle->prays[i].dir.x, bundle->prays[i].dir.y, bundle->prays[i].dir.z);
		}
	}
	else printf("error: null ptr detected");
#endif
#endif

	return;
}

template<typename T>
void init_2D_dualpolar_v3(raybundle<T>* bundle, OpticalConfig* thisOpticalConfig, LuminousPoint point)
{
	vec3<T> origin(T(point.x), T(point.y), T(point.z));

	T epr = static_cast<T>(thisOpticalConfig->entrance.y);
	T epz = static_cast<T>(thisOpticalConfig->entrance.z);

	//find the vertical extends
	T deltaz = abs(origin.z - epz);
	T rho = sqrt(origin.x*origin.x + origin.y*origin.y);
	if (origin.y > 0) rho = -rho;
	vec3<T> center = vec3<T>(deltaz, rho, 0);
	vec3<T> up = center + vec3<T>(0, epr, 0);
	vec3<T> down = center + vec3<T>(0, -epr, 0);
	T thetaup = asin(norm(cross(center, up)) / (norm(center)*norm(up)));
	T thetadown = asin(norm(cross(center, down)) / (norm(center)*norm(down)));

	//find the horizontal extends
	T d = sqrt(rho*rho + deltaz * deltaz);
	T phiMax = atan(epr / d);
	T phiMin = atan(-epr / d);

	//clamping the step size to the min of any of the four extends
	T step = T(0.1);
	if ((phiMax + abs(phiMin)) <= (thetaup + abs(thetadown)))
	{
		step = (phiMax + abs(phiMin)) / PI_linearRayDensity;
	}
	else
	{
		step = (thetaup + abs(thetadown)) / PI_linearRayDensity;
	}

	//TODO: fix this
	T min_horz = phiMin;
	T min_vert = -thetadown;
	T max_horz = phiMax;
	T max_vert = thetaup;

	//find the transformation matrix
	vec3<T> kp(origin.x, origin.y, origin.z - epz);
	vec3<T> jp(0, 1, 0);
	vec3<T> ip(1, 0, 0);
	if (origin.x == 0 && origin.y == 0)
	{
		kp = vec3<T>(0, 0, 1);
	}
	else
	{
		jp = vec3<T>(abs(origin.x), abs(origin.y), 0);
		jp.z = (-jp.x*kp.x - jp.y*kp.y) / (kp.z);
		ip = cross(jp, kp);
		ip = normalize(ip);
		jp = normalize(jp);
		kp = normalize(kp);
	}
	vec3<T> transformmat[3] = { vec3<T>(ip.x,jp.x,kp.x),vec3<T>(ip.y,jp.y,kp.y) ,vec3<T>(ip.z,jp.z,kp.z) };

	//calculate a guess for array size
	int temp_size = static_cast<int>((max_horz / step - min_horz / step + 1)*
		(max_vert / step - min_vert / step + 1));

	//for safety, reclean the object before initialization
	bundle->cleanObject();

	//assign temporary memory
	raysegment<T>* temp_prays = new raysegment<T>[temp_size];
	point2D<int>* temp_samplinggrid = new point2D<int>[temp_size];

	//declaration
	T angle_horz;
	T angle_vert;
	T semi_axis_horz;
	T semi_axis_vert;
	T x, y, z;

#define QUADRAND_ALL
#ifdef QUADRAND_ALL
	for (int i = static_cast<int>(min_horz / step); i < (max_horz / step) + 1; i++)
	{
		for (int j = static_cast<int>(min_vert / step); j < (max_vert / step) + 1; j++)
		{
#endif
			//if the sampling point is within ellipse-bound and smaller than pi/2
			angle_horz = i * step;
			angle_vert = j * step;
			semi_axis_horz = (angle_horz < 0) ? min_horz : max_horz;
			semi_axis_vert = (angle_vert < 0) ? min_vert : max_vert;
			if (((angle_horz / semi_axis_horz)*(angle_horz / semi_axis_horz) +
				(angle_vert / semi_axis_vert)*(angle_vert / semi_axis_vert)
				<= 1)
				&& (angle_horz < MYPI / 2 && angle_vert < MYPI / 2)
				&& (sin(angle_horz)*sin(angle_horz) + sin(angle_vert)*sin(angle_vert) <= 1)
				)
			{
				//register
				temp_samplinggrid[bundle->size] = point2D<int>(i, j);
				/*
				z = -1 / sqrt(1 + tan(angle_horz)*tan(angle_horz) + tan(angle_vert)*tan(angle_vert));
				x = -z * tan(angle_horz);
				y = -z * tan(angle_vert);
				*/
				x = sin(angle_horz);
				y = sin(angle_vert);
				z = -sqrt(1 - x * x - y * y);
				vec3<T> pretransformed(x, y, z);
				vec3<T> transformed(dot(pretransformed, transformmat[0]), dot(pretransformed, transformmat[1]), dot(pretransformed, transformmat[2]));
				temp_prays[bundle->size] = raysegment<T>(origin, transformed);
				temp_prays[bundle->size].intensity = 1.0f*float(step)*float(step)*point.intensity;
				bundle->size += 1;
			}
		}
	}

	//copy temporary arrays to final arrays
	bundle->prays = new raysegment<T>[bundle->size];
	bundle->samplinggrid = new point2D<int>[bundle->size];
	memcpy(bundle->prays, temp_prays, bundle->size * sizeof(raysegment<T>));
	memcpy(bundle->samplinggrid, temp_samplinggrid, bundle->size * sizeof(point2D<int>));
	delete[] temp_prays;
	delete[] temp_samplinggrid;

	//debugging: printout trace
#ifdef _MYDEBUGMODE
#ifdef _DEBUGMODE2
	if (bundle->samplinggrid != nullptr && bundle->prays != nullptr)
	{
		for (int i = 0; i < bundle->size; i++)
		{
			printf("i = %d\t (u,v) = (%d,%d)\t at (%f,%f,%f)\t pointing (%f,%f,%f)\n", i,
				bundle->samplinggrid[i].u, bundle->samplinggrid[i].v,
				bundle->prays[i].pos.x, bundle->prays[i].pos.y, bundle->prays[i].pos.z,
				bundle->prays[i].dir.x, bundle->prays[i].dir.y, bundle->prays[i].dir.z);
		}
	}
	else printf("error: null ptr detected");
#endif
#endif

	return;
}


//template's explicit instantiation
template
void init_1D_fan<double>(raybundle<double>* bundle, double z_position, double startTheta, double endTheta, double phi, int insize);

template
void init_1D_fan<float>(raybundle<float>* bundle, float z_position, float startTheta, float endTheta, float phi, int insize);

template
void init_2D_dualpolar<double>(raybundle<double>* bundle, vec3<double> originpos, double min_horz, double max_horz, double min_vert, double max_vert, double step);

template
void init_2D_dualpolar<float>(raybundle<float>* bundle, vec3<float> originpos, float min_horz, float max_horz, float min_vert, float max_vert, float step);

template
void init_2D_dualpolar_v2<float>(raybundle<float>* bundle, OpticalConfig* thisOpticalConfig, vec3<float> origin, float step);

template
void init_2D_dualpolar_v2<double>(raybundle<double>* bundle, OpticalConfig* thisOpticalConfig, vec3<double> origin, double step);

template
void init_2D_dualpolar_v3<float>(raybundle<float>* bundle, OpticalConfig* thisOpticalConfig, LuminousPoint point);

template
void init_2D_dualpolar_v3<double>(raybundle<double>* bundle, OpticalConfig* thisOpticalConfig, LuminousPoint point);