#include "ManagerFunctions.cuh"
#include "StorageManager.cuh"
#include "../ConsoleApplication/src/ImageFacilities.h"
#include "Auxiliaries.cuh"

#include "ProgramInterface.h"

#include <vector>
#include <algorithm>
#include <list>
#include <thread>
#include <mutex>
#include <chrono>

//global variables

//external global variables
//as the storage lies in this project, the regulator function should also be here
extern StorageManager mainStorageManager;
extern float activeWavelength;
extern int PI_ThreadsPerKernelLaunch;
extern int PI_traceJobSize;
extern int PI_renderJobSize;
extern int PI_maxParallelThread;
extern float PI_renderProgress;

// external function definitions:
extern __global__ void quadrictracer(QuadricTracerKernelLaunchParams kernelparams);
extern __global__ void nonDiffractiveBasicRenderer(RendererKernelLaunchParams kernelLaunchParams);

//just an interface
class GPUJob
{
public:
	//enum JobStatus { unconstructed, underconstruction, readytolaunch, jobinprogress, jobfinised };

	bool isEmpty = true;

	//JobStatus jobStatus = unconstructed;
	virtual ~GPUJob() = 0; //virtual destructor, so that delete works properly with polymorphic object

	virtual void preLaunchPreparation() = 0;

	// should the kernel launcher launch again?
	virtual bool goAhead() const = 0;

	//both functions below should also have an Async variants, for CUDA streams
	virtual void kernelLaunch() = 0;

	virtual void update() = 0;

	virtual void postLaunchCleanUp() = 0;
};

inline GPUJob::~GPUJob() {} //this comes directly from stackoverflow, reason is a linker error when base destructor is unimplemented

class QuadricTracerJob :public GPUJob
{
public:
	QuadricTracerKernelLaunchParams kernelLaunchParams;

	int wanted_job_size = PI_traceJobSize; //settable from outside
	int job_size = 0; // real size of a batch depends on how many columns are left in the Storage
	int numofsurfaces = 0;
	float wavelength = 0.0;

	RayBundleColumn** pcolumns = nullptr;
	OpticalConfig* thisOpticalConfig = nullptr;
	raybundle<MYFLOATTYPE>* b_inbundles = nullptr;
	raybundle<MYFLOATTYPE>* b_outbundles = nullptr;

	int currentSurfaceCount = 0;

	int blocksToLaunch = 0;
	int threadsToLaunch = 0;

	typedef StorageHolder<RayBundleColumn*>::Status columnStatus;

	QuadricTracerJob(int _wanted_job_size = PI_traceJobSize) :wanted_job_size(_wanted_job_size)
	{
		pcolumns = new RayBundleColumn*[wanted_job_size];

		//BIG QUESTION: where does the wavelength comes from?
		wavelength = activeWavelength;

		//this is bad coding: job_size is used here as the counting variable
		while ((job_size < wanted_job_size) && mainStorageManager.takeOne(pcolumns[job_size], columnStatus::initialized, wavelength))
		{
			job_size++;
		}

		if (job_size == 0)
		{
			isEmpty = true; //no job to be done, signal to the calling function
		}
		else
		{
			isEmpty = false;
		}
	}

	void preLaunchPreparation() override
	{
		if (isEmpty) return;

		mainStorageManager.infoCheckOut(thisOpticalConfig, wavelength);
		numofsurfaces = thisOpticalConfig->numofsurfaces;
		b_inbundles = new raybundle<MYFLOATTYPE>[job_size];
		b_outbundles = new raybundle<MYFLOATTYPE>[job_size];
		for (int i = 0; i < job_size; i++)
		{
			b_inbundles[i] = (*pcolumns[i])[0];
			b_outbundles[i] = (*pcolumns[i])[0]; //this seems redundant, but it initializes the outbundles to size different than 0
			b_inbundles[i].copytosibling();
			b_outbundles[i].copytosibling();
		}

		CUDARUN(cudaMalloc((void**)&(kernelLaunchParams.d_inbundles), job_size * sizeof(raybundle<MYFLOATTYPE>*)));
		CUDARUN(cudaMalloc((void**)&(kernelLaunchParams.d_outbundles), job_size * sizeof(raybundle<MYFLOATTYPE>*)));
		for (int i = 0; i < job_size; i++)
		{
			CUDARUN(cudaMemcpy(kernelLaunchParams.d_inbundles + i,
				&(b_inbundles[i].d_sibling),
				sizeof(raybundle<MYFLOATTYPE>*), cudaMemcpyHostToDevice));
			CUDARUN(cudaMemcpy(kernelLaunchParams.d_outbundles + i,
				&(b_outbundles[i].d_sibling),
				sizeof(raybundle<MYFLOATTYPE>*), cudaMemcpyHostToDevice));
		}

		kernelLaunchParams.otherparams[0] = job_size;
		//blocksToLaunch = job_size;
		int maxBundleSize = 0;
		for (int i = 0; i < job_size; i++)
		{
			int temp = (*pcolumns[i])[0].size;
			if (maxBundleSize < temp)
			{
				maxBundleSize = temp;
			}
		}
		//threadsToLaunch = maxBundleSize;
		kernelLaunchParams.otherparams[1] = maxBundleSize;
		int kernelToLaunch = job_size * maxBundleSize;

		threadsToLaunch = PI_ThreadsPerKernelLaunch;
		blocksToLaunch = (kernelToLaunch + threadsToLaunch - 1) / (threadsToLaunch);

		std::cout << "Tracing " << maxBundleSize << " rays from each of " << job_size << " points\n";
	}

	// should the kernel launcher launch again?
	bool goAhead() const override
	{
		if (isEmpty) return false;
		return currentSurfaceCount < numofsurfaces;
	}

	void kernelLaunch() override
	{
		if (isEmpty) return;

		kernelLaunchParams.otherparams[2] = currentSurfaceCount;
		kernelLaunchParams.pquad = static_cast<quadricsurface<MYFLOATTYPE>*>((*thisOpticalConfig)[currentSurfaceCount]->d_sibling);

		quadrictracer <<<blocksToLaunch, threadsToLaunch >>> (kernelLaunchParams);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__);
			fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		}
		cudaDeviceSynchronize();
	}

	//update the kernel launch parameters, decrease stages_left count, swap pointers etc.
	void update() override
	{
		if (isEmpty) return;

		for (int i = 0; i < job_size; i++)
		{
			if (currentSurfaceCount % 2 == 0)
			{
				(*pcolumns[i])[currentSurfaceCount + 1] = b_outbundles[i].copyfromsibling();
			}
			else
			{
				(*pcolumns[i])[currentSurfaceCount + 1] = b_inbundles[i].copyfromsibling();
			}
		}

		swap(kernelLaunchParams.d_inbundles, kernelLaunchParams.d_outbundles);

		currentSurfaceCount += 1;
	}

	void postLaunchCleanUp() override
	{
		if (isEmpty) return;

		//marking the traced columns as "completed1"
		for (int i = 0; i < job_size; i++)
		{
			mainStorageManager.jobCheckIn(pcolumns[i], columnStatus::completed1);
		}
#ifdef _MYDEBUGMODE
		//writing results out:
		for (int i = 0; i < job_size; i++)
		{
			int rays_per_bundle = (*pcolumns[i])[0].size;
			for (int j = 0; j < rays_per_bundle; j++)
			{
				LOG2("ray " << j);
				for (int k = 0; k < numofsurfaces + 1; k++)
				{
					switch (((*pcolumns[i])[k].prays)[j].status)
					{
					case (raysegment<MYFLOATTYPE>::Status::deactivated):
						LOG2(" deactivated")
							break;
					case (raysegment<MYFLOATTYPE>::Status::active):
						LOG2(" " << ((*pcolumns[i])[k].prays)[j])
							break;
					case (raysegment<MYFLOATTYPE>::Status::finished):
						if (((*pcolumns[i])[k - 1].prays)[j].status != raysegment<MYFLOATTYPE>::Status::deactivated)
							LOG2(" " << ((*pcolumns[i])[k].prays)[j] << " done")
							break;
					}
				}
				LOG2("\n");
			}
		}
#endif
	}

	~QuadricTracerJob()
	{
		LOG1("QuadricTracerJob destructor called");
		delete[] pcolumns;
		delete[] b_inbundles;
		delete[] b_outbundles;
		cudaFree(kernelLaunchParams.d_inbundles);
		cudaFree(kernelLaunchParams.d_outbundles);
	}

private:

	inline void swap(raybundle<MYFLOATTYPE>**& a, raybundle<MYFLOATTYPE>**& b)
	{
		raybundle<MYFLOATTYPE>** temp = a;
		a = b;
		b = temp;
	}
};

class TriangleRendererJob :public GPUJob
{
public:
	RendererKernelLaunchParams m_kernelLaunchParams;
	//SimpleRetinaDescriptor* p_retinaDescriptor;
	PixelArrayDescriptor* p_retinaDescriptor;
	RetinaImageChannel* p_rawChannel;

	int wanted_job_size = PI_renderJobSize;
	int job_size = 0; 
	float wavelength = 0;
	long m_triangleCount = 0;
	
	mutable int m_launchCount = 0; 

	RayBundleColumn** pcolumns = nullptr;
	typedef StorageHolder<RayBundleColumn*>::Status columnStatus;

	TriangleRendererJob(int _wanted_job_size = PI_renderJobSize) :wanted_job_size(_wanted_job_size)
	{
		pcolumns = new RayBundleColumn*[wanted_job_size];

		//BIG QUESTION: where does the wavelength comes from?
		wavelength = activeWavelength;

		//this is bad coding: job_size is used here as the counting variable
		while ((job_size < wanted_job_size) && mainStorageManager.takeOne(pcolumns[job_size], columnStatus::completed1, wavelength))
		{
			job_size++;
		}

		if (job_size == 0)
		{
			isEmpty = true; //no job to be done, signal to the calling function
		}
		else
		{
			isEmpty = false;
		}
	}

	~TriangleRendererJob() override
	{}

	void preLaunchPreparation() override
	{
		if (isEmpty) return;
		
		/*
		raybundle<MYFLOATTYPE> testbundle;
		testbundle.init_2D_dualpolar(vec3<MYFLOATTYPE>(-10, 0, 0), -1, 1, -1, 1, 0.45);

		for (int i = 0; i < testbundle.size; i++)
		{
			(testbundle.prays)[i].status = raysegment<MYFLOATTYPE>::Status::finished;
		}
		*/

		std::list<std::thread> list_thread;

		//for (int i = 0; i < job_size; i++)
		//{
		//	auto lastindex = (pcolumns[i])->numofsurfaces; //two code lines, make it easier to debug
		//	RenderingTrianglesCreator((*pcolumns[i])[lastindex]);
		//}

		for (int i = 0; i < job_size; i++)
		{
			auto lastindex = (pcolumns[i])->numofsurfaces; //two code lines, make it easier to debug
			list_thread.emplace_back(&TriangleRendererJob::RenderingTrianglesCreator, this, std::ref((*pcolumns[i])[lastindex]));
			//RenderingTrianglesCreator((*pcolumns[i])[lastindex]);
		}
		
		std::for_each(list_thread.begin(), list_thread.end(), std::mem_fn(&std::thread::join));

		transferMeshToDevice();

		cudaDeviceSynchronize();

		//TODO: add these to the storage manager
		//data setup
		//p_retinaDescriptor = new SimpleRetinaDescriptor(2, 20);
		//p_rawChannel = new RetinaImageChannel(*p_retinaDescriptor);
		OpticalConfig* thisOpticalConfig = nullptr;
		mainStorageManager.infoCheckOut(thisOpticalConfig, wavelength);
		p_retinaDescriptor = thisOpticalConfig->p_retinaDescriptor;
		p_rawChannel = thisOpticalConfig->p_rawChannel;

		//the following line should not be here
		//p_rawChannel->createSibling();

		//setup the kernel launch params
		//TODO: FIX THIS POLYMORPHISM!!
		m_kernelLaunchParams.retinaDescriptorIn = *(dynamic_cast<SimpleRetinaDescriptor*>(p_retinaDescriptor));
		m_kernelLaunchParams.dp_triangles = dp_triangles;
		m_kernelLaunchParams.dp_rawChannel = p_rawChannel->dp_sibling;
		m_kernelLaunchParams.otherparams[0] = m_triangleCount;

		std::cout << "Rendering " << m_triangleCount << " triangles from " << job_size << " points\n";
	}

	// should the kernel launcher launch again?
	bool goAhead() const override
	{
		// it runs just once
		if (m_launchCount == 0)
		{
			m_launchCount++;
			return true;
		}
		return false;
	}

	//both functions below should also have an Async variants, for CUDA streams
	void kernelLaunch() override
	{
		if (isEmpty) return;
		if (m_triangleCount == 0)
		{
			std::cerr << "Warning, rendering job has no triangles\n";
			return;
		}
		int threadsToLaunch = PI_ThreadsPerKernelLaunch;
		int blocksToLaunch = (m_triangleCount + static_cast<int>(threadsToLaunch)-1) / (threadsToLaunch);

		//for debugging
		//threadsToLaunch = 1;
		//blocksToLaunch = 1;

		nonDiffractiveBasicRenderer <<<blocksToLaunch, threadsToLaunch >>> (m_kernelLaunchParams);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__);
			fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		}
		cudaDeviceSynchronize();
	}

	void update() override
	{
		//nothing here, as it runs just once
	}

	void postLaunchCleanUp() override
	{
		if (isEmpty) return;

		/*
		//data copy out
		p_rawChannel->copyFromSibling();

		//for testing
		//p_rawChannel->setToValue(1.0, *p_retinaDescriptor);

		//display data with scaling
		void* mapX = nullptr;
		void* mapY = nullptr;
		SimpleRetinaDescriptor* tempDescriptor = dynamic_cast<SimpleRetinaDescriptor*>(p_retinaDescriptor);
		MYFLOATTYPE scalingargs[4] = { tempDescriptor->m_thetaR, tempDescriptor->m_R0, tempDescriptor->m_maxTheta, tempDescriptor->m_maxPhi };
		generateProjectionMap(mapX, mapY, p_rawChannel->m_dimension.y, p_rawChannel->m_dimension.x, IF_PROJECTION_ALONGZ, 4, scalingargs);
		quickDisplayv2(p_rawChannel->hp_raw, p_rawChannel->m_dimension.y, p_rawChannel->m_dimension.x, mapX, mapY, 700);
		//quickDisplayv2<MYFLOATTYPE>(p_rawChannel->hp_raw, p_rawChannel->m_dimension.y, p_rawChannel->m_dimension.x, IF_PROJECTION_NONE, 850);
		
		//quickDisplay(p_rawChannel->hp_raw, p_rawChannel->m_dimension.y, p_rawChannel->m_dimension.x, 700);

		//these two lines shouldn't be here, too
		p_rawChannel->deleteSibling();
		p_rawChannel->deleteHostImage();
		*/

		//TODO: copy to the output image

		//delete the ray columns and free the resources
		CUDARUN(cudaFree(dp_triangles));
		
	}

private:
	std::mutex v_triangles_lock;
	std::list<PerKernelRenderingInput> v_triangles;
	PerKernelRenderingInput* hp_triangles = nullptr;
	PerKernelRenderingInput* dp_triangles = nullptr;

	class PointSearcher //basically a hash table implementation to speed up the search
	{
	public:
		PointSearcher(point2D<int>* p_points, int bundleSize) :mppoints(p_points),totalCount(bundleSize)
		{
			if (mppoints == nullptr)
				return;

			//build up hash structure here

			//1st pass
			for (int i = 0; i < totalCount; i++)
			{
				int value = mppoints[i].u;
				offsetLower = (value < offsetLower) ? value : offsetLower;
				offsetUpper = (value > offsetUpper) ? value : offsetUpper;
			}

			stageOneCount = offsetUpper - offsetLower + 1;

			if (offsetUpper == 0 || offsetLower == 0) std::cout << "Offset is 0!\n";

			mtable = new std::vector<pointIndex>[stageOneCount];

			//2nd pass
			for (int i = 0; i < totalCount; i++)
			{
				int value = hash(mppoints[i]);
				mtable[value].push_back({ mppoints[i],i });
			}
		}

		~PointSearcher()
		{
			//clean up resources
			delete[] mtable;
			mtable = nullptr;
		}

		int find(const point2D<int>& tofind) const
		{
			if (mppoints == nullptr)
				return -1;

			//do the search here
			int value = hash(tofind);

			if (value < 0 || value >= stageOneCount)
				return -1;

			for (int i = 0; i < mtable[value].size(); i++)
			{
				if (tofind == mtable[value][i].point)
					return  mtable[value][i].index;
			}

			return -1;
		}


	private:
		point2D<int>* mppoints = nullptr;
		int totalCount = 0;
		int offsetLower = 0;
		int offsetUpper = 0;
		int stageOneCount = 0;

		struct pointIndex
		{
			point2D<int> point;
			int index;
		};

		std::vector<pointIndex>* mtable;

		inline int hash(const point2D<int>& tohash) const
		{
			return tohash.u - offsetLower;
		}
	};

	int RenderingTrianglesCreator(const raybundle<MYFLOATTYPE>& thisbundle) //build up vector containing the mesh
	{
#ifdef _MYDEBUGMODE
		std::cout << "Tesselating...\n";
#endif
		//test data
		int arraySize = thisbundle.size;
		point2D<int>* inputArray = thisbundle.samplinggrid;

		if (arraySize == 0) std::cout << "Zero size bundle!\n";

		////lambda for searching
		//auto searchForPoint = [&](const point2D<int>& p) -> int {
		//	for (int i = 0; i < arraySize; i++)
		//	{
		//		if (p == inputArray[i])
		//		{
		//			//only finished rays should be rendered
		//			if ((thisbundle.prays)[i].status == raysegment<MYFLOATTYPE>::Status::finished)
		//			return i;
		//		}
		//	}
		//	return -1;
		//};

		//new search version
		PointSearcher searcher(inputArray, arraySize);
		auto searchForPoint = [&](const point2D<int>& p) -> int {
			int ret = searcher.find(p);
			if (ret != -1)
			{
				if ((thisbundle.prays)[ret].status == raysegment<MYFLOATTYPE>::Status::finished)
				return ret;
			}
			return -1;
		};

		//vector for saving result
		struct IndexTriangle { int i1, i2, i3; };
		std::vector<IndexTriangle> indexTriangles;

		//scan and build up the triangle mesh
		for (int i = 0; i < arraySize; i++)
		{
			int results[6];
			results[0] = searchForPoint(inputArray[i]);
			results[1] = searchForPoint({ inputArray[i].x, inputArray[i].y + 1 });
			results[2] = searchForPoint({ inputArray[i].x + 1, inputArray[i].y + 1 });
			results[3] = searchForPoint({ inputArray[i].x + 1, inputArray[i].y });
			results[4] = searchForPoint({ inputArray[i].x + 1, inputArray[i].y - 1 });
			results[5] = searchForPoint({ inputArray[i].x, inputArray[i].y - 1 });

			//if (results[0] == -1) std::cout << "Result 0 is -1\n";

			if (results[0] != -1)
			{
				if (results[1] != -1)
				{
					if (results[2] != -1)
					{
						if (results[3] != -1)
						{
							indexTriangles.push_back({ results[0], results[2],results[1] });
							indexTriangles.push_back({ results[0], results[3],results[2] });
						}
						else
						{
							indexTriangles.push_back({ results[0], results[2],results[1] });
						}
					}
					else if (results[3] != -1)
					{
						indexTriangles.push_back({ results[0], results[3],results[1] });
					}
				}
				else if (results[2] != -1 && results[3] != -1)
				{
					indexTriangles.push_back({ results[0], results[3],results[2] });
				}
			}
			/*
			else if (results[1] != -1 && results[2] != -1 && results[3] != -1)
			{
				indexTriangles.push_back({ results[1], results[3],results[2] });
			}
			*/
			if (results[5] == -1 && results[0] != -1 && results[3] != -1 && results[4] != -1)
			{
				indexTriangles.push_back({ results[0], results[4],results[3] });
			}
		}

		//debug print
#ifdef _MYDEBUGMODE
		for (IndexTriangle current : indexTriangles)
		{
			std::cout << "[ " << current.i1 << ", " << current.i2 << ", " << current.i3 << "]\n";
		}
#endif	

		//parse the built-up mesh and copy data in vector
		{
			std::lock_guard<std::mutex> lock(v_triangles_lock);
			for (IndexTriangle indextriangle : indexTriangles)
			{
				v_triangles.push_back({
					(thisbundle.prays)[indextriangle.i1].pos,
					(thisbundle.prays)[indextriangle.i2].pos,
					(thisbundle.prays)[indextriangle.i3].pos,
					(thisbundle.prays)[indextriangle.i1].dir,
					(thisbundle.prays)[indextriangle.i2].dir,
					(thisbundle.prays)[indextriangle.i3].dir,
					(thisbundle.prays)[indextriangle.i1].intensity,
					(thisbundle.prays)[indextriangle.i2].intensity,
					(thisbundle.prays)[indextriangle.i3].intensity
					});
			}
		}
		

		return 0;
	}

	int transferMeshToDevice() //as the name suggests....
	{
		//delete early
		for (int i = 0; i < job_size; i++)
		{
			mainStorageManager.pleaseDelete(pcolumns[i]);
		}
		delete[] pcolumns;

		m_triangleCount = v_triangles.size();

		hp_triangles = new PerKernelRenderingInput[m_triangleCount];

		auto v_triangle_iterator = v_triangles.begin();
		for (long i = 0; i < m_triangleCount; i++)
		{
			hp_triangles[i] = *v_triangle_iterator;
			v_triangle_iterator++;
		}

		CUDARUN(cudaMalloc((void**)&dp_triangles, m_triangleCount * sizeof(PerKernelRenderingInput)));
		CUDARUN(cudaMemcpy(dp_triangles, hp_triangles, m_triangleCount * sizeof(PerKernelRenderingInput), cudaMemcpyHostToDevice));

		delete[] hp_triangles;
		hp_triangles = nullptr;
		v_triangles.clear();

		return 0;
	}
};

//managers implementation
int OpticalConfigManager(int argc, char** argv)
{
	//input: set up the surfaces manually, or get data from console
	LOG1("[main]setup the surfaces\n");
	MYFLOATTYPE diam = 40;
	int numofsurfaces = 3;
	float wavelength1 = 555.0;

	//check out an opticalconfig as output
	OpticalConfig* newConfig = nullptr;
	mainStorageManager.jobCheckOut(newConfig, numofsurfaces, wavelength1);

	//construct the surfaces
	//(newConfig->surfaces)[0] = new quadricsurface<MYFLOATTYPE>(mysurface<MYFLOATTYPE>::SurfaceTypes::refractive, quadricparam<MYFLOATTYPE>(1, 1, 1, 0, 0, 0, 0, 0, 0, -400), 1, 1.5168, vec3<MYFLOATTYPE>(0, 0, 38.571), diam);
	bool output = constructSurface((newConfig->surfaces)[0], MF_REFRACTIVE, vec3<MYFLOATTYPE>(0, 0, 25.933), 40.0, diam, MF_CONVEX, 1.0, 2.5168);
	output = constructSurface((newConfig->surfaces)[1], MF_REFRACTIVE, vec3<MYFLOATTYPE>(0, 0, 10.933), 40.0, diam, MF_CONCAVE, 2.5168, 1.0);

	//(newConfig->surfaces)[1] = new quadricsurface<MYFLOATTYPE>(mysurface<MYFLOATTYPE>::SurfaceTypes::image, quadricparam<MYFLOATTYPE>(0, 0, 0, 0, 0, 0, 0, 0, 1, 0), 1, INFINITY, vec3<MYFLOATTYPE>(0, 0, 0), diam);
	//output = constructSurface((newConfig->surfaces)[1], MF_IMAGE, vec3<MYFLOATTYPE>(0, 0, 0), FP_INFINITE, diam, MF_FLAT, 1.0, FP_INFINITE);
	output = constructSurface((newConfig->surfaces)[2], MF_IMAGE, vec3<MYFLOATTYPE>(0, 0, 0), 20, diam, MF_CONCAVE, 1.0, FP_INFINITE);

	// rules for constructing image surface:
	// + retina vertex always at 0,0,0
	// + the surface is always either spherical concave, or flat

	//test adding data
	char* teststr = "hello";
	((newConfig->surfaces)[0])->add_data(teststr, 6);

	//copy to sibling surfaces on GPU side
	LOG1("[main]create sibling surfaces\n");
	newConfig->copytosiblings();

	//find entrance pupil
	locateSimpleEntrancePupil(newConfig);

	//add a retina
	PixelArrayDescriptor* p_retinaDescriptor = nullptr;
	MYFLOATTYPE angularResolution = 1.6; //0.16 or 0.016 for release, 2.0 for debug
	MYFLOATTYPE angularExtend = 90; //in degrees
	MYFLOATTYPE R = 10; //random initial value
	if (dynamic_cast<quadricsurface<MYFLOATTYPE>*>(newConfig->surfaces[numofsurfaces - 1])->isFlat())
	{
		R = newConfig->entrance.z; //take the distance from entrance pupil to retina vertex as R
	}
	else //if it is not flat
	{
		MYFLOATTYPE Rsqr = abs(dynamic_cast<quadricsurface<MYFLOATTYPE>*>(newConfig->surfaces[numofsurfaces - 1])->param.J);
		R = sqrt(Rsqr);
	}
	p_retinaDescriptor = new SimpleRetinaDescriptor(angularResolution, R, angularExtend, angularExtend); //doesn't need to explicitly delete this
	newConfig->createImageChannel(p_retinaDescriptor);

	return 0;
}

int KernelLauncher(int argc, char** argv)//this is the non-Async variant
{
	GPUJob* job = new QuadricTracerJob; // still need to initialize this by new or by getting from main memory

	if (job->isEmpty)//no jobs to be done
	{
		delete job; //TODO: fix why the destructor wasn't called
		return -2;
	}

	//create event for timing: to GPU manager
	/*
	cudaEvent_t start, stop;
	CUDARUN(cudaEventCreate(&start));
	CUDARUN(cudaEventCreate(&stop));
	*/
	job->preLaunchPreparation();
	/*
	//start timing 
	CUDARUN(cudaDeviceSynchronize());
	CUDARUN(cudaEventRecord(start, 0));
	*/
	while (job->goAhead())
	{
		job->kernelLaunch();
		job->update();
	}
	/*
	//kernel finished, stop timing, print out elapsed time: in gpu manager
	CUDARUN(cudaEventRecord(stop, 0));
	CUDARUN(cudaEventSynchronize(stop));
	float elapsedtime;
	CUDARUN(cudaEventElapsedTime(&elapsedtime, start, stop));
	std::cout<<"kernel run time: " << elapsedtime << " ms\n";
	CUDARUN(cudaEventDestroy(start));
	CUDARUN(cudaEventDestroy(stop));
	*/
	job->postLaunchCleanUp();

	//delete the jobs
	delete job;

	return 0;
}

int KernelLauncher2(int argc, char** argv)
{
	GPUJob* job = new TriangleRendererJob; // still need to initialize this by new or by getting from main memory

	if (job->isEmpty)//no jobs to be done
	{
		delete job; //TODO: fix why the destructor wasn't called
		return -2;
	}
	/*
	//create event for timing: to GPU manager
	cudaEvent_t start, stop;
	CUDARUN(cudaEventCreate(&start));
	CUDARUN(cudaEventCreate(&stop));
	*/
	job->preLaunchPreparation();
	/*
	//start timing 
	CUDARUN(cudaEventRecord(start));
	CUDARUN(cudaEventSynchronize(start));
	*/
	while (job->goAhead())
	{
		job->kernelLaunch();
		job->update();
	}
	/*
	//kernel finished, stop timing, print out elapsed time: in gpu manager
	CUDARUN(cudaEventRecord(stop));
	CUDARUN(cudaEventSynchronize(stop)); 
	float elapsedtime;
	CUDARUN(cudaEventElapsedTime(&elapsedtime, start, stop));
	std::cout << "kernel run time: " << elapsedtime << " ms\n";
	CUDARUN(cudaEventDestroy(start));
	CUDARUN(cudaEventDestroy(stop));
	CUDARUN(cudaDeviceSynchronize());
	*/
	job->postLaunchCleanUp();

	//delete the jobs
	delete job;

	return 0;
}

/*int QuadricTracerJobPreparator(int argc = 0, char** argv = nullptr)
{
	int wanted_job_size = 3;
	int job_size = 0; // real size of a batch depends on how many columns are left in the Storage

	auto pcolumns = new RayBundleColumn*[wanted_job_size];

	//this is bad coding: job_size is used here as the counting variable
	while ((job_size < wanted_job_size) && mainStorageManager.takeOne(pcolumns[job_size]))
	{
		job_size++;
	}



	QuadricTracerJob* newjob = nullptr;
	//mainStorageManager.jobCheckOut(newjob,jobsize); // check out a new job as output

	//aggregating columns and prepare the job

	return 0;
}*/

int ColumnCreator(int argc, char** argv)
{
	//where does the wavelength come from??
	float wavelength1 = activeWavelength;
	
	//if there is no job to do in the storage
	if (false) return -2;

	//get the number of surfaces
	OpticalConfig* thisOpticalConfig = nullptr;
	mainStorageManager.infoCheckOut(thisOpticalConfig, wavelength1);
	int numofsurfaces = thisOpticalConfig->numofsurfaces;

	//get a column pointer from the main storage
	RayBundleColumn* job = nullptr;
	mainStorageManager.jobCheckOut(job, numofsurfaces, wavelength1);

	//call initializer of the first bundle in column
	(*job)[0].init_2D_dualpolar(vec3<MYFLOATTYPE>(0, 0, 80), -3.0/180*MYPI, 3.0 / 180 * MYPI, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, 0.7 / 180 * MYPI);
	
	//mark the column as initialized after initializing it
	mainStorageManager.jobCheckIn(job, StorageHolder<RayBundleColumn*>::Status::initialized);

	return 0;
}

int ColumnCreator2(vec3<MYFLOATTYPE> point)
{
	//where does the wavelength come from??
	float wavelength1 = activeWavelength;

	//if there is no job to do in the storage
	if (false) return -2;

	//get the number of surfaces
	OpticalConfig* thisOpticalConfig = nullptr;
	mainStorageManager.infoCheckOut(thisOpticalConfig, wavelength1);
	int numofsurfaces = thisOpticalConfig->numofsurfaces;

	//get a column pointer from the main storage
	RayBundleColumn* job = nullptr;
	mainStorageManager.jobCheckOut(job, numofsurfaces, wavelength1);

	//call initializer of the first bundle in column
	//(*job)[0].init_2D_dualpolar(point, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, 0.7 / 180 * MYPI);
	//init_2D_dualpolar(&((*job)[0]), point, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, 0.7 / 180 * MYPI);
	init_2D_dualpolar_v2(&((*job)[0]), thisOpticalConfig, point, (MYFLOATTYPE)0.1 / (MYFLOATTYPE)180 * (MYFLOATTYPE)MYPI); //0.1 for release, 2.0 for debug

	//mark the column as initialized after initializing it
	mainStorageManager.jobCheckIn(job, StorageHolder<RayBundleColumn*>::Status::initialized);

	return 0;
}

int ColumnCreator3()
{
	float wavelength1 = activeWavelength;

	LuminousPoint* p_point = nullptr;
	bool output = mainStorageManager.takeOne(p_point, StorageHolder<LuminousPoint>::Status::uninitialized, wavelength1);
	if (!output) //no jobs to do
		return -2;

	//get the number of surfaces
	OpticalConfig* thisOpticalConfig = nullptr;
	mainStorageManager.infoCheckOut(thisOpticalConfig, wavelength1);
	int numofsurfaces = thisOpticalConfig->numofsurfaces;

	//get a column pointer from the main storage
	RayBundleColumn* job = nullptr;
	mainStorageManager.jobCheckOut(job, numofsurfaces, wavelength1);

	//vec3<MYFLOATTYPE> point(p_point->x, p_point->y, p_point->z);

	//call initializer of the first bundle in column
	//(*job)[0].init_2D_dualpolar(point, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, 0.7 / 180 * MYPI);
	//init_2D_dualpolar(&((*job)[0]), point, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, 0.7 / 180 * MYPI);
	//init_2D_dualpolar_v2(&((*job)[0]), thisOpticalConfig, point, 2.0 / 180 * MYPI); //0.1 for release, 2.0 for debug
	init_2D_dualpolar_v3(&((*job)[0]), thisOpticalConfig, *p_point);

	mainStorageManager.jobCheckIn(p_point, StorageHolder<LuminousPoint>::Status::initialized);

	//mark the column as initialized after initializing it
	mainStorageManager.jobCheckIn(job, StorageHolder<RayBundleColumn*>::Status::initialized);

	return 0;
}

int ColumnCreator4()
{
	float wavelength1 = activeWavelength;

	//get the number of surfaces
	OpticalConfig* thisOpticalConfig = nullptr;
	if (!mainStorageManager.infoCheckOut(thisOpticalConfig, wavelength1))
	{
		std::cout << "Cannot find optical config at " << wavelength1 << " nm\n";
		return -2;
	}
	int numofsurfaces = thisOpticalConfig->numofsurfaces;

	std::list<LuminousPoint*> list_p_point;
	std::list<RayBundleColumn*> list_job;
	std::list<std::thread> list_thread;

	int runCount = 0;
	for (int i = 0; i < PI_traceJobSize; i++, runCount++)
	{
		LuminousPoint* temp_p_point = nullptr;
		bool output = mainStorageManager.takeOne(temp_p_point, StorageHolder<LuminousPoint>::Status::uninitialized, wavelength1);
		if (!output) //no jobs to do
			break;
		list_p_point.push_back(temp_p_point);

		RayBundleColumn* temp_job = nullptr;
		if (mainStorageManager.jobCheckOut(temp_job, numofsurfaces, wavelength1))
			list_job.push_back(temp_job);

		list_thread.emplace_back(init_2D_dualpolar_v3<MYFLOATTYPE>, &((*(list_job.back()))[0]), thisOpticalConfig, *(list_p_point.back()));
	}

	if (runCount == 0)
	{
		return -2;
	}

	std::for_each(list_thread.begin(), list_thread.end(), std::mem_fn(&std::thread::join));

	for (auto each_p_point : list_p_point)
	{
		mainStorageManager.jobCheckIn(each_p_point, StorageHolder<LuminousPoint>::Status::initialized);
	}

	for (auto each_job : list_job)
	{
		//mark the column as initialized after initializing it
		mainStorageManager.jobCheckIn(each_job, StorageHolder<RayBundleColumn*>::Status::initialized);
	}

	return 0;
}

extern void SingleTest();

void testbenchGPU()
{
	//test pipeline
	
	OpticalConfigManager();
	//SingleTest();
	
	ColumnCreator2(vec3<MYFLOATTYPE>(0,0,500));
	ColumnCreator2(vec3<MYFLOATTYPE>(-20,0,280));
	ColumnCreator2(vec3<MYFLOATTYPE>(30,30,200));

	KernelLauncher();
	KernelLauncher2();

	RayBundleColumn* pcolumn = nullptr;
	while (mainStorageManager.takeOne(pcolumn, StorageHolder<RayBundleColumn*>::Status::completed1, 555))
	{
		mainStorageManager.pleaseDelete(pcolumn);
	}
	

	/*
	extern void testRenderer();
	testRenderer();
	*/
	//test single feature
	/*
	TriangleRendererJob testjob;
	testjob.preLaunchPreparation();
	*/
}

bool constructSurface(mysurface<MYFLOATTYPE>*& p_surface, unsigned short int surfaceType, vec3<MYFLOATTYPE> vertexPos, MYFLOATTYPE R, MYFLOATTYPE diam, unsigned short int side, MYFLOATTYPE n1, MYFLOATTYPE n2, MYFLOATTYPE K, unsigned short int apodization, std::string customApoPath, point2D<MYFLOATTYPE> tiptilt)
{
	//NOTE: tip/tilt has not been taken into account
	
	R = abs(R);

	vec3<MYFLOATTYPE> center(0, 0, 0);
	
	bool antiParallel = true;

	mysurface<MYFLOATTYPE>::SurfaceTypes type;
	switch (surfaceType)
	{
	case MF_REFRACTIVE:
		type = mysurface<MYFLOATTYPE>::SurfaceTypes::refractive;
		break;
	case MF_IMAGE:
		type = mysurface<MYFLOATTYPE>::SurfaceTypes::image;
		break;
	case MF_STOP:
		type = mysurface<MYFLOATTYPE>::SurfaceTypes::stop;
		n2 = FP_INFINITE;
		break;
	default:
		return false;
		break;
	}

	//apodization translator, incase somebody mess things up
	unsigned short int translatedApo = APD_UNIFORM;
	float* p_customApoData = nullptr;
	int customApoDataSize = 0;
	switch (apodization)
	{
	case PI_APD_BARTLETT:
		translatedApo = APD_BARTLETT;
		break;
	case PI_APD_CUSTOM:
		if (importCustomApo(p_customApoData, customApoDataSize, customApoPath))
		{
			translatedApo = APD_CUSTOM;
		}
		else
		{
			translatedApo = APD_UNIFORM; //fall back to uniform when cannot import custom apo data
		}
		break;
	case PI_APD_UNIFORM:
	default:
		translatedApo = APD_UNIFORM;
		break;
	}

	//if (K == 1.0 || side == MF_FLAT) //spherical or flat surface
	//{
	//	MYFLOATTYPE ABC = 1;
	//	MYFLOATTYPE I = 0;
	//	MYFLOATTYPE J = -R * R;
	//	vec3<MYFLOATTYPE> primaryGeoAxis(0, 0, R);

	//	switch (side)
	//	{
	//	case MF_CONVEX:
	//		center = vertexPos - primaryGeoAxis;
	//		antiParallel = true;
	//		break;
	//	case MF_CONCAVE:
	//		center = vertexPos + primaryGeoAxis;
	//		antiParallel = false;
	//		break;
	//	case MF_FLAT:
	//		center = vertexPos;
	//		ABC = 0;
	//		I = 1; 
	//		J = 0;
	//		break;
	//	default:
	//		return false;
	//		break;
	//	}

	//	p_surface = new quadricsurface<MYFLOATTYPE>(type,
	//		quadricparam<MYFLOATTYPE>(ABC, ABC, ABC, 0, 0, 0, 0, 0, I, J), n1, n2, center, diam,
	//		antiParallel, tiptilt);
	//	p_surface->apodizationType = translatedApo;
	//	if (p_surface->apodizationType == APD_CUSTOM)
	//	{
	//		p_surface->add_data(reinterpret_cast<char*>(p_customApoData), customApoDataSize);
	//		if (p_customApoData != nullptr)
	//		{
	//			delete[] p_customApoData;
	//			p_customApoData = nullptr;
	//		}
	//	}
	//}
	if (side == MF_FLAT) // flat surface
	{
		MYFLOATTYPE ABC = 0;
		MYFLOATTYPE I = 1;
		MYFLOATTYPE J = 0;
		center = vertexPos;

		p_surface = new quadricsurface<MYFLOATTYPE>(type,
			quadricparam<MYFLOATTYPE>(ABC, ABC, ABC, 0, 0, 0, 0, 0, I, J), n1, n2, center, diam,
			antiParallel, tiptilt);
		p_surface->apodizationType = translatedApo;
		if (p_surface->apodizationType == APD_CUSTOM)
		{
			p_surface->add_data(reinterpret_cast<char*>(p_customApoData), customApoDataSize);
			if (p_customApoData != nullptr)
			{
				delete[] p_customApoData;
				p_customApoData = nullptr;
			}
		}
	}
	else
	{
		center = vertexPos;
		MYFLOATTYPE I = 0.0;

		switch (side)
		{
		case MF_CONVEX:
			I = 2.0*R;
			antiParallel = true; //should check this again
			break;
		case MF_CONCAVE:
			I = -2.0*R;
			antiParallel = false;
			break;
		default:
			return false;
			break;
		}

		p_surface = new quadricsurface<MYFLOATTYPE>(type,
			quadricparam<MYFLOATTYPE>(1.0, 1.0, (1.0+K), 0, 0, 0, 0, 0, I, 0), n1, n2, center, diam,
			antiParallel, tiptilt);
		p_surface->apodizationType = translatedApo;
		if (p_surface->apodizationType == APD_CUSTOM)
		{
			p_surface->add_data(reinterpret_cast<char*>(p_customApoData), customApoDataSize);
			if (p_customApoData != nullptr)
			{
				delete[] p_customApoData;
				p_customApoData = nullptr;
			}
		}
	}
	return true;
}
