#include "ManagerFunctions.cuh"
#include "StorageManager.cuh"
#include "../ConsoleApplication/src/ImageFacilities.h"

#include <vector>
#include <algorithm>

//global variables

//as the storage lies in this project, the regulator function should also be here
StorageManager mainStorageManager;



// external definitions:

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

	int wanted_job_size = 3; //settable from outside
	int job_size = 0; // real size of a batch depends on how many columns are left in the Storage
	int numofsurfaces = 0;
	int wavelength = 0;

	RayBundleColumn** pcolumns = nullptr;
	OpticalConfig* thisOpticalConfig = nullptr;
	raybundle<MYFLOATTYPE>* b_inbundles = nullptr;
	raybundle<MYFLOATTYPE>* b_outbundles = nullptr;

	int currentSurfaceCount = 0;

	dim3 blocksToLaunch = 0;
	dim3 threadsToLaunch = 0;

	typedef StorageHolder<RayBundleColumn*>::Status columnStatus;

	QuadricTracerJob(int _wanted_job_size = 3) :wanted_job_size(_wanted_job_size)
	{
		pcolumns = new RayBundleColumn*[wanted_job_size];

		//BIG QUESTION: where does the wavelength comes from?
		wavelength = 555;

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
		blocksToLaunch = job_size;
		int maxBundleSize = 0;
		for (int i = 0; i < job_size; i++)
		{
			int temp = (*pcolumns[i])[0].size;
			if (maxBundleSize < temp)
			{
				maxBundleSize = temp;
			}
		}
		threadsToLaunch = maxBundleSize;
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
	SimpleRetinaDescriptor* p_retinaDescriptor;
	RetinaImageChannel* p_rawChannel;

	int wanted_job_size = 3;
	int job_size = 0; 
	int wavelength = 0;
	int m_triangleCount = 0;
	
	mutable int m_launchCount = 0; 

	RayBundleColumn** pcolumns = nullptr;
	typedef StorageHolder<RayBundleColumn*>::Status columnStatus;

	TriangleRendererJob(int _wanted_job_size = 3) :wanted_job_size(_wanted_job_size)
	{
		pcolumns = new RayBundleColumn*[wanted_job_size];

		//BIG QUESTION: where does the wavelength comes from?
		wavelength = 555;

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

		for (int i = 0; i < job_size; i++)
		{
			auto lastindex = (pcolumns[i])->numofsurfaces; //two code lines, make it easier to debug
			RenderingTrianglesCreator((*pcolumns[i])[lastindex]);
		}
		
		transferMeshToDevice();

		//TODO: add these to the storage manager
		//data setup
		p_retinaDescriptor = new SimpleRetinaDescriptor(0.167, 10);
		p_rawChannel = new RetinaImageChannel(*p_retinaDescriptor);

		//the following line should not be here
		p_rawChannel->createSibling();

		//setup the kernel launch params
		m_kernelLaunchParams.retinaDescriptorIn = *p_retinaDescriptor;
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
		int threadsToLaunch = 16;
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

		//data copy out
		p_rawChannel->copyFromSibling();
		quickDisplay<MYFLOATTYPE>(p_rawChannel->hp_raw, p_rawChannel->m_dimension.y, p_rawChannel->m_dimension.x);
		
		//these two lines shouldn't be here, too
		p_rawChannel->deleteSibling();
		p_rawChannel->deleteHostImage();

		//TODO: copy to the output image

		//delete the ray columns and free the resources
		CUDARUN(cudaFree(dp_triangles));
		
	}

private:
	std::vector<PerKernelRenderingInput> v_triangles;
	PerKernelRenderingInput* hp_triangles = nullptr;
	PerKernelRenderingInput* dp_triangles = nullptr;

	int RenderingTrianglesCreator(const raybundle<MYFLOATTYPE>& thisbundle) //build up vector containing the mesh
	{
		//test data
		int arraySize = thisbundle.size;
		point2D<int>* inputArray = thisbundle.samplinggrid;


		//lambda for searching
		auto searchForPoint = [&](const point2D<int>& p) -> int {
			for (int i = 0; i < arraySize; i++)
			{
				if (p == inputArray[i])
				{
					//only finished rays should be rendered
					if ((thisbundle.prays)[i].status == raysegment<MYFLOATTYPE>::Status::finished)
					return i;
				}
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
			else if (results[1] != -1 && results[2] != -1 && results[3] != -1)
			{
				indexTriangles.push_back({ results[1], results[3],results[2] });
			}

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
		for (IndexTriangle indextriangle : indexTriangles)
		{
			v_triangles.push_back({ 
				(thisbundle.prays)[indextriangle.i1].pos,
				(thisbundle.prays)[indextriangle.i2].pos,
				(thisbundle.prays)[indextriangle.i3].pos,
				(thisbundle.prays)[indextriangle.i1].dir,
				(thisbundle.prays)[indextriangle.i2].dir,
				(thisbundle.prays)[indextriangle.i3].dir,
				});
		}

		//delete early
		for (int i = 0; i < job_size; i++)
		{
			mainStorageManager.pleaseDelete(pcolumns[i]);
		}
		delete[] pcolumns;

		return 0;
	}

	int transferMeshToDevice() //as the name suggests....
	{
		m_triangleCount = v_triangles.size();

		hp_triangles = new PerKernelRenderingInput[m_triangleCount];

		for (int i = 0; i < m_triangleCount; i++)
		{
			hp_triangles[i] = v_triangles[i];
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
	MYFLOATTYPE diam = 10;
	int numofsurfaces = 2;
	int wavelength1 = 555;

	//check out an opticalconfig as output
	OpticalConfig* newConfig = nullptr;
	mainStorageManager.jobCheckOut(newConfig, numofsurfaces, wavelength1);

	//construct the surfaces
	//(newConfig->surfaces)[0] = new quadricsurface<MYFLOATTYPE>(mysurface<MYFLOATTYPE>::SurfaceTypes::refractive, quadricparam<MYFLOATTYPE>(1, 1, 1, 0, 0, 0, 0, 0, 0, -400), 1, 1.5168, vec3<MYFLOATTYPE>(0, 0, 38.571), diam);
	bool output = constructSurface((newConfig->surfaces)[0], MF_REFRACTIVE, vec3<MYFLOATTYPE>(0, 0, 58.571), 20, diam, MF_CONVEX, 1.0, 1.5168);

	//(newConfig->surfaces)[1] = new quadricsurface<MYFLOATTYPE>(mysurface<MYFLOATTYPE>::SurfaceTypes::image, quadricparam<MYFLOATTYPE>(0, 0, 0, 0, 0, 0, 0, 0, 1, 0), 1, INFINITY, vec3<MYFLOATTYPE>(0, 0, 0), diam);
	output = constructSurface((newConfig->surfaces)[1], MF_IMAGE, vec3<MYFLOATTYPE>(0, 0, 0), FP_INFINITE, diam, MF_FLAT, 1.0, FP_INFINITE);
	

	//test adding data
	char* teststr = "hello";
	((newConfig->surfaces)[0])->add_data(teststr, 6);

	//copy to sibling on GPU side
	LOG1("[main]create sibling surfaces\n");
	newConfig->copytosiblings();

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
	cudaEvent_t start, stop;
	CUDARUN(cudaEventCreate(&start));
	CUDARUN(cudaEventCreate(&stop));

	job->preLaunchPreparation();

	//start timing 
	CUDARUN(cudaEventRecord(start, 0));

	while (job->goAhead())
	{
		job->kernelLaunch();
		job->update();
	}

	//kernel finished, stop timing, print out elapsed time: in gpu manager
	CUDARUN(cudaEventRecord(stop, 0));
	CUDARUN(cudaEventSynchronize(stop));
	float elapsedtime;
	CUDARUN(cudaEventElapsedTime(&elapsedtime, start, stop));
	LOG2("kernel run time: " << elapsedtime << " ms\n");
	CUDARUN(cudaEventDestroy(start));
	CUDARUN(cudaEventDestroy(stop));

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

	//create event for timing: to GPU manager
	cudaEvent_t start, stop;
	CUDARUN(cudaEventCreate(&start));
	CUDARUN(cudaEventCreate(&stop));

	job->preLaunchPreparation();

	//start timing 
	CUDARUN(cudaEventRecord(start, 0));

	while (job->goAhead())
	{
		job->kernelLaunch();
		job->update();
	}

	//kernel finished, stop timing, print out elapsed time: in gpu manager
	CUDARUN(cudaEventRecord(stop, 0));
	CUDARUN(cudaEventSynchronize(stop));
	float elapsedtime;
	CUDARUN(cudaEventElapsedTime(&elapsedtime, start, stop));
	LOG2("kernel run time: " << elapsedtime << " ms\n");
	CUDARUN(cudaEventDestroy(start));
	CUDARUN(cudaEventDestroy(stop));

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
	//get a point/field direction as input, in this example we use field direction
	vec3<MYFLOATTYPE> dir1(0, 0, -1);
	int wavelength1 = 555;
	//vec3<MYFLOATTYPE> dir12(0, 0, -0.9);
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
	//(*job)[0].init_1D_parallel(dir1, 5, 80);
	(*job)[0].init_2D_dualpolar(vec3<MYFLOATTYPE>(0, 0, 80), -3.0/180*MYPI, 3.0 / 180 * MYPI, -3.0 / 180 * MYPI, 3.0 / 180 * MYPI, 0.9 / 180 * MYPI);
	//mark the column as initialized after initializing it
	mainStorageManager.jobCheckIn(job, StorageHolder<RayBundleColumn*>::Status::initialized);

	return 0;
}


void testbenchGPU()
{
	//test pipeline
	
	OpticalConfigManager();

	ColumnCreator();
	//ColumnCreator();
	//ColumnCreator();

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

bool constructSurface(mysurface<MYFLOATTYPE>*& p_surface, unsigned short int surfaceType, vec3<MYFLOATTYPE> vertexPos, MYFLOATTYPE R, MYFLOATTYPE diam, unsigned short int side, MYFLOATTYPE n1, MYFLOATTYPE n2, MYFLOATTYPE K, point2D<MYFLOATTYPE> tiptilt)
{
	//NOTE: this is a very basic implementation, tip/tilt has not been taken into account
	R = abs(R);
	vec3<MYFLOATTYPE> center(0, 0, 0);
	vec3<MYFLOATTYPE> primaryGeoAxis(0, 0, R);
	mysurface<MYFLOATTYPE>::SurfaceTypes type;
	MYFLOATTYPE ABC = 1;
	MYFLOATTYPE I = 0;
	MYFLOATTYPE J = -R * R;
	bool antiParallel = true;
	if (K == 0) //spherical surface
	{
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
		switch (side)
		{
		case MF_CONVEX:
			center = vertexPos - primaryGeoAxis;
			antiParallel = true;
			break;
		case MF_CONCAVE:
			center = vertexPos + primaryGeoAxis;
			antiParallel = false;
			break;
		case MF_FLAT:
			center = vertexPos;
			ABC = 0;
			I = 1; 
			J = 0;
			break;
		default:
			return false;
			break;
		}

		p_surface = new quadricsurface<MYFLOATTYPE>(type,
			quadricparam<MYFLOATTYPE>(ABC, ABC, ABC, 0, 0, 0, 0, 0, I, J), n1, n2, center, diam,
			antiParallel, tiptilt);
	}
	else
	{
		//TODO: fill in this
	}
	return true;
}
