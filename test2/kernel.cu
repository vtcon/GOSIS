#include "mycommon.cuh"
#include "CommonClasses.h"
#include "vec3.cuh"
#include "class_hierarchy.cuh"
#include "StorageManager.cuh"
#include "ManagerFunctions.cuh"

/**************************************************************************************************/
/****************                   EXPERIMENTAL ZONE STARTS                         **************/
/**************************************************************************************************/
#ifdef nothing

class Test3
{
public:
	Test3()
	{
		printf("Hello from kernel.cu");
	}
};

auto ptest3 = new Test3;

void test3function()
{
	ptest3;
}


#endif

typedef float(*apodizationFunction_t)(double, double, int, char*);

__device__ float apdUniform(double rho, double phi, int datasize = 0, char* p_data = nullptr)
{
	if (float(rho) <= 1.0f)
		return 1.0f;
	else
		return 0.0f;
}

__device__ float apdBartlett(double rho, double phi, int datasize = 0, char* p_data = nullptr)
{
	if (float(rho) <= 1.0f)
	{
		return (1.0f - float(rho));
	}
	else
		return 0.0f;
}

__device__ float apdCustom(double rho, double phi, int datasize = 0, char* p_data = nullptr)
{
	if (rho > 1.0 || datasize == 0 || p_data == nullptr)
	{
		return 0.0f;
	}
	else
	{
		float retVal = 0.0f;
		float *p_reader = reinterpret_cast<float*>(p_data);
		int rowCount = static_cast<int>(p_reader[0]);
		int colCount = static_cast<int>(p_reader[1]);
		float dx = float(rho * cos(phi));
		float dy = float(-rho * sin(phi)); //the minus is due to the fact that openCV reads the y axis from TOP TO BOTTOM
		int x = (dx + 1) / 2 * colCount;
		x = (x >= colCount) ? colCount - 1 : x;
		x = (x < 0) ? 0 : x;
		int y = (dy + 1)*rowCount / 2;
		y = (y >= rowCount) ? rowCount - 1 : y;
		y = (y < 0) ? 0 : y;
		int readIndex = y * colCount + x + 2;
		retVal = p_reader[readIndex];
		return retVal;
	}
}

__device__ apodizationFunction_t testApd = apdUniform;

__device__ apodizationFunction_t apdFunctionLookUp(unsigned short int apdcode)
{
	switch (apdcode)
	{
	case APD_BARTLETT:
		return apdBartlett;
	case APD_CUSTOM:
		return apdCustom;
	case APD_UNIFORM:
	default:
		return apdUniform;
	}
}

/**************************************************************************************************/
/*******************                EXPERIMENTAL ZONE ENDS                 ************************/
/**************************************************************************************************/


/**************************************************************************************************/
/****************                       GLOBAL VARIABLES STARTS                      **************/
/**************************************************************************************************/


/**************************************************************************************************/
/*******************                 GLOBAL VARIABLES ENDS                 ************************/
/**************************************************************************************************/

//forward declaration


template<typename T = float>
__global__ void printoutdevicedatakernel(mysurface<T>* testobject)
{
	printf(testobject->p_data);
}





//quadric tracer kernel, each block handles one bundle, each thread handles one ray
__global__ void quadrictracer(
	//raybundle<T>** d_inbundles
	//, raybundle<T>** d_outbundles
	//, quadricsurface<T>* pquad
	//, 
	QuadricTracerKernelLaunchParams kernelparams
	)
{
	//testing
	//int debugID = 190;
	//adapt this kernel to the new structure

	//get the index
	int ID = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (ID >= kernelparams.otherparams[0] * kernelparams.otherparams[1]) return;

	int bundleID = ID / kernelparams.otherparams[1];
	int rayID = ID-bundleID*kernelparams.otherparams[1];
	//get the block index, clamp to total number of bundles
	//int blockidx = (blockIdx.x < kernelparams.otherparams[0]) ? blockIdx.x : kernelparams.otherparams[0]; //number of bundles
	
	//grab the correct in and out ray bundles
	//raybundle<MYFLOATTYPE>* inbundle = (kernelparams.d_inbundles)[blockidx];
	//raybundle<MYFLOATTYPE>* outbundle = (kernelparams.d_outbundles)[blockidx];
	raybundle<MYFLOATTYPE>* inbundle = (kernelparams.d_inbundles)[bundleID];
	raybundle<MYFLOATTYPE>* outbundle = (kernelparams.d_outbundles)[bundleID];

	//get the thread index, clamp to the number of rays in this bundle
	//int idx = (threadIdx.x < inbundle->size) ? threadIdx.x : inbundle->size; //number of rays in a bundle
	int idx = (rayID < inbundle->size) ? rayID : (inbundle->size - 1); //number of rays in a bundle

	//grab the correct ray of this thread
	//raysegment<MYFLOATTYPE> before = (inbundle->prays)[idx];
	raysegment<MYFLOATTYPE> before = (inbundle->prays)[idx];

	//quit if ray is deactivated
	if (before.status != (raysegment<MYFLOATTYPE>::Status::active))
	{
		(outbundle->prays)[idx] = (inbundle->prays)[idx];
		return;
	}

	//load the surface
	//quadricsurface<MYFLOATTYPE>& quadric = *pquad;
	quadricsurface<MYFLOATTYPE>* pquad = kernelparams.pquad;

	//test case
	//auto pquad = new quadricsurface<MYFLOATTYPE>(quadricparam<MYFLOATTYPE>(1, 1, 1, 0, 0, 0, 0, 0, 0, -1));

	// copy to the shared memory (at the time not possible)

	//coordinate transformation
	//todo: memory access violation here, check my surface copy to device
	//before = quadric.coordinate_transform(before);
	//before = pquad->coordinate_transform(before);
	before.pos = before.pos - pquad->pos;
	/*
	__shared__ raysegment<MYFLOATTYPE> loadedbundle[bundlesize];
	loadedbundle[idx] = *pray;
	auto before = loadedbundle[idx];
	*/

	//find intersection

	//define references, else it will look too muddy
	MYFLOATTYPE A = pquad->param.A,
		B = pquad->param.B,
		C = pquad->param.C,
		D = pquad->param.D,
		E = pquad->param.E,
		F = pquad->param.F,
		G = pquad->param.G,
		H = pquad->param.H,
		K = pquad->param.I, // in order not to mix with imaginary unit, due to the symbolic calculation in Maple
		J = pquad->param.J;
	MYFLOATTYPE p1 = before.pos.x,
		p2 = before.pos.y,
		p3 = before.pos.z,
		d1 = before.dir.x,
		d2 = before.dir.y,
		d3 = before.dir.z;
	MYFLOATTYPE t1 = 0.0, t2 = 0.0, deno = 0.0;
	deno = -2.0 * (A*d1*d1 + B * d2*d2 + C * d3*d3 + D * d1*d2 + E * d1*d3 + F * d2*d3);
	if (deno != 0)
	{
		MYFLOATTYPE delta = 0.0, beforedelta = 0.0;
		delta = 
			-4.0*A*B*d1*d1*p2*p2 + 8.0*A*B*d1*d2*p1*p2 - 4.0*A*B*d2*d2*p1*p1 - 4.0*A*C*d1*d1*p3*p3
			+ 8.0*A*C*d1*d3*p1*p3 - 4.0*A*C*d3*d3*p1*p1 - 4.0*A*F*d1*d1*p2*p3 + 4.0*A*F*d1*d2*p1*p3
			+ 4.0*A*F*d1*d3*p1*p2 - 4.0*A*F*d2*d3*p1*p1 - 4.0*B*C*d2*d2*p3*p3 + 8.0*B*C*d2*d3*p2*p3
			- 4.0*B*C*d3*d3*p2*p2 + 4.0*B*E*d1*d2*p2*p3 - 4.0*B*E*d1*d3*p2*p2 - 4.0*B*E*d2*d2*p1*p3
			+ 4.0*B*E*d2*d3*p1*p2 - 4.0*C*D*d1*d2*p3*p3 + 4.0*C*D*d1*d3*p2*p3 + 4.0*C*D*d2*d3*p1*p3
			- 4.0*C*D*d3*d3*p1*p2 + 1.0*D*D*d1*d1*p2*p2 - 2.0*D*D*d1*d2*p1*p2 + 1.0*D*D*d2*d2*p1*p1
			+ 2.0*D*E*d1*d1*p2*p3 - 2.0*D*E*d1*d2*p1*p3 - 2.0*D*E*d1*d3*p1*p2 + 2.0*D*E*d2*d3*p1*p1
			- 2.0*D*F*d1*d2*p2*p3 + 2.0*D*F*d1*d3*p2*p2 + 2.0*D*F*d2*d2*p1*p3 - 2.0*D*F*d2*d3*p1*p2
			+ 1.0*E*E*d1*d1*p3*p3 - 2.0*E*E*d1*d3*p1*p3 + 1.0*E*E*d3*d3*p1*p1 + 2.0*E*F*d1*d2*p3*p3
			- 2.0*E*F*d1*d3*p2*p3 - 2.0*E*F*d2*d3*p1*p3 + 2.0*E*F*d3*d3*p1*p2 + 1.0*F*F*d2*d2*p3*p3
			- 2.0*F*F*d2*d3*p2*p3 + 1.0*F*F*d3*d3*p2*p2
			- 4.0*A*H*d1*d1*p2 + 4.0*A*H*d1*d2*p1 - 4.0*A*K*d1*d1*p3 + 4.0*A*K*d1*d3*p1 + 4.0*B*G*d1*d2*p2
			- 4.0*B*G*d2*d2*p1 - 4.0*B*K*d2*d2*p3 + 4.0*B*K*d2*d3*p2 + 4.0*C*G*d1*d3*p3 - 4.0*C*G*d3*d3*p1
			+ 4.0*C*H*d2*d3*p3 - 4.0*C*H*d3*d3*p2 + 2.0*D*G*d1*d1*p2 - 2.0*D*G*d1*d2*p1 - 2.0*D*H*d1*d2*p2
			+ 2.0*D*H*d2*d2*p1 - 4.0*D*K*d1*d2*p3 + 2.0*D*K*d1*d3*p2 + 2.0*D*K*d2*d3*p1 + 2.0*E*G*d1*d1*p3
			- 2.0*E*G*d1*d3*p1 + 2.0*E*H*d1*d2*p3 - 4.0*E*H*d1*d3*p2 + 2.0*E*H*d2*d3*p1 - 2.0*E*K*d1*d3*p3
			+ 2.0*E*K*d3*d3*p1 + 2.0*F*G*d1*d2*p3 + 2.0*F*G*d1*d3*p2 - 4.0*F*G*d2*d3*p1 + 2.0*F*H*d2*d2*p3
			- 2.0*F*H*d2*d3*p2 - 2.0*F*K*d2*d3*p3 + 2.0*F*K*d3*d3*p2
			- 4.0*A*J*d1*d1 - 4.0*B*J*d2*d2 - 4.0*C*J*d3*d3 - 4.0*D*J*d1*d2 - 4.0*E*J*d1*d3 - 4.0*F*J*d2*d3
			+ 1.0*G*G*d1*d1 + 2.0*G*H*d1*d2 + 2.0*G*K*d1*d3 + 1.0*H*H*d2*d2 + 2.0*H*K*d2*d3 + 1.0*K*K*d3*d3;
		beforedelta = 2.0 * A*d1*p1 + 2.0 * B*d2*p2 + 2.0 * C*d3*p3 + D*d1*p2 + D*d2*p1 +
			E*d1*p3 + E*d3*p1 + F*d2*p3 + F*d3*p2 + G*d1 + H*d2 + K*d3;
		t1 = (delta >= 0.0) ? (beforedelta + sqrt(delta)) / deno : INFINITY;
		t2 = (delta >= 0.0) ? (beforedelta - sqrt(delta)) / deno : INFINITY;
		//what to do if delta <0
		if (delta < 0) goto deactivate_ray;
	}
	else
	{
		MYFLOATTYPE num = 0.0, den = 0.0;
		num = -A * p1*p1 - B * p2*p2 - C * p3*p3 - D * p1*p2 - E * p1*p3 - F * p2*p3 - G * p1 - H * p2 - K * p3 - J;
		den = 2.0 * A*d1*p1 + 2.0 * B*d2*p2 + 2.0 * C*d3*p3 + D * d1*p2 + D * d2*p1 + E * d1*p3 + E * d3*p1
			+ F * d2*p3 + F * d3*p2 + G * d1 + H * d2 + K * d3;
		t1 = num / den;
		t2 = -INFINITY;
	}

	MYFLOATTYPE t = 0.0, otherT = 0.0;
	//pick the nearest positive intersection
	if (t1 >= 0.0 && t2 >= 0.0)
	{
		//t = (t1 < t2) ? t1 : t2;
		if (t1 <= t2)
		{
			t = t1;
			otherT = t2;
		}
		else
		{
			t = t2;
			otherT = t1;
		}
	}
	else if (t1 < 0.0 && t2 >= 0.0)
	{
		t = t2; otherT = INFINITY;
	}
	else if (t2 < 0.0 && t1 >= 0.0)
	{
		t = t1; otherT = INFINITY;
	}
	else
	{
		t = INFINITY; otherT = INFINITY;
	}

	// if there is an intersection
	if (t < INFINITY)
	{
		//first determine if the hit was on the right (convex/concave) side by examining the anti-parallelism of
		//...ray direction and surface normal
		bool acceptDirection = false;
		raysegment<MYFLOATTYPE> at;
		vec3<MYFLOATTYPE> surfnormal;
		MYFLOATTYPE ddotn;

		for (int run = 0; run < 2; run++)
		{
			at = raysegment<MYFLOATTYPE>(before.pos + t * before.dir, before.dir);
			
			//printf("surface %d, check equation: %f\n", kernelparams.otherparams[2], at.pos.x*at.pos.x + at.pos.y*at.pos.y + at.pos.z*at.pos.z);

			//attenuation could be implemented here
			at.intensity = before.intensity;

			MYFLOATTYPE &x = at.pos.x,
				&y = at.pos.y,
				&z = at.pos.z;
			surfnormal = normalize(vec3<MYFLOATTYPE>(2.0 * A*x + D * y + E * z + G, 2.0 * B*y + D * x + F * z + H, 2.0 * C*z + E * x + F * y + K));

			ddotn = dot(at.dir, surfnormal);

			if ((pquad->antiParallel == true && ddotn >= 0.0) ||
				(pquad->antiParallel == false && ddotn <= 0.0))
			{
				if (otherT == INFINITY)
				{
					break;
				}
				else
				{
					t = otherT;
					continue;
				}
			}
			acceptDirection = true;
			break;
		}
		if (!acceptDirection)
			goto deactivate_ray;

		ddotn = (ddotn < 0) ? ddotn : -ddotn; // so that the surface normal and ray are in opposite direction

		if (pquad->antiParallel == false)
			surfnormal = -surfnormal;

		//is the intersection within hit box ? if not, then deactivate the ray
		//if ((at.pos.x*at.pos.x + at.pos.y*at.pos.y) > (pquad->diameter*pquad->diameter / 4.0)) goto deactivate_ray;

		//calculate normalized pupil coordinate at the intersection
		double npc_rho = 0.0;
		npc_rho = sqrt(at.pos.x*at.pos.x + at.pos.y*at.pos.y);
		npc_rho = npc_rho * 2.0 / (pquad->diameter);
		double npc_phi = 0.0;
		if (npc_rho != 0.0)
		{
			npc_phi = acos(at.pos.x / (npc_rho * (pquad->diameter / 2)));
			if (at.pos.y < 0)
				npc_phi = -npc_phi;
		}
		

		//check the multiplication factor from the aperture function
		apodizationFunction_t apdToUse = apdFunctionLookUp(pquad->apodizationType);
		float apertureFactor = apdToUse(npc_rho, npc_phi, pquad->data_size, pquad->p_data);
		if (apertureFactor == 0.0f) goto deactivate_ray;

		// if it is a refractive surface, do refractive ray transfer
		if (pquad->type == mysurface<MYFLOATTYPE>::SurfaceTypes::refractive)
		{
			//refractive surface transfer
			auto after = raysegment<MYFLOATTYPE>(at.pos, at.dir);
			
			MYFLOATTYPE factor1 = 0.0;
			/*
			auto d1 = pquad->n1*pquad->n1;
			auto d2 = pquad->n2*pquad->n2;
			auto d3 = (1 - ddotn * ddotn);
			auto d4 = (pquad->n1*pquad->n1) / (pquad->n2*pquad->n2);
			auto d5 = (pquad->n1*pquad->n1) / (pquad->n2*pquad->n2)*(1 - ddotn * ddotn);
			*/
			factor1 = 1.0 - (pquad->n1*pquad->n1) / (pquad->n2*pquad->n2)*(1.0 - ddotn * ddotn);
			/*
			if (rayID == debugID)
				printf("d1 = %f d2 = %f d3 = %f d4 = %f d5 = %f factor1 = %f\n", d1, d2, d3, d4, d5, factor1);
			*/
			if (factor1 < 0.0)
			{
#ifdef _DEBUGMODE2
				printf("TIR at ray %d, bundle %d, surface %d\n", rayID, bundleID, kernelparams.otherparams[2]);
#endif
				goto deactivate_ray;
			}

			after.dir = (pquad->n1)*(at.dir - surfnormal * ddotn) / (pquad->n2) - surfnormal * (MYFLOATTYPE)sqrt(factor1);
			after.dir = normalize(after.dir);
			//light attenuation due to apodization
			after.intensity = at.intensity*apertureFactor;

			//coordinate detransformation, write out result
			after.pos = after.pos + pquad->pos;

			(outbundle->prays)[idx] = after;

			//printout calculation
			/*
			printf("delta = %f ,beforedelta = %f ,deno = %f \n", delta, beforedelta, deno);
			printf("t1 = %f ,t2 = %f ,t = %f\n", t1, t2, t);
			printf("%d at t = %f ,pos = (%f,%f,%f), surfnormal (%f,%f,%f), factor1 = %f, at dir (%f,%f,%f), after dir (%f,%f,%f)\n",
				idx, t, at.pos.x, at.pos.y, at.pos.z,
				surfnormal.x, surfnormal.y, surfnormal.z, factor1,
				at.dir.x, at.dir.y, at.dir.z,
				after.dir.x, after.dir.y, after.dir.z);
			*/
			goto final;
		}
		// else if it is an image surface
		else if (pquad->type == mysurface<MYFLOATTYPE>::SurfaceTypes::image)
		{
			//coordinate detranformation of at and write out result
			//at = pquad->coordinate_detransform(at);
			at.pos = at.pos + pquad->pos;
			//at.status = 2;
			at.status = raysegment<MYFLOATTYPE>::Status::finished;
			(outbundle->prays)[idx] = at;
			goto final;
		}
	}
	//else there is no intersection, deactivate the ray
	else
		goto deactivate_ray;

deactivate_ray:
	{
		// TO DO: write out ray status, copy input to output
		(outbundle->prays)[idx] = (inbundle->prays)[idx];
		(outbundle->prays)[idx].status = raysegment<MYFLOATTYPE>::Status::deactivated;
		(outbundle->prays)[idx].intensity = 0.0f;
	};

	//clean up the test case
	final :
}



//deprecated
#ifdef nothing
int GPUmanager(int argc = 0, char** argv = nullptr)
{
	LOG1("this is main program");

	//create event for timing: to GPU manager
	cudaEvent_t start, stop;
	CUDARUN(cudaEventCreate(&start));
	CUDARUN(cudaEventCreate(&stop));

	//load the optical configuration
	OpticalConfigManager(0,nullptr);
	OpticalConfig* thisOpticalConfig = nullptr;
	mainStorageManager.infoCheckOut(thisOpticalConfig);


	//test the function column creator
	ColumnCreator(0, nullptr);
	RayBundleColumn* pthiscolumn = nullptr;
	mainStorageManager.takeOne(pthiscolumn);


	int numofsurfaces = thisOpticalConfig->numofsurfaces;

	//TODO: DEBUG MEMORY USAGE OF INITIALIZER
	//creating an array of ray bundles: in tracing job manager, data from object and image manager 
	LOG1("[main]creating ray bundles\n");
	raybundle<MYFLOATTYPE>* bundles = new raybundle<MYFLOATTYPE>[numofsurfaces + 1];

	//initialize the first bundle
	LOG1("[main]initialize 1st bundle\n");
	//bundles[0].init_2D_dualpolar(vec3<MYFLOATTYPE>(0, 0, 20), -0.5, 0.5, -0.5, 0.5, 0.25);
	bundles[0].init_1D_parallel(vec3<double>(0, 0, -1), 5, 80);
	int rays_per_bundle = bundles[0].size;

	//initializes other bundles.........not really needed
	/*
	for (int i = 1; i < numofsurfaces + 1; i++)
	{
		bundles[i] = raybundle<MYFLOATTYPE>(rays_per_bundle);
	}
	*/

	//create 2 bundles to pass in and out the kernel: also in tracing job manager
	LOG1("[main]creating 2 siblings bundles\n");
	raybundle<MYFLOATTYPE> h_inbundle = bundles[0];
	raybundle<MYFLOATTYPE> h_outbundle = bundles[0];
	h_inbundle.copytosibling();
	h_outbundle.copytosibling();
	//bundles[0].copytosibling();
	//bundles[numofsurfaces].copytosibling();

	//start timing 
	CUDARUN(cudaEventRecord(start, 0));

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
	//thisparam.otherparams[1] = rays_per_bundle;

	/*
	typedef void(*KernelFunctionType)(QuadricTracerKernelLaunchParams);
	typedef void(*KernelType)(KernelLaunchParams);

	KernelFunctionType thiskernel = quadrictracer;
	*/

	// launch kernel, copy result out, swap memory: goal, only pass one object to kernel
	// while (job.state!= finished) { kernel launch; job.update();}
	for (int i = 0; i < numofsurfaces; i++)
	{
		LOG1("[main]kernel launch \n");

		//create an object for param: should already be done in the job manager
		
		thisparam.otherparams[2] = i;
		thisparam.pquad = static_cast<quadricsurface<MYFLOATTYPE>*>(((thisOpticalConfig->surfaces)[i])->d_sibling);

		quadrictracer<<<job_size, rays_per_bundle>>>(
			//d_injob, 
			//d_outjob, 
			//static_cast<quadricsurface<MYFLOATTYPE>*>(surfaces[i]->d_sibling), 
			thisparam);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__);
			fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		}
		
		//all the following should be in tracing job manager
		//TODO: should be a for loop for a job with many bundles
		cudaDeviceSynchronize();
		LOG1("[main]copy sibling out");
		bundles[i + 1] = (i % 2 == 0) ? h_outbundle.copyfromsibling() : h_inbundle.copyfromsibling();
		
		swap(thisparam.d_inbundles, thisparam.d_outbundles);
	}

	//kernel finished, stop timing, print out elapsed time: in gpu manager
	CUDARUN(cudaEventRecord(stop, 0));
	CUDARUN(cudaEventSynchronize(stop));
	float elapsedtime;
	CUDARUN(cudaEventElapsedTime(&elapsedtime, start, stop));
	LOG2("kernel run time: " << elapsedtime << " ms\n");

	//writing results out: could be in tracing job manager?
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
				if ((bundles[j-1].prays)[i].status != raysegment<MYFLOATTYPE>::Status::deactivated)
					LOG2(" " << (bundles[j].prays)[i] << " done")
					break;
			}
		}
		LOG2("\n");
	}

	//destroy cuda timing events
	CUDARUN(cudaEventDestroy(start));
	CUDARUN(cudaEventDestroy(stop));

	// free device heap momory now automatically when object goes out of scale
	// free host heap memory STILL NEEDED

	return 0;
}
#endif
