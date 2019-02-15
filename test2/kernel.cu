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

	//adapt this kernel to the new structure
	//get the block index, clamp to total number of bundles
	int blockidx = (blockIdx.x < kernelparams.otherparams[0]) ? blockIdx.x : kernelparams.otherparams[0]; //number of bundles
	
	//grab the correct in and out ray bundles
	raybundle<MYFLOATTYPE>* inbundle = (kernelparams.d_inbundles)[blockidx];
	raybundle<MYFLOATTYPE>* outbundle = (kernelparams.d_outbundles)[blockidx];

	//get the thread index, clamp to the number of rays in this bundle
	int idx = (threadIdx.x < inbundle->size) ? threadIdx.x : inbundle->size; //number of rays in a bundle

	//grab the correct ray of this thread
	raysegment<MYFLOATTYPE> before = (inbundle->prays)[idx];

	//quit if ray is deactivated
	if (before.status == (raysegment<MYFLOATTYPE>::Status::deactivated))
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
	MYFLOATTYPE t, t1, t2, otherT;
	MYFLOATTYPE deno = -2 * (A*d1*d1 + B * d2*d2 + C * d3*d3 + D * d1*d2 + E * d1*d3 + F * d2*d3);
	if (deno != 0)
	{
		MYFLOATTYPE delta = -4 * A*B*d1*d1*p2*p2 + 8 * A*B*d1*d2*p1*p2 - 4 * A*B*d2*d2*p1*p1
			- 4 * A*C*d1*d1*p3*p3 + 8 * A*C*d1*d3*p1*p3 - 4 * A*C*d3*d3*p1*p1 - 4 * A*F*d1*d1*p2*p3
			+ 4 * A*F*d1*d2*p1*p3 + 4 * A*F*d1*d3*p1*p2 - 4 * A*F*d2*d3*p1*p1 - 4 * B*C*d2*d2*p3*p3
			+ 8 * B*C*d2*d3*p2*p3 - 4 * B*C*d3*d2*p2*p2 + 4 * B*E*d1*d2*p2*p3 - 4 * B*E*d1*d3*p2*p2
			- 4 * B*E*d2*d2*p1*p3 + 4 * B*E*d2*d3*p1*p2 - 4 * C*D*d1*d2*p3*p3 + 4 * C*D*d1*d3*p2*p3
			+ 4 * C*D*d2*d3*p1*p3 - 4 * C*D*d3*d3*p1*p2 + D * D*d1*d1*p2*p2 - 2 * D*D*d1*d2*p1*p2
			+ D * D*d2*d2*p1*p1 + 2 * D*E*d1*d1*p2*p3 - 2 * D*E*d1*d2*p1*p3 - 2 * D*E*d1*d3*p1*p2
			+ 2 * D*E*d2*d3*p1*p1 - 2 * D*F*d1*d2*p2*p3 + 2 * D*F*d1*d3*p2*p2 + 2 * D*F*d2*d2*p1*p3
			- 2 * D*F*d2*d3*p1*p2 + E * E*d1*d1*p3*p3 - 2 * E*E*d1*d3*p1*p3 + E * E*d3*d3*p1*p1
			+ 2 * E*F*d1*d2*p3*p3 - 2 * E*F*d1*d3*p2*p3 - 2 * E*F*d2*d3*p1*p3 + 2 * E*F*d3*d3*p1*p2
			+ F * F*d2*d2*p3*p3 - 2 * F*F*d2*d3*p2*p3 + F * F*d3*d3*p2*p2 - 4 * A*H*d1*d1*p2
			+ 4 * A*H*d1*d2*p1 - 4 * A*K*d1*d1*p3 + 4 * A*K*d1*d3*p1 + 4 * B*G*d1*d2*p2 - 4 * B*G*d2*d2*p1
			- 4 * B*K*d2*d2*p3 + 4 * B*K*d2*d3*p2 + 4 * C*G*d1*d3*p3 - 4 * C*G*d3*d3*p1 + 4 * C*H*d2*d3*p3
			- 4 * C*H*d3*d3*p2 + 2 * D*G*d1*d1*p2 - 2 * D*G*d1*d2*p1 - 2 * D*H*d1*d2*p2
			+ 2 * D*H*d2*d2*p1 - 4 * D*K*d1*d2*p3 + 2 * D*K*d1*d3*p2 + 2 * D*K*d2*d3*p1 + 2 * E*G*d1*d1*p3
			- 2 * E*G*d1*d3*p1 + 2 * E*H*d1*d2*p3 - 4 * E*H*d1*d3*p2 + 2 * E*H*d2*d3*p1 - 2 * E*K*d1*d3*p3
			+ 2 * E*K*d3*d3*p1 + 2 * F*G*d1*d2*p3 + 2 * F*G*d1*d3*p2 - 4 * F*G*d2*d3*p1 + 2 * F*H*d2*d2*p3
			- 2 * F*H*d2*d3*p2 - 2 * F*K*d2*d3*p3 + 2 * F*K*d3*d3*p2 - 4 * A*J*d1*d1 - 4 * B*J*d2*d2
			- 4 * C*J*d3*d3 - 4 * D*J*d1*d2 - 4 * E*J*d1*d3 - 4 * F*J*d2*d3 + G * G*d1*d1 + 2 * G*H*d1*d2
			+ 2 * G*K*d1*d3 + H * H*d2*d2 + 2 * H*K*d2*d3 + K * K*d3*d3;
		MYFLOATTYPE beforedelta = 2 * A*d1*p1 + 2 * B*d2*p2 + 2 * C*d3*p3 + D * (d1*p2 + d2 * p1) +
			E * (d1*p3 + d3 * p1) + F * (d2*p3 + d3 * p2) + G * d1 + H * d2 + K * d3;
		t1 = (delta >= 0) ? (beforedelta + sqrt(delta)) / deno : INFINITY;
		t2 = (delta >= 0) ? (beforedelta - sqrt(delta)) / deno : INFINITY;
	}
	else
	{
		MYFLOATTYPE num = -A * p1*p1 - B * p2*p2 - C * p3*p3 - D * p1*p2 - E * p1*p3 - F * p2*p3 - G * p1 - H * p2 - K * p3 - J;
		MYFLOATTYPE den = 2 * A*d1*p1 + 2 * B*d2*p2 + 2 * C*d3*p3 + D * d1*p2 + D * d2*p1 + E * d1*p3 + E * d3*p1
			+ F * d2*p3 + F * d3*p2 + G * d1 + H * d2 + K * d3;
		t1 = num / den;
		t2 = -INFINITY;
	}

	//TODO: fix this, not always the nearest!
	//pick the nearest positive intersection
	if (t1 >= 0 && t2 >= 0)
	{
		t = (t1 < t2) ? t1 : t2;
	}
	else if (t1 < 0 && t2 >= 0)
	{
		t = t2; otherT = INFINITY;
	}
	else if (t2 < 0 && t1 >= 0)
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
		auto at = raysegment<MYFLOATTYPE>(before.pos + t * before.dir, before.dir);


		//is the intersection within hit box ? if not, then deactivate the ray
		if ((at.pos.x*at.pos.x + at.pos.y*at.pos.y) > (pquad->diameter*pquad->diameter / 4)) goto deactivate_ray;

		// if it is a refractive surface, do refractive ray transfer
		if (pquad->type == mysurface<MYFLOATTYPE>::SurfaceTypes::refractive)
		{
			//refractive surface transfer
			auto after = raysegment<MYFLOATTYPE>(at.pos, at.dir);
			MYFLOATTYPE &x = at.pos.x,
				&y = at.pos.y,
				&z = at.pos.z;
			auto surfnormal = normalize(vec3<MYFLOATTYPE>(2 * A*x + D * y + E * z + G, 2 * B*y + D * x + F * z + H, 2 * C*z + E * x + F * y + K));

			auto ddotn = dot(at.dir, surfnormal);
			ddotn = (ddotn < 0) ? ddotn : -ddotn; // so that the surface normal and ray are in opposite direction

			MYFLOATTYPE factor1 = 1 - pquad->n1*pquad->n1 / (pquad->n2*pquad->n2)
				*(1 - ddotn * ddotn);
			if (factor1 < 0)
			{
#ifdef _DEBUGMODE2
				printf("something is wrong with transfer refractive vectors");
#endif
				goto deactivate_ray;
			}

			after.dir = (pquad->n1)*(at.dir - surfnormal * ddotn) / (pquad->n2) - surfnormal * (MYFLOATTYPE)sqrt(factor1);

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

