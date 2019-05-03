#include "PerformanceTester.cuh"
#include "mycommon.cuh"
#include "CommonClasses.h"
#include "vec3.cuh"
#include "class_hierarchy.cuh"
#include "StorageManager.cuh"
#include "ManagerFunctions.cuh"
#include "../ConsoleApplication/src/ImageFacilities.h"
#include "Auxiliaries.cuh"

#include "ProgramInterface.h"

#include <vector>
#include <algorithm>
#include <list>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstdlib>

//forward declarations
__global__ void quadrictracer_bare(quadricsurface<MYFLOATTYPE>* pquad_in);
void quadrictracer_CPU();

//external global variables
extern StorageManager mainStorageManager;
extern float activeWavelength;
extern int PI_ThreadsPerKernelLaunch;
extern int PI_traceJobSize;
extern int PI_renderJobSize;
extern int PI_maxParallelThread;



int testTracingGPU(int repeat)
{
	int kernelToLaunch = repeat;

	int threadsToLaunch = PI_ThreadsPerKernelLaunch;
	int blocksToLaunch = (kernelToLaunch + threadsToLaunch - 1) / (threadsToLaunch);

	auto test_surface = quadricsurface<MYFLOATTYPE>(mysurface<MYFLOATTYPE>::SurfaceTypes::refractive, quadricparam<MYFLOATTYPE>(1, 1, 1, 0, 0, 0, 0, 0, 0, -1));
	test_surface.copytosibling();

	quadricsurface<MYFLOATTYPE>* pquad = static_cast<quadricsurface<MYFLOATTYPE>*>(test_surface.d_sibling);


	quadrictracer_bare <<<blocksToLaunch, threadsToLaunch>>> (pquad);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__);
		fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();

	return 0;
}

int testRenderingGPU(int repeat)
{
	return 0;
}

int testTracingCPU(int repeat)
{
	for (int i = 0; i < repeat; i++)
	{
		quadrictracer_CPU();
	}
	return 0;
}

int testRenderingCPU(int repeat)
{
	return 0;
}

__global__ void quadrictracer_bare(quadricsurface<MYFLOATTYPE>* pquad_in)
{
	//testing
	//int debugID = 190;
	//adapt this kernel to the new structure

	//get the index
	//int ID = (blockIdx.x*blockDim.x) + threadIdx.x;
	//if (ID >= kernelparams.otherparams[0] * kernelparams.otherparams[1]) return;

	//int bundleID = ID / kernelparams.otherparams[1];
	//int rayID = ID - bundleID * kernelparams.otherparams[1];
	//get the block index, clamp to total number of bundles
	//int blockidx = (blockIdx.x < kernelparams.otherparams[0]) ? blockIdx.x : kernelparams.otherparams[0]; //number of bundles

	//grab the correct in and out ray bundles
	//raybundle<MYFLOATTYPE>* inbundle = (kernelparams.d_inbundles)[blockidx];
	//raybundle<MYFLOATTYPE>* outbundle = (kernelparams.d_outbundles)[blockidx];
	//raybundle<MYFLOATTYPE>* inbundle = (kernelparams.d_inbundles)[bundleID];
	//raybundle<MYFLOATTYPE>* outbundle = (kernelparams.d_outbundles)[bundleID];

	//get the thread index, clamp to the number of rays in this bundle
	//int idx = (threadIdx.x < inbundle->size) ? threadIdx.x : inbundle->size; //number of rays in a bundle
	//int idx = (rayID < inbundle->size) ? rayID : (inbundle->size - 1); //number of rays in a bundle

	//grab the correct ray of this thread
	//raysegment<MYFLOATTYPE> before = (inbundle->prays)[idx];
	raysegment<MYFLOATTYPE> before(vec3<MYFLOATTYPE>(0, 0, 1), vec3<MYFLOATTYPE>(0, 0, -1));

	//quit if ray is deactivated
	//if (before.status != (raysegment<MYFLOATTYPE>::Status::active))
	//{
	//	(outbundle->prays)[idx] = (inbundle->prays)[idx];
	//	return;
	//}

	//load the surface
	//quadricsurface<MYFLOATTYPE>& quadric = *pquad;
	quadricsurface<MYFLOATTYPE>* pquad = pquad_in;

	//test case
	//auto pquad = new quadricsurface<MYFLOATTYPE>(mysurface<MYFLOATTYPE>::SurfaceTypes::refractive, quadricparam<MYFLOATTYPE>(1, 1, 1, 0, 0, 0, 0, 0, 0, -1));

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
		beforedelta = 2.0 * A*d1*p1 + 2.0 * B*d2*p2 + 2.0 * C*d3*p3 + D * d1*p2 + D * d2*p1 +
			E * d1*p3 + E * d3*p1 + F * d2*p3 + F * d3*p2 + G * d1 + H * d2 + K * d3;
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
		double npc_rho = (double)(sqrt(at.pos.x*at.pos.x + at.pos.y*at.pos.y) / (pquad->diameter / 2));
		double npc_phi = 0.0;
		if (npc_rho != 0.0)
		{
			npc_phi = acos(at.pos.x / (npc_rho * (pquad->diameter / 2)));
			if (at.pos.y < 0)
				npc_phi = -npc_phi;
		}
		//check the multiplication factor from the aperture function
		//apodizationFunction_t apdToUse = apdFunctionLookUp(pquad->apodizationType);
		float apertureFactor = 1.0f;
			//apdToUse(npc_rho, npc_phi, pquad->data_size, pquad->p_data);
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

			//(outbundle->prays)[idx] = after;

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
			//(outbundle->prays)[idx] = at;
			goto final;
		}
	}
	//else there is no intersection, deactivate the ray
	else
		goto deactivate_ray;

deactivate_ray:
	{
		// TO DO: write out ray status, copy input to output
		//(outbundle->prays)[idx] = (inbundle->prays)[idx];
		//(outbundle->prays)[idx].status = raysegment<MYFLOATTYPE>::Status::deactivated;
		//(outbundle->prays)[idx].intensity = 0.0f;
	};

	//clean up the test case
	final :
}

void quadrictracer_CPU()
{
	//testing
	//int debugID = 190;
	//adapt this kernel to the new structure

	//get the index
	//int ID = (blockIdx.x*blockDim.x) + threadIdx.x;
	//if (ID >= kernelparams.otherparams[0] * kernelparams.otherparams[1]) return;

	//int bundleID = ID / kernelparams.otherparams[1];
	//int rayID = ID - bundleID * kernelparams.otherparams[1];
	//get the block index, clamp to total number of bundles
	//int blockidx = (blockIdx.x < kernelparams.otherparams[0]) ? blockIdx.x : kernelparams.otherparams[0]; //number of bundles

	//grab the correct in and out ray bundles
	//raybundle<MYFLOATTYPE>* inbundle = (kernelparams.d_inbundles)[blockidx];
	//raybundle<MYFLOATTYPE>* outbundle = (kernelparams.d_outbundles)[blockidx];
	//raybundle<MYFLOATTYPE>* inbundle = (kernelparams.d_inbundles)[bundleID];
	//raybundle<MYFLOATTYPE>* outbundle = (kernelparams.d_outbundles)[bundleID];

	//get the thread index, clamp to the number of rays in this bundle
	//int idx = (threadIdx.x < inbundle->size) ? threadIdx.x : inbundle->size; //number of rays in a bundle
	//int idx = (rayID < inbundle->size) ? rayID : (inbundle->size - 1); //number of rays in a bundle

	//grab the correct ray of this thread
	//raysegment<MYFLOATTYPE> before = (inbundle->prays)[idx];
	raysegment<MYFLOATTYPE> before(vec3<MYFLOATTYPE>(0, 0, 1), vec3<MYFLOATTYPE>(0, 0, -1));

	//quit if ray is deactivated
	//if (before.status != (raysegment<MYFLOATTYPE>::Status::active))
	//{
	//	(outbundle->prays)[idx] = (inbundle->prays)[idx];
	//	return;
	//}

	//load the surface
	//quadricsurface<MYFLOATTYPE>& quadric = *pquad;
	//quadricsurface<MYFLOATTYPE>* pquad = kernelparams.pquad;

	//test case
	auto pquad = new quadricsurface<MYFLOATTYPE>(mysurface<MYFLOATTYPE>::SurfaceTypes::refractive, quadricparam<MYFLOATTYPE>(1, 1, 1, 0, 0, 0, 0, 0, 0, -1));

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
		beforedelta = 2.0 * A*d1*p1 + 2.0 * B*d2*p2 + 2.0 * C*d3*p3 + D * d1*p2 + D * d2*p1 +
			E * d1*p3 + E * d3*p1 + F * d2*p3 + F * d3*p2 + G * d1 + H * d2 + K * d3;
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
		double npc_rho = (double)(sqrt(at.pos.x*at.pos.x + at.pos.y*at.pos.y) / (pquad->diameter / 2));
		double npc_phi = 0.0;
		if (npc_rho != 0.0)
		{
			npc_phi = acos(at.pos.x / (npc_rho * (pquad->diameter / 2)));
			if (at.pos.y < 0)
				npc_phi = -npc_phi;
		}
		//check the multiplication factor from the aperture function
		//apodizationFunction_t apdToUse = apdFunctionLookUp(pquad->apodizationType);
		float apertureFactor = 1.0f;
		//apdToUse(npc_rho, npc_phi, pquad->data_size, pquad->p_data);
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

			//(outbundle->prays)[idx] = after;

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
			//(outbundle->prays)[idx] = at;
			goto final;
		}
	}
	//else there is no intersection, deactivate the ray
	else
		goto deactivate_ray;

deactivate_ray:
	{
		// TO DO: write out ray status, copy input to output
		//(outbundle->prays)[idx] = (inbundle->prays)[idx];
		//(outbundle->prays)[idx].status = raysegment<MYFLOATTYPE>::Status::deactivated;
		//(outbundle->prays)[idx].intensity = 0.0f;
	};

	//clean up the test case
	final :
}