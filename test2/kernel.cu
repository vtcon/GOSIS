#include "mycommon.cuh"
#include "vec3.cuh"

template<typename T = float>
__device__ T operator/(T lhs, T rhs)
{
	return (rhs == 0) ? (T)MYINFINITY : lhs / rhs;
}

template<typename pT>
__host__ __device__ inline void swap(pT& a, pT& b)
{
	pT temp = a;
	a = b;
	b = temp;
}


template<typename T = float>
class raysegment
{
public:
	vec3<T> pos, dir;
	int status = 1; // 1 is active, 0 is deactive, 2 is finised, more to come

	__host__ __device__ raysegment(const vec3<T>& pos = vec3<T>(0, 0, 0), const vec3<T>& dir = vec3<T>(0,0,-1)):
		pos(pos), dir(dir)
	{
		LOG1("ray segment created")
	}

	__host__ __device__ ~raysegment()
	{
		LOG1("ray segment destructor called")
	}

	template<typename T>
	friend std::ostream& operator<<(std::ostream& os, const raysegment<T>& rs);
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const raysegment<T>& rs)
{
	os << "ray at " << rs.pos << " pointing " << rs.dir;
	return os;
}


template<typename T = float>
class mysurface
{
public:
	vec3<T> pos; // at first no rotation of axis
	T diameter; // default to 10 mm, see constructor
	int type; //0 is image surface, 1 is power surface, 2 is stop surface

	__host__ __device__ mysurface(const vec3<T>& pos = vec3<T>(0,0,0), T diameter = 10, int type = 0) :
		pos(pos), diameter(diameter), type(type)
	{
		LOG1("my surface created")
	}

	__host__ __device__ ~mysurface()
	{
		LOG1("surface destructor called")
	}

	virtual int size()
	{
		return sizeof(*this);
	}
};


template<typename T = float>
class powersurface:public mysurface<T>
{
public:
	T power;//just random number, default to 0.1 mm^-1

	__host__ __device__ powersurface(T power = 0.1, const vec3<T>& pos = vec3<T>(0, 0, 0), T diameter = 10) :
		mysurface(pos, diameter, 1), power(power)
	{
		LOG1("power surface created")
	}

	__host__ __device__ ~powersurface()
	{
		LOG1("power surface destructor called")
	}

	int size()
	{
		return sizeof(*this);
	}
};

template<typename T = float>
class quadricparam
{
public:
	T A, B, C, D, E, F, G, H, I, J; //implicit equation A*x^2+B*y^2+C*z^2+D*x*y+E*x*z+F*y*z+G*x+H*y+I*z+J = 0

	__host__ __device__ quadricparam(T A = 1, T B = 1, T C = 1, T D = 0, T E = 0, T F = 0, T G = 0, T H = 0, T I = 0, T J = 0) :
		A(A), B(B), C(C), D(D), E(E), F(F), G(G), H(H), I(I), J(J)
	{}
};

template<typename T = float>
class quadricsurface :public mysurface<T>
{
public:
	quadricparam<T> param;
	T n1, n2;

	__host__ __device__ quadricsurface(const quadricparam<T>& param = quadricparam<T>(1,1,1,0,0,0,0,0,0,-1),
		T n1 = 1, T n2 = 1, const vec3<T>& pos = vec3<T>(0, 0, 0), T diameter = 10):
		mysurface(pos, diameter, 1), param(param), n1(n1), n2(n2)
	{
		LOG1("quadric surface created")
	}

	__host__ __device__ ~quadricsurface()
	{
		LOG1("quadric surface destructor called")
	}

	int size()
	{
		return sizeof(*this);
	}
};

//main tracing kernel
#ifdef nothing
template <typename T = float>
__global__ void tracer(raysegment<T>* inbundle, raysegment<T>* outbundle, const mysurface<T>* nextsurface)
{
	// get thread index
	int idx = threadIdx.x;
	
	//return if it is an inactive ray segment
	if (inbundle[idx].status == 0)
	{
		outbundle[idx] = inbundle[idx];
		return;
	}

	auto surfacetype = nextsurface->type;

    // coordinate transformation
	auto before = raysegment<MYFLOATTYPE>(inbundle[idx].pos - nextsurface->pos,inbundle[idx].dir);


	// intersection find 
	auto t = ((MYFLOATTYPE)0 - before.pos.z) / (before.dir.z);// in surface's own coordinate, the surface is at z = 0
	auto at = raysegment<MYFLOATTYPE>(before.pos + t * before.dir,before.dir);
	
	// determine if valid intersection
	if (norm(vec3<MYFLOATTYPE>(at.pos.x, at.pos.y, 0)) > (nextsurface->diameter) / 2)
	{
		inbundle[idx].status = 0;
		outbundle[idx] = inbundle[idx];
		return;
	}

	if (surfacetype == 1) // if next surface is a power surface
	{
		//surface transfer
		auto normalvec = vec3<MYFLOATTYPE>(0, 0, 1);
		auto radialvec = vec3<MYFLOATTYPE>(at.pos.x, at.pos.y, 0);
		auto binormal = normalize(cross(normalvec, radialvec));
		auto tangential = dot(at.dir, binormal)*binormal;
		auto radial = at.dir - tangential;
		auto u = acosf(dot(normalize(radial), normalize(-normalvec)));
		auto uprime = u - norm(radialvec)*((powersurface<MYFLOATTYPE>*)nextsurface)->power;

		auto newradial = norm(radial)*normalize(((-normalvec) + 
			normalize(radialvec)*((MYFLOATTYPE)tanf(uprime))));
		auto after = raysegment<MYFLOATTYPE>(at.pos, tangential + newradial);

		//printf("%d at u = %f, u' = %f\n", idx, u, uprime);

		// coordinate detransformation
		after.pos = after.pos + nextsurface->pos;

		// write results
		outbundle[idx] = after;
	}
	else if (surfacetype == 0) // if next surface is an image surface
	{
		// coordinate detransformation
		at.pos = at.pos + nextsurface->pos;
		at.status = 2;

		// write results
		outbundle[idx] = at;
	}

	

	/*printf("%d at t = %f at dir (%f,%f,%f), after dir (%f,%f,%f)\n", idx, t, at.dir.x, at.dir.y, 
		at.dir.z, after.dir.x, after.dir.y, after.dir.z );*/
}
#endif

//tester kernel
__global__ void tester()
{
	int idx = threadIdx.x;

	//test case
	auto pquad = new quadricsurface<MYFLOATTYPE>(quadricparam<MYFLOATTYPE>(1,1,1,0,0,0,0,0,0,-1));
	auto pray = new raysegment<MYFLOATTYPE>(vec3<MYFLOATTYPE>(0, 0, 0), vec3<MYFLOATTYPE>(0, 1, 1));

	// copy to the stack
	auto quadric = *pquad;
	auto before = *pray; 

	MYFLOATTYPE &A = quadric.param.A,
		&B = quadric.param.B,
		&C = quadric.param.C,
		&D = quadric.param.D,
		&E = quadric.param.E,
		&F = quadric.param.F,
		&G = quadric.param.G,
		&H = quadric.param.H,
		&K = quadric.param.I, // in order not to mix with imaginary unit, due to the symbolic calculation in Maple
		&J = quadric.param.J;
	MYFLOATTYPE &p1 = before.pos.x,
		&p2 = before.pos.y,
		&p3 = before.pos.z,
		&d1 = before.dir.x,
		&d2 = before.dir.y,
		&d3 = before.dir.z;
	MYFLOATTYPE delta = - 4*A*B*d1*d1*p2*p2 + 8*A*B*d1*d2*p1*p2 - 4*A*B*d2*d2*p1*p1 
		- 4*A*C*d1*d1*p3*p3 + 8 * A*C*d1*d3*p1*p3 - 4 * A*C*d3*d3*p1*p1 - 4*A*F*d1*d1*p2*p3 
		+ 4*A*F*d1*d2*p1*p3 + 4*A*F*d1*d3*p1*p2 - 4*A*F*d2*d3*p1*p1 - 4 * B*C*d2*d2*p3*p3 
		+ 8*B*C*d2*d3*p2*p3 - 4*B*C*d3*d2*p2*p2 + 4*B*E*d1*d2*p2*p3 - 4*B*E*d1*d3*p2*p2 
		- 4*B*E*d2*d2*p1*p3 + 4*B*E*d2*d3*p1*p2 - 4*C*D*d1*d2*p3*p3 + 4*C*D*d1*d3*p2*p3 
		+ 4*C*D*d2*d3*p1*p3 - 4*C*D*d3*d3*p1*p2 + D*D*d1*d1*p2*p2 - 2*D*D*d1*d2*p1*p2 
		+ D*D*d2*d2*p1*p1 + 2*D*E*d1*d1*p2*p3 - 2*D*E*d1*d2*p1*p3 - 2*D*E*d1*d3*p1*p2 
		+ 2*D*E*d2*d3*p1*p1 - 2*D*F*d1*d2*p2*p3 + 2*D*F*d1*d3*p2*p2 + 2*D*F*d2*d2*p1*p3 
		- 2*D*F*d2*d3*p1*p2 + E*E*d1*d1*p3*p3 - 2*E*E*d1*d3*p1*p3 + E*E*d3*d3*p1*p1 
		+ 2*E*F*d1*d2*p3*p3 - 2*E*F*d1*d3*p2*p3 - 2*E*F*d2*d3*p1*p3 + 2*E*F*d3*d3*p1*p2 
		+ F*F*d2*d2*p3*p3 - 2*F*F*d2*d3*p2*p3 + F*F*d3*d3*p2*p2 - 4*A*H*d1*d1*p2 
		+ 4*A*H*d1*d2*p1 - 4*A*K*d1*d1*p3 + 4*A*K*d1*d3*p1 + 4*B*G*d1*d2*p2 - 4*B*G*d2*d2*p1 
		- 4*B*K*d2*d2*p3 + 4*B*K*d2*d3*p2 + 4*C*G*d1*d3*p3 - 4*C*G*d3*d3*p1 + 4*C*H*d2*d3*p3 
		- 4*C*H*d3*d3*p2 + 2*D*G*d1*d1*p2 - 2*D*G*d1*d2*p1 - 2*D*H*d1*d2*p2 
		+ 2*D*H*d2*d2*p1 - 4*D*K*d1*d2*p3 + 2*D*K*d1*d3*p2 + 2*D*K*d2*d3*p1 + 2*E*G*d1*d1*p3 
		- 2*E*G*d1*d3*p1 + 2*E*H*d1*d2*p3 - 4*E*H*d1*d3*p2 + 2*E*H*d2*d3*p1 - 2*E*K*d1*d3*p3 
		+ 2*E*K*d3*d3*p1 + 2*F*G*d1*d2*p3 + 2*F*G*d1*d3*p2 - 4*F*G*d2*d3*p1 + 2*F*H*d2*d2*p3 
		- 2*F*H*d2*d3*p2 - 2*F*K*d2*d3*p3 + 2*F*K*d3*d3*p2 - 4*A*J*d1*d1 - 4*B*J*d2*d2 
		- 4*C*J*d3*d3 - 4*D*J*d1*d2 - 4*E*J*d1*d3 - 4*F*J*d2*d3 + G*G*d1*d1 + 2*G*H*d1*d2 
		+ 2*G*K*d1*d3 + H*H*d2*d2+ 2*H*K*d2*d3 + K*K*d3*d3;
	MYFLOATTYPE deno = -2 * (A*d1*d1 + B*d2*d2 + C*d3*d3 + D*d1*d2 + E*d1*d3 + F*d2*d3);
	MYFLOATTYPE beforedelta = 2 * A*d1*p1 + 2 * B*d2*p2 + 2 * C*d3*p3 + D * (d1*p2 + d2 * p1) + E * (d1*p3 + d3 * p1) + F * (d2*p3 + d3 * p2) + G * d1 + H * d2 + K * d3;
	MYFLOATTYPE t, t1, t2;
	t1 = (delta >= 0) ? (beforedelta + sqrtf(delta)) / deno : MYINFINITY + 1;
	t2 = (delta >= 0) ? (beforedelta - sqrtf(delta)) / deno : MYINFINITY + 1;
	if (t1 >= 0 && t2 >= 0)
		t = (t1 < t2) ? t1 : t2;
	else if (t1 < 0 && t2 >= 0)
		t = t2;
	else if (t2 < 0 && t1 >= 0)
		t = t1;
	else
		t = MYINFINITY + 1;

	auto at = raysegment<MYFLOATTYPE>();
	auto after = raysegment<MYFLOATTYPE>();
	auto surfnormal = vec3<MYFLOATTYPE>();
	MYFLOATTYPE factor1;

	if (t < MYINFINITY)
	{
		at = raysegment<MYFLOATTYPE>(before.pos + t * before.dir, before.dir);
		after = raysegment<MYFLOATTYPE>(at.pos, at.dir);
		MYFLOATTYPE &x = at.pos.x,
			&y = at.pos.y,
			&z = at.pos.z;
		surfnormal = normalize(vec3<MYFLOATTYPE>(2*A*x+D*y+E*z+G, 2*B*y+D*x+F*z+H,2*C*z+E*x+F*y+K));

		auto ddotn = dot(at.dir, surfnormal);
		ddotn = (ddotn < 0) ? ddotn : -ddotn; // so that the surface normal and ray are in opposite direction

		factor1 = 1 - quadric.n1*quadric.n1 / (quadric.n2*quadric.n2)
			*(1 - ddotn*ddotn);
		if (factor1 < 0)
		{
			printf("something is wrong with transfer refractive vectors");
			//deactivate ray and 
			return;
		}
		
		after.dir = quadric.n1*(at.dir - surfnormal * ddotn) / quadric.n2 - surfnormal*(MYFLOATTYPE)sqrtf(factor1);
	}
	//TODO: else deactivate the ray

	printf("delta = %f ,beforedelta = %f ,deno = %f \n", delta, beforedelta, deno);
	printf("t1 = %f ,t2 = %f ,t = %f\n", t1, t2,t);
	printf("%d at t = %f ,pos = (%f,%f,%f), surfnormal (%f,%f,%f), factor1 = %f, at dir (%f,%f,%f), after dir (%f,%f,%f)\n", 
		idx, t, at.pos.x, at.pos.y, at.pos.z, 
		surfnormal.x, surfnormal.y,surfnormal.z,factor1,
		at.dir.x, at.dir.y, at.dir.z,
		after.dir.x, after.dir.y, after.dir.z );

	//clean up the test case
	delete pquad;
	delete pray;
}

#ifdef nothing
class test
{
public:
	int t;
	void method()
	{
		LOG1("test method")
	}
	test(int t = 0):t(t)
	{
		LOG1("test object created")
	}
	~test()
	{
		LOG1("test destructor called")
	}
};
#endif

int main()
{
	LOG1("this is main program");
#ifdef something
	//create event for timing
	cudaEvent_t start, stop;
	CUDARUN(cudaEventCreate(&start));
	CUDARUN(cudaEventCreate(&stop));

	//start timing 
	CUDARUN(cudaEventRecord(start, 0));

	//launch kernel
	{tester << <1, 1 >> > ();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__);
		fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
	}
	}

	//kernel finished, stop timing, print out elapsed time
	CUDARUN(cudaEventRecord(stop, 0));
	CUDARUN(cudaEventSynchronize(stop));
	float elapsedtime;
	CUDARUN(cudaEventElapsedTime(&elapsedtime, start, stop));
	LOG2("kernel run time: " << elapsedtime << " ms\n");
#endif

#ifdef nothing
	//create event for timing
	cudaEvent_t start, stop;
	CUDARUN(cudaEventCreate(&start));
	CUDARUN(cudaEventCreate(&stop));

	//set up the surfaces manually !!!!
	float diam = 10;
	int numofsurfaces = 3;
	mysurface<MYFLOATTYPE>** psurfaces = new mysurface<MYFLOATTYPE>*[numofsurfaces];
	psurfaces[0] = new powersurface<MYFLOATTYPE>(-0.1, vec3<MYFLOATTYPE>(0, 0, 13),diam);
	psurfaces[1] = new powersurface<MYFLOATTYPE>(0.2, vec3<MYFLOATTYPE>(0, 0, 11), diam);
	psurfaces[2] = new mysurface<MYFLOATTYPE>(vec3<MYFLOATTYPE>(0, 0, 0), diam);
	

	//create ray bundles for tracing
	raysegment<MYFLOATTYPE>** bundles = new raysegment<MYFLOATTYPE>*[numofsurfaces+1];
	for (int i = 0; i < numofsurfaces+1; i++)
	{
		bundles[i] = new raysegment<MYFLOATTYPE>[bundlesize];
	}


	//set up the original bundle, manually
	for (int i = 0; i < bundlesize; i++)
	{
		static float step = diam / 32;
		static float start = -(diam / 2) + (step / 2);
		bundles[0][i] = raysegment<MYFLOATTYPE>(vec3<MYFLOATTYPE>(start + step * i, 0, 20), vec3<MYFLOATTYPE>(0, 0, -1));
		//LOG2(i <<" "<< bundle[i]);
	}


	// allocate device memory for 2 bundles
	size_t batchsize = bundlesize * sizeof(raysegment<MYFLOATTYPE>);

	raysegment<MYFLOATTYPE>* d_inbundle;
	CUDARUN(cudaMalloc((void**)&d_inbundle, batchsize));
	raysegment<MYFLOATTYPE>* d_outbundle;
	CUDARUN(cudaMalloc((void**)&d_outbundle, batchsize));


	//allocate device memory for surfaces
	void** d_psurfaces = new void*[numofsurfaces];

	for (int i = 0; i < numofsurfaces; i++)
	{
		CUDARUN(cudaMalloc((void**)&(d_psurfaces[i]), psurfaces[i]->size()));
	}


	//start timing 
	CUDARUN(cudaEventRecord(start, 0));


	//copy original bundle data to device
	CUDARUN(cudaMemcpy(d_inbundle, bundles[0], batchsize, cudaMemcpyHostToDevice));


	//copy surfaces data to device
	for (int i = 0; i < numofsurfaces; i++)
	{
		CUDARUN(cudaMemcpy(d_psurfaces[i], psurfaces[i], psurfaces[i]->size(), cudaMemcpyHostToDevice));
	}
	
	// launch kernel, copy result out, swap memory

	for (int i = 0; i < numofsurfaces; i++)
	{
		tracer <<<1, 32 >>> (d_inbundle, d_outbundle, static_cast<mysurface<MYFLOATTYPE>*>(d_psurfaces[i]));
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__);
			fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		}
		CUDARUN(cudaMemcpy(bundles[i+1], d_outbundle, batchsize, cudaMemcpyDeviceToHost));
		swap(d_inbundle, d_outbundle);
	}


	//kernel finished, stop timing, print out elapsed time
	CUDARUN(cudaEventRecord(stop, 0));
	CUDARUN(cudaEventSynchronize(stop));
	float elapsedtime;
	CUDARUN(cudaEventElapsedTime(&elapsedtime, start, stop));
	LOG2("kernel run time: " << elapsedtime << " ms\n");

	//writing results out
	for (int i = 0; i < bundlesize; i++)
	{
		LOG2("ray " << i);
		for (int j = 0; j < numofsurfaces+1; j++)
		{
			switch (bundles[j][i].status)
			{
			case 0:
				LOG2(" deactivated")
				break;
			case 1:
				LOG2(" " << bundles[j][i])
				break;
			case 2:
				if (bundles[j-1][i].status != 0)
					LOG2(" " << bundles[j][i] << " done")
				break;
			}
		}
		LOG2("\n");
	}

	//destroy cuda timing events
	CUDARUN(cudaEventDestroy(start));
	CUDARUN(cudaEventDestroy(stop));

	// free device heap momory
	cudaFree(d_inbundle);
	cudaFree(d_outbundle);
	for (int i = 0; i < numofsurfaces; i++)
	{
		cudaFree(d_psurfaces[i]);
	}
	delete[] d_psurfaces;

	//free host heap momory
	for (int i = 0; i < numofsurfaces; i++)
	{
		delete psurfaces[i];
	}
	delete[] psurfaces;
	
	for (int i = 0; i < numofsurfaces+1; i++)
	{
		delete[] bundles[i];
	}

	delete[] bundles;
#endif
}
