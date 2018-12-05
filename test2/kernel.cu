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
	int status = 1; // 1 is active, 0 is deactive, more to come

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

	mysurface(const vec3<T>& pos = vec3<T>(0,0,0), T diameter = 10, int type = 0) :
		pos(pos), diameter(diameter), type(type)
	{
		LOG1("my surface created")
	}

	~mysurface()
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

	powersurface(T power = 0.1, const vec3<T>& pos = vec3<T>(0, 0, 0), T diameter = 10) :
		mysurface(pos, diameter, 1), power(power)
	{
		LOG1("power surface created")
	}

	~powersurface()
	{
		LOG1("power surface destructor called")
	}

	int size()
	{
		return sizeof(*this);
	}
};


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
		at.status = 0;

		// write results
		outbundle[idx] = at;
	}

	

	/*printf("%d at t = %f at dir (%f,%f,%f), after dir (%f,%f,%f)\n", idx, t, at.dir.x, at.dir.y, 
		at.dir.z, after.dir.x, after.dir.y, after.dir.z );*/
}
#ifdef nothing
#endif

__global__ void tester()
{
#ifdef nothing
	vec3<MYFLOATTYPE> lhs(sqrtf((float)2), sqrtf((float)2), 0);
	vec3<MYFLOATTYPE> rhs(-sqrtf((float)2), sqrtf((float)2), 0);
	auto result1 = lhs + rhs;
	auto result2 = lhs - rhs;
	auto result3 = dot(lhs, rhs);
	auto result4 = cross(lhs, rhs);
	auto result5 = norm(lhs);
	printf("(%f,%f,%f) (%f,%f,%f) %f (%f,%f,%f) %f", result1.x, result1.y, result1.z, result2.x, result2.y, result2.z,
		result3, result4.x, result4.y, result4.z, result5);
#endif
	float a = 5;
	float b = 6;
	float c = a / b;
	printf("%f", c);
}


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

int main()
{
	LOG1("this is main program");


	//create event for timing
	cudaEvent_t start, stop;
	CUDARUN(cudaEventCreate(&start));
	CUDARUN(cudaEventCreate(&stop));


	//set up the surfaces
	float diam = 10;
	int numofsurfaces = 2;
	mysurface<MYFLOATTYPE>** psurfaces = new mysurface<MYFLOATTYPE>*[numofsurfaces];
	psurfaces[0] = new powersurface<MYFLOATTYPE>(0.1, vec3<MYFLOATTYPE>(0, 0, diam));
	psurfaces[1] = new mysurface<MYFLOATTYPE>(vec3<MYFLOATTYPE>(0, 0, 0), diam);
	

	//create ray bundles for tracing
	raysegment<MYFLOATTYPE>** bundles = new raysegment<MYFLOATTYPE>*[numofsurfaces+1];
	for (int i = 0; i < numofsurfaces+1; i++)
	{
		bundles[i] = new raysegment<MYFLOATTYPE>[bundlesize];
	}


	//set up the original bundle
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
		int a = psurfaces[i]->size();
		LOG2(a);
	}

	auto test = powersurface<MYFLOATTYPE>(1);
	int a = sizeof(test);
	LOG2(a);
	
	// launch kernel, copy result out, swap memory
#ifdef something
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
#endif
#ifdef nothing
	{tester << <1, 1 >> > (); 
		cudaError_t cudaStatus = cudaGetLastError(); 
		if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__); 
				fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus)); 
		}
	}
#endif

#ifdef nothing
	//to do: copy result out
	raysegment<MYFLOATTYPE>* bundle2 = new raysegment<MYFLOATTYPE>[bundlesize];
	CUDARUN(cudaMemcpy(bundle2, d_outbundle, batchsize, cudaMemcpyDeviceToHost));

	//swap memory, relaunch kernel, copy result out
	{
		tracer << <1, 32 >> > (d_outbundle, d_inbundle, d_surface2);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Error at file %s line %d, ", __FILE__, __LINE__);
			fprintf(stderr, "code %d, reason %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		}
	}
	raysegment<MYFLOATTYPE>* bundle3 = new raysegment<MYFLOATTYPE>[bundlesize];
	CUDARUN(cudaMemcpy(bundle3, d_inbundle, batchsize, cudaMemcpyDeviceToHost));
#endif

	//kernel finished, stop timing, print out elapsed time
	CUDARUN(cudaEventRecord(stop, 0));
	CUDARUN(cudaEventSynchronize(stop));
	float elapsedtime;
	CUDARUN(cudaEventElapsedTime(&elapsedtime, start, stop));
	LOG2("kernel run time: " << elapsedtime << " ms\n");

	//writing results out
	for (int i = 0; i < bundlesize; i++)
	{
		LOG2(i <<" "<< bundles[0][i] << "\n" << bundles[1][i] << "\n" << bundles[2][i]);
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
}
