#pragma once

#include "mycommon.cuh"
#include "vec3.cuh"

//forward declaration
/*
template<typename T = float>
class point2D
{
public:
	T u = 0;
	T v = 0;
	__host__ __device__ point2D(T u = 0, T v = 0)
		:u(u), v(v)
	{}
	__host__ __device__ ~point2D()
	{}
};
*/
//the grammar is: device can manipulate both ray objects and pointers
template<typename T = float>
class raysegment
{
public:
	vec3<T> pos, dir;
	point2D<int> spos;
	T intensity = 0.0; //radiant intensity in W/sr

	//obsolete: 1 is active, 0 is deactivated, 2 is finised, more to come

	static enum Status {active, deactivated, finished, inactive};
	Status status = active;

	__host__ __device__ raysegment(const vec3<T>& pos = vec3<T>(0, 0, 0), const vec3<T>& dir = vec3<T>(0, 0, -1), const point2D<int>& spos = point2D<int>(0, 0), T intensity = 0) :
		pos(pos), dir(dir), spos(spos), intensity(intensity)
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

//the grammar is: host manipulate the bundle objects, device can only manipulate the bundle pointers...
template <typename T>
class raybundle
{
public:
	int size; //modifiable by member function
	float wavelength = 555; //wavelength in nm
	//to enable CUDA streams, both of those below should be converted to page-locked host memory
	// i.e. using cudaHostAlloc and cudaFreeHost
	raysegment<T>*prays = nullptr; 
	point2D<int>*samplinggrid = nullptr;
	raybundle<T>* d_sibling = nullptr; //sibling to this bundle on the device
									   //one bundle can only have one sibling at a time
									   //attempt to create a new sibling will delete the existing one

	//constructor and destructor, creation is limited to host only, to transfer ray bundle to device...
	//...must create the device sibling of the bundle
	__host__ raybundle(int size = 32, float wavelength = 555)
		:size(size), wavelength(wavelength)
	{
		/*
		prays = new raysegment<T>[size];
		samplinggrid = new point2D<int>[size];
		*/
		cleanObject();
	}
	__host__  ~raybundle()
	{
		LOG1("raybundle destructor called")
		cleanObject();
		freesibling();
		/*
		if (d_sibling != nullptr) freesibling();
		delete[] prays;
		delete[] samplinggrid;
		*/
	}

	//copy constructor
	__host__ raybundle(const raybundle <T>& origin)
		:size(origin.size), wavelength(origin.wavelength)
	{
		prays = new raysegment<T>[size];
		samplinggrid = new point2D<int>[size];
		memcpy(prays, origin.prays, size * sizeof(raysegment<T>));
		memcpy(samplinggrid, origin.samplinggrid, size * sizeof(point2D<int>));
	}

	//copy assignment operator
	__host__ raybundle<T> operator=(const raybundle<T>& origin)
	{
		cleanObject();
		size = origin.size;
		wavelength = origin.wavelength;
		prays = new raysegment<T>[size];
		samplinggrid = new point2D<int>[size];
		memcpy(prays, origin.prays, size * sizeof(raysegment<T>));
		memcpy(samplinggrid, origin.samplinggrid, size * sizeof(point2D<int>));
		return *this;
	}

	//free the current sibling
	__host__ void freesibling()
	{
		if (d_prays != nullptr)
			cudaFree(d_prays);
		if (d_samplinggrid != nullptr)
			cudaFree(d_samplinggrid);
		if (d_sibling != nullptr)
			cudaFree(d_sibling);
		d_prays = nullptr;
		d_samplinggrid = nullptr;
		d_sibling = nullptr;
	}

	//create the sibling raybundle on GPU, copy this raybundle to it, and return a device pointer
	__host__ raybundle<T>* copytosibling()
	{
		//delete the current sibling
		if (d_sibling != nullptr)
		{
			freesibling();
		}

		//allocate memory on device
		CUDARUN(cudaMalloc((void**)&d_sibling, sizeof(raybundle<T>)));
		CUDARUN(cudaMalloc((void**)&(d_prays), size * sizeof(raysegment<T>)));
		CUDARUN(cudaMalloc((void**)&(d_samplinggrid), size * sizeof(point2D<int>)));

		//copy data to device
		CUDARUN(cudaMemcpy(d_sibling, this, sizeof(*this), cudaMemcpyHostToDevice));
		CUDARUN(cudaMemcpy(d_prays, prays, size * sizeof(raysegment<T>), cudaMemcpyHostToDevice));
		CUDARUN(cudaMemcpy(d_samplinggrid, samplinggrid, size * sizeof(point2D<int>), cudaMemcpyHostToDevice));
		CUDARUN(cudaMemcpy(&(d_sibling->prays), &d_prays, sizeof(char*), cudaMemcpyHostToDevice));
		CUDARUN(cudaMemcpy(&(d_sibling->samplinggrid), &d_samplinggrid, sizeof(char*), cudaMemcpyHostToDevice));

		if (cudaGetLastError() != cudaSuccess)
		{
			freesibling();
			return nullptr;
		}
		else
			return d_sibling;
	}

	//copy the sibling bundle from GPU to this ray bundle, return this object
	__host__ raybundle<T> copyfromsibling()
	{
		if (d_sibling != nullptr)
		{
			//copy new data in, assume size and wavelength is correctly mirrored between siblings
			CUDARUN(cudaMemcpy(prays, d_prays, size * sizeof(raysegment<T>), cudaMemcpyDeviceToHost));
			CUDARUN(cudaMemcpy(samplinggrid, d_samplinggrid, size * sizeof(point2D<int>), cudaMemcpyDeviceToHost));
		}
		return *this;
	}

	//copy the sibling bundle from GPU to this ray bundle, Asynchronous variant
	__host__ raybundle<T> copyFromSiblingAsync(cudaStream_t thisstream)
	{
		if (d_sibling != nullptr)
		{
			//copy new data in, assume size and wavelength is correctly mirrored between siblings
			CUDARUN(cudaMemcpyAsync(prays, d_prays, size * sizeof(raysegment<T>), cudaMemcpyDeviceToHost, thisstream));
			CUDARUN(cudaMemcpyAsync(samplinggrid, d_samplinggrid, size * sizeof(point2D<int>), cudaMemcpyDeviceToHost, thisstream));
		}
		return *this;
	}

	//simplest initializer: generate 1D parallel ray fan along vertical direction
	__host__ raybundle<T>& init_1D_parallel(vec3<T> dir, T diam, T z_position, int insize = 32)
	{
		size = insize;
		prays = new raysegment<T>[size];
		samplinggrid = new point2D<int>[size];

		T step = diam / size;
		T start = -(diam / 2) + (step / 2);
		for (int i = 0; i < size; i++)
		{
			prays[i] = raysegment<T>(vec3<T>(start + step * i, 0, z_position), dir);
			samplinggrid[i] = point2D<int>(i - size / 2, 0);
			printf("i = %d, (u,v) = (%d,%d), pos = (%f,%f,%f), dir = (%f,%f,%f) \n", i
				, samplinggrid[i].u, samplinggrid[i].v
				, prays[i].pos.x, prays[i].pos.y, prays[i].pos.z
				, prays[i].dir.x, prays[i].dir.y, prays[i].dir.z);
		}
		return *this;
	}

	__host__ raybundle<T>& init_1D_fan(T z_position, T startTheta, T endTheta,  T phi = 0.0, int insize = 32)
	{
		cleanObject();
		size = insize;
		prays = new raysegment<T>[size];
		samplinggrid = new point2D<int>[size];

		if (startTheta == endTheta)
			endTheta = startTheta + 1;

		startTheta = (startTheta >= 0 && startTheta < 90) ? startTheta : 0.0;
		startTheta = startTheta / 180.0 * MYPI;
		endTheta = (endTheta > startTheta && endTheta < 90) ? endTheta : 89.0;
		endTheta = endTheta / 180.0 * MYPI;

		T step = (endTheta - startTheta) / size;
		T currentTheta = startTheta;
		for (int i = 0; i < size; i++)
		{
			currentTheta = startTheta + step * i;
			vec3<MYFLOATTYPE> dir;
			dir.z = -cos(currentTheta);
			dir.x = sin(currentTheta)*sin(phi);
			dir.y = sin(currentTheta)*cos(phi);
			prays[i] = raysegment<T>(vec3<T>(0, 0, z_position), dir);
			samplinggrid[i] = point2D<int>(i, 0);
			printf("i = %d, (u,v) = (%d,%d), pos = (%f,%f,%f), dir = (%f,%f,%f) \n", i
				, samplinggrid[i].u, samplinggrid[i].v
				, prays[i].pos.x, prays[i].pos.y, prays[i].pos.z
				, prays[i].dir.x, prays[i].dir.y, prays[i].dir.z);
		}
		return *this;
	}

	//more sophisticated 2D equi-spherical-area initializer, note: x is horizontal, y is vertical, z is towards observer
	__host__ raybundle<T>& init_2D_dualpolar(vec3<T> originpos, T min_horz, T max_horz, T min_vert, T max_vert, T step)
	{
		//clamping the limits to pi/2
		min_horz = (min_horz < -MYPI / 2) ? -MYPI / 2 : min_horz;
		min_vert = (min_vert < -MYPI / 2) ? -MYPI / 2 : min_vert;
		max_horz = (max_horz > MYPI / 2) ? MYPI / 2 : max_horz;
		max_vert = (max_vert > MYPI / 2) ? MYPI / 2 : max_vert;

		//checking the max and min limits, they must be at least one step apart
		min_horz = (min_horz > max_horz - step) ? (max_horz - step) : min_horz;
		min_vert = (min_vert > max_vert - step) ? (max_vert - step) : min_vert;


		int temp_size = static_cast<int>((max_horz / step - min_horz / step + 1)*
			(max_vert / step - min_vert / step + 1));

		//for safety, reclean the object before initialization
		cleanObject();
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
					<= 1)
					&& (angle_horz < MYPI / 2 && angle_vert < MYPI / 2)
					&& (sin(angle_horz)*sin(angle_horz) + sin(angle_vert)*sin(angle_vert) <= 1)
					)
				{
					//register
					temp_samplinggrid[size] = point2D<int>(i, j);
					/*
					z = -1 / sqrt(1 + tan(angle_horz)*tan(angle_horz) + tan(angle_vert)*tan(angle_vert));
					x = -z * tan(angle_horz);
					y = -z * tan(angle_vert);
					*/
					x = sin(angle_horz);
					y = sin(angle_vert);
					z = -sqrt(1 - x * x - y * y);
					temp_prays[size] = raysegment<T>(originpos, vec3<T>(x, y, z));
					size += 1;
				}
			}
		}

		//copy temporary arrays to final arrays
		prays = new raysegment<T>[size];
		samplinggrid = new point2D<int>[size];
		memcpy(prays, temp_prays, size * sizeof(raysegment<T>));
		memcpy(samplinggrid, temp_samplinggrid, size * sizeof(point2D<int>));
		delete[] temp_prays;
		delete[] temp_samplinggrid;

		//debugging: printout trace
#ifdef _MYDEBUGMODE
#ifdef _DEBUGMODE2
		if (samplinggrid != nullptr && prays != nullptr)
		{
			for (int i = 0; i < size; i++)
			{
				printf("i = %d\t (u,v) = (%d,%d)\t at (%f,%f,%f)\t pointing (%f,%f,%f)\n", i,
					samplinggrid[i].u, samplinggrid[i].v,
					prays[i].pos.x, prays[i].pos.y, prays[i].pos.z,
					prays[i].dir.x, prays[i].dir.y, prays[i].dir.z);
			}
		}
		else printf("error: null ptr detected");
#endif
#endif

		return *this;
	}

	void cleanObject(bool resetSize = true)
	{
		if (d_sibling != nullptr) freesibling();
		if (prays != nullptr)
		{
			delete[] prays;
			prays = nullptr;
		}
		if (samplinggrid != nullptr)
		{
			delete[] samplinggrid;
			samplinggrid = nullptr;
		}
		if (resetSize) size = 0;
	}

private:
	
	raysegment<T>* d_prays = nullptr;
	point2D<int>* d_samplinggrid = nullptr;

};

//defining the different apodization types, there is a look up function in kernel.cu
//implementations of the apo functions are in kernel.cu
#define APD_UNIFORM 0
#define APD_BARTLETT 1
#define APD_CUSTOM 2

template<typename T = float>
class mysurface
{
public:
	vec3<T> pos; // at first no rotation of axis
	T diameter; // default to 10 mm, see constructor
	int group; // the index of the optical group this surface belongs to

	//int type; 
	//0 is image surface, 1 is power surface, 2 is stop surface
	static enum SurfaceTypes {image, refractive, stop};
	SurfaceTypes type;

	//three members which are necessary for apodization here
	unsigned short int apodizationType = APD_UNIFORM;
	char* p_data = nullptr; // each char is 1 byte...
	int data_size = 0;//... so that the data size and offset is in bytes

	mysurface<T>* d_sibling = nullptr;

	__host__ mysurface(const vec3<T>& pos = vec3<T>(0, 0, 0), T diameter = 10, SurfaceTypes type = image) :
		pos(pos), diameter(diameter), type(type)
	{
		LOG1("my surface created")
	}

	__host__ ~mysurface()
	{
		if (d_sibling != nullptr)
			freesibling();
		if (p_data != nullptr)
		{
			data_size = 0;
			delete[] p_data;
		}
		LOG1("surface destructor called")
	}

	//for the two "copies" the sibling is NOT copied
	//copy constructor
	__host__ mysurface(const mysurface <T>& origin)
		:pos(origin.pos), diameter(origin.diameter), type(origin.type), data_size(origin.data_size)
	{
		//clear out existing data
		if (d_sibling != nullptr) freesibling();
		if (p_data != nullptr)
		{
			delete[] p_data;
			data_size = 0;
		}

		//copy the data, if it exists
		if (data_size != 0)
		{
			p_data = new char[data_size];
			memcpy(p_data, origin.p_data, data_size);
		}
		//if original object has sibling, also create this object's sibling
		if (origin.d_sibling != nullptr) copytosibling();
	}

	//copy assignment operator
	__host__ mysurface<T>& operator=(const mysurface<T>& origin)
	{
		//clean up first
		if (d_sibling != nullptr) freesibling();
		if (p_data != nullptr)
		{
			delete[] p_data;
			data_size = 0;
		}

		//copy in new data
		pos = origin.pos;
		diameter = origin.diameter;
		type = origin.type;
		data_size = origin.data_size;

		//copy the data, if it exists
		if (data_size != 0)
		{
			p_data = new char[data_size];
			memcpy(p_data, origin.p_data, data_size);
		}

		//if original object has sibling, also create this object's sibling
		if (origin.d_sibling != nullptr) copytosibling();

		return *this;
	}

	//TO DO: needed a more sophisticated implementation of this hit box function
	// return true if position is inside hit box
	//WARNING: polymorphism unusable if the object is not create inside the kernel
	/*
	__host__ __device__ bool hitbox(vec3<T> pos) const
	{
		return ((pos.x*pos.x + pos.y*pos.y) <= (diameter*diameter / 4)) ? true : false;
	}

	__host__ __device__ inline virtual raysegment<T> coordinate_transform(const raysegment<T>& original)
	{
		return raysegment<T>(original.pos - pos, original.dir);
	}

	__host__ __device__ inline virtual raysegment<T> coordinate_detransform(const raysegment<T>& original)
	{
		return raysegment<T>(original.pos + pos, original.dir);
	}
	*/
	__host__ __device__ virtual int size()
	{
		//int size = sizeof(*this);
		//int size1 = sizeof(mysurface<MYFLOATTYPE>);
		//int size2 = sizeof(quadricsurface<MYFLOATTYPE>);
		return sizeof(*this);
	}

	__host__ bool add_data(void* p_originaldata, int datasize)
	{
		if (datasize <= 0) return false;
		data_size = datasize;
		p_data = new char[datasize];
		memcpy(p_data, p_originaldata, datasize);
		return true;
	}

	__host__ void freesibling()
	{
		cudaFree(d_p_data);
		cudaFree(d_sibling);
		d_p_data = nullptr;
		d_sibling = nullptr; // redundancy
	}

	__host__ mysurface<T>* copytosibling()
	{
		//delete the current sibling
		if (d_sibling != nullptr)
		{
			freesibling();
		}

		//allocate memory on device
		CUDARUN(cudaMalloc((void**)&d_sibling, this->size()));
		CUDARUN(cudaMalloc((void**)&(d_p_data), data_size));

		//copy data to device
		CUDARUN(cudaMemcpy(d_sibling, this, this->size(), cudaMemcpyHostToDevice));
		CUDARUN(cudaMemcpy(d_p_data, this->p_data, data_size, cudaMemcpyHostToDevice));
		CUDARUN(cudaMemcpy(&(d_sibling->p_data), &d_p_data, sizeof(char*), cudaMemcpyHostToDevice));

		if (cudaGetLastError() != cudaSuccess)
		{
			freesibling();
			return nullptr;
		}
		else
			return d_sibling;
	}

	/*
	__host__ bool printoutdevicedata() const
	{
		if (d_sibling != nullptr)
		{
			printoutdevicedatakernel << <1, 1 >> > (this->d_sibling);
			return true;
		}
		else
			return false;

	}
	*/

private:
	
	char* d_p_data = nullptr;
};

template<typename T = float>
class powersurface :public mysurface<T>
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
	point2D<T> tipTilt;
	bool antiParallel = true; //if the ray and the surface normal should be anti-parallel 
							  //e.g. for a sphere, true for convex side, false for concave side

	__host__ quadricsurface(SurfaceTypes type,
		const quadricparam<T>& param = quadricparam<T>(1, 1, 1, 0, 0, 0, 0, 0, 0, -1),
		T n1 = 1, T n2 = 1,
		const vec3<T>& pos = vec3<T>(0, 0, 0),
		T diameter = 10,
		bool antiparallel = true,
		point2D<T> tipTiltAxis = point2D<T>(0,0)) 
		:mysurface(pos, diameter, type), param(param), n1(n1), n2(n2), antiParallel(antiparallel), tipTilt(tipTiltAxis)
	{
		LOG1("quadric surface created")
	}

	__host__ ~quadricsurface()
	{
		LOG1("quadric surface destructor called")
	}

	//needs to overwrite this function in every sub class inorder for it to return proper result
	__host__ __device__ int size() override
	{
		//int size = sizeof(*this);
		//int size1 = sizeof(mysurface<MYFLOATTYPE>);
		//int size2 = sizeof(quadricsurface<MYFLOATTYPE>);
		return sizeof(*this);
	}

	__host__ bool isFlat()
	{
		return (param.A == 0 && param.B == 0 && param.C == 0 && param.D == 0 && param.E == 0 && param.F == 0);
	}

	//WARNING: polymorphism unusable if the object is not create inside the kernel
	/*
	__host__ __device__ inline raysegment<T> coordinate_transform(const raysegment<T>& original) const
	{
		return raysegment<T>(original.pos - pos, original.dir);
	}

	__host__ __device__ inline raysegment<T> coordinate_detransform(const raysegment<T>& original) const
	{
		return raysegment<T>(original.pos + pos, original.dir);
	}

	__device__ bool hitbox(vec3<T> pos) const
	{
		return ((pos.x*pos.x + pos.y*pos.y) <= (diameter*diameter / 4)) ? true : false;
	}
	*/
};

class PixelArrayDescriptor
	//to define your own pixel array structure, implement this pure virtual class 
{
public:
	//test function
	//__host__ __device__ virtual bool testBool() = 0;

	__host__ __device__ virtual bool cartesian2Array(const vec3<MYFLOATTYPE>& vtx, point2D<int>& pixelCoor) const = 0;
	//convert world point to pixel coordinate, check if the point and the pixel are valid

	__host__ __device__ virtual bool array2Cartesian(const point2D<int>& pixelCoor, vec3<MYFLOATTYPE>& p1,
		vec3<MYFLOATTYPE>& p2, vec3<MYFLOATTYPE>& p3, vec3<MYFLOATTYPE>& p4) const = 0;
	//given a pixel coordinate, check whether it is valid and compute 4 corners of that pixel in world coordinate

	__host__ __device__ virtual inline bool checkArrayBound(const point2D<int>& pixelCoor) const = 0;
	//check if the point is within bound

	__host__ __device__ virtual point2D<int> getDimension() const = 0;

	__host__ __device__ virtual PixelArrayDescriptor* clone() const = 0;
	//for deep copy of polymorphic object
	//implementation's side code should be:
	//__host__ __device__ PixelArrayDescriptor* clone() const override
	//{	return new DerivedClassName(*this); }
};

class SimpleRetinaDescriptor : public PixelArrayDescriptor
{
public:
	MYFLOATTYPE m_thetaR;
	MYFLOATTYPE m_R0;
	MYFLOATTYPE m_maxTheta;
	MYFLOATTYPE m_maxPhi;
	MYFLOATTYPE m_r;

	//test function
	__host__ __device__ inline bool testBool()
	{
		return true;
	}

	__host__ __device__ SimpleRetinaDescriptor(MYFLOATTYPE thetaR = 10, MYFLOATTYPE R0 = 1,
		MYFLOATTYPE maxTheta = 90, MYFLOATTYPE maxPhi = 90)
		:m_thetaR(thetaR / 180 * MYPI), m_R0(R0), m_maxTheta(maxTheta / 180 * MYPI), m_maxPhi(maxPhi / 180 * MYPI)
	{
		m_r = m_thetaR * m_R0;
	}

	__host__ __device__ ~SimpleRetinaDescriptor()
	{}

	__host__  __device__ bool cartesian2Array(const vec3<MYFLOATTYPE>& worldCoor, point2D<int>& pixelCoor) const override
	{
		if (abs((worldCoor.x*worldCoor.x + worldCoor.y*worldCoor.y + worldCoor.z*worldCoor.z) - (m_R0*m_R0)) > MYEPSILONBIG) //numerical inaccuracies...
			return false;

		MYFLOATTYPE nyf = asin(worldCoor.y / m_R0) / m_thetaR;
		MYFLOATTYPE Rp = m_R0 * cos(nyf*m_thetaR);
		//
		//old version: this could be soooo wrong, should definitely check again
		//MYFLOATTYPE nxf = asin(worldCoor.x / Rp) / m_thetaR;
		//new version
		MYFLOATTYPE nxf = asin(worldCoor.x / Rp) * Rp / (m_thetaR * m_R0);
		pixelCoor = { static_cast<int>(floor(nxf)) ,static_cast<int>(floor(nyf)) };

		bool returnVal = checkArrayBound(pixelCoor);
		return returnVal;
	}

	__host__  __device__ bool array2Cartesian(const point2D<int>& pixelCoor, vec3<MYFLOATTYPE>& p1,
		vec3<MYFLOATTYPE>& p2, vec3<MYFLOATTYPE>& p3, vec3<MYFLOATTYPE>& p4) const override
	{
		p1 = vec3<MYFLOATTYPE>(0, 0, 0);
		p2 = p1; p3 = p1; p4 = p1;

		MYFLOATTYPE thetaY1 = static_cast<MYFLOATTYPE>(pixelCoor.ny) * m_thetaR;
		MYFLOATTYPE thetaY2 = (static_cast<MYFLOATTYPE>(pixelCoor.ny) + 1)*m_thetaR;

		//the equal prevents numerical inaccuracies
		if ((thetaY1 <= -m_maxTheta) || (thetaY2 >= m_maxTheta))
			return false;


		p1.y = m_R0 * sin(thetaY1);
		p2.y = p1.y;
		p3.y = m_R0 * sin(thetaY2);
		p4.y = p3.y;
		MYFLOATTYPE R1p = m_R0 * cos(thetaY1);
		MYFLOATTYPE R2p = m_R0 * cos(thetaY2);

		MYFLOATTYPE thetaX1;
		MYFLOATTYPE thetaX2;

		if (pixelCoor.ny >= 0)
		{
			thetaX1 = static_cast<MYFLOATTYPE>(pixelCoor.nx) * m_r / R1p;
			thetaX2 = (static_cast<MYFLOATTYPE>(pixelCoor.nx) + 1)*m_r / R1p;
		}
		else if (pixelCoor.ny < 0)
		{
			thetaX1 = static_cast<MYFLOATTYPE>(pixelCoor.nx) * m_r / R2p;
			thetaX2 = (static_cast<MYFLOATTYPE>(pixelCoor.nx) + 1)*m_r / R2p;
		}

		if ((thetaX1 <= -m_maxPhi) || (thetaX2 >= m_maxPhi))
			return false;

		p1.x = R1p * sin(thetaX1);
		p2.x = R1p * sin(thetaX2);
		p3.x = R2p * sin(thetaX2);
		p4.x = R2p * sin(thetaX1);

		p1.z = sqrt(m_R0*m_R0 - p1.x * p1.x - p1.y * p1.y);
		p2.z = sqrt(m_R0*m_R0 - p2.x * p2.x - p2.y * p2.y);
		p3.z = sqrt(m_R0*m_R0 - p3.x * p3.x - p3.y * p3.y);
		p4.z = sqrt(m_R0*m_R0 - p4.x * p4.x - p4.y * p4.y);

		return true;
	}

	__host__  __device__ inline bool checkArrayBound(const point2D<int>& pixelCoor) const override
	{
		MYFLOATTYPE thetaY1 = static_cast<MYFLOATTYPE>(pixelCoor.ny) * m_thetaR;
		MYFLOATTYPE thetaY2 = (static_cast<MYFLOATTYPE>(pixelCoor.ny) + 1)*m_thetaR;

		//the equal prevents numerical inaccuracies
		if ((thetaY1 <= -m_maxTheta) || (thetaY2 >= m_maxTheta))
			return false;

		MYFLOATTYPE R1p = m_R0 * cos(thetaY1);
		MYFLOATTYPE R2p = m_R0 * cos(thetaY2);

		MYFLOATTYPE thetaX1;
		MYFLOATTYPE thetaX2;

		if (pixelCoor.ny >= 0)
		{
			thetaX1 = static_cast<MYFLOATTYPE>(pixelCoor.nx) * m_r / R1p;
			thetaX2 = (static_cast<MYFLOATTYPE>(pixelCoor.nx) + 1)*m_r / R1p;
		}
		else if (pixelCoor.ny < 0)
		{
			thetaX1 = static_cast<MYFLOATTYPE>(pixelCoor.nx) * m_r / R2p;
			thetaX2 = (static_cast<MYFLOATTYPE>(pixelCoor.nx) + 1)*m_r / R2p;
		}

		if ((thetaX1 <= -m_maxPhi) || (thetaX2 >= m_maxPhi))
			return false;

		return true;
	}

	__host__ __device__ point2D<int> getDimension() const override
	{
		return { (int)((m_maxPhi + m_thetaR) / m_thetaR) * 2, (int)((m_maxTheta + m_thetaR) / m_thetaR) * 2 };
	}

	__host__ __device__ PixelArrayDescriptor* clone() const override
	{
		return new SimpleRetinaDescriptor(*this);
	}
};

class RetinaImageChannel
{
private:
	//disable copy constructor and copy assignment operator
	/*
	__host__ RetinaImageChannel(const RetinaImageChannel& origin)
	{}
	__host__ RetinaImageChannel operator=(const RetinaImageChannel& origin)
	{}
	*/

public:
	point2D<int> m_dimension;
	point2D<int> m_zeroOffset;
	PixelArrayDescriptor* mp_descriptor;
	MYFLOATTYPE* hp_raw = nullptr;
	MYFLOATTYPE* dp_raw = nullptr;
	RetinaImageChannel* dp_sibling = nullptr;

	__host__ RetinaImageChannel(const PixelArrayDescriptor& retinaDescriptor)
		:m_dimension(retinaDescriptor.getDimension())
	{
		m_zeroOffset = m_dimension / 2;
		createHostImage();
	}

	__host__ void setToValue(MYFLOATTYPE value, const PixelArrayDescriptor& retinaDescriptor)
	{
		for (int i = 0; i < m_dimension.x; i++)
		{
			for (int j = 0; j < m_dimension.y; j++)
			{
				point2D<int> pixelCoor(i - m_zeroOffset.x, j - m_zeroOffset.y);
				if (retinaDescriptor.checkArrayBound(pixelCoor))
				{
					hp_raw[j*m_dimension.x + i] = value;
				}

			}
		}
	}

	__host__ __device__  RetinaImageChannel(const RetinaImageChannel& origin)
		: m_dimension(origin.m_dimension), m_zeroOffset(origin.m_zeroOffset),
		hp_raw(origin.hp_raw), dp_raw(origin.dp_raw)
	{}

	__host__ __device__ ~RetinaImageChannel()
	{
#ifndef __CUDA_ARCH__ //these codes only execute on the host side, not on device's side
		deleteSibling();
		//clear all memory
		deleteHostImage();
#endif 
	}

	__host__ void createHostImage()
	{
		deleteHostImage();
		hp_raw = new MYFLOATTYPE[m_dimension.x*m_dimension.y];
		clearHostImage();
	}

	__host__ void clearHostImage()
	{
		if (hp_raw != nullptr)
			memset(hp_raw, 0, m_dimension.x * m_dimension.y * sizeof(MYFLOATTYPE));
	}

	__host__ void deleteHostImage()
	{
		if (hp_raw != nullptr)
			delete[] hp_raw;
		hp_raw = nullptr;
	}

	__host__ void createSibling()
	{
		if (hp_raw == nullptr) createHostImage();
		deleteSibling();
		CUDARUN(cudaMalloc((void**)&dp_raw, m_dimension.x * m_dimension.y * sizeof(MYFLOATTYPE)));
		CUDARUN(cudaMemcpy(dp_raw, hp_raw, m_dimension.x * m_dimension.y * sizeof(MYFLOATTYPE), cudaMemcpyHostToDevice));
		CUDARUN(cudaMalloc((void**)&dp_sibling, sizeof(RetinaImageChannel)));
		CUDARUN(cudaMemcpy(dp_sibling, this, sizeof(RetinaImageChannel), cudaMemcpyHostToDevice));
		clearSibling();
	}

	__host__ void deleteSibling()
	{
		if (dp_sibling != nullptr) cudaFree(dp_sibling);
		dp_sibling = nullptr;
		if (dp_raw != nullptr) cudaFree(dp_raw);
		dp_raw = nullptr;
	}

	__host__ void copyToSibling()
	{
		if (hp_raw == nullptr) createHostImage();
		if (dp_raw == nullptr) createSibling();
		CUDARUN(cudaMemcpy(dp_raw, hp_raw, m_dimension.x * m_dimension.y * sizeof(MYFLOATTYPE), cudaMemcpyHostToDevice));
	}

	__host__ void copyFromSibling()
	{
		if (hp_raw == nullptr) createHostImage();
		if (dp_raw != nullptr)
			CUDARUN(cudaMemcpy(hp_raw, dp_raw, m_dimension.x * m_dimension.y * sizeof(MYFLOATTYPE), cudaMemcpyDeviceToHost));
	}

	__host__ void clearSibling() //callable from host's side
	{
		CUDARUN(cudaMemset(dp_raw, 0, m_dimension.x * m_dimension.y * sizeof(MYFLOATTYPE)));
	}

	__host__ void saveToFile(const std::string& folderpath, const std::string& filename, const std::string& format)
	{}

	__device__ void resetDeviceImage() //callable from device's side
	{
		memset(dp_raw, 0, m_dimension.x * m_dimension.y * sizeof(MYFLOATTYPE));
	}

	__device__ void writePixel(const point2D<int>& pixelCoor, MYFLOATTYPE value)
	{
		dp_raw[(pixelCoor.y + m_zeroOffset.y)*m_dimension.x + (pixelCoor.x + m_zeroOffset.x)] = value;
	}

	__device__ void addToPixel(const point2D<int>& pixelCoor, MYFLOATTYPE value)
	{
		atomicAdd(&(dp_raw[(pixelCoor.y + m_zeroOffset.y)*m_dimension.x + (pixelCoor.x + m_zeroOffset.x)]), value);
	}

	__device__ MYFLOATTYPE getPixelValue(const point2D<int>& pixelCoor)
	{
		return dp_raw[(pixelCoor.y + m_zeroOffset.y)*m_dimension.x + (pixelCoor.x + m_zeroOffset.x)];
	}

private:

	//CUDA doesn't have native double-precision atomicAdd
	__device__ double atomicAdd(double* address, double val)
	{
		unsigned long long int* address_as_ull =
			(unsigned long long int*)address;
		unsigned long long int old = *address_as_ull, assumed;
		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));
		} while (assumed != old);
		return __longlong_as_double(old);
	}
};

class OpticalConfig
{
public:
	int numofsurfaces = 2;
	float wavelength = 555.0; // wavelength in nm, all surfaces data within this config uses this same wavelength
	mysurface<MYFLOATTYPE>** surfaces = nullptr;

	RetinaImageChannel* p_rawChannel = nullptr;
	PixelArrayDescriptor* p_retinaDescriptor = nullptr;

	void createImageChannel(PixelArrayDescriptor* p_RetinaDescriptorIn)
	{
		deleteImageChannel();
		p_retinaDescriptor = p_RetinaDescriptorIn;
		p_rawChannel = new RetinaImageChannel(*p_retinaDescriptor);
	}

	void deleteImageChannel()
	{
		if (p_rawChannel != nullptr)
			delete p_rawChannel;
		p_rawChannel = nullptr;

		if (p_retinaDescriptor != nullptr)
			delete p_retinaDescriptor;
		p_retinaDescriptor = nullptr;
	}

	struct EntrancePupilLocation
	{
		MYFLOATTYPE y;
		MYFLOATTYPE z;
		MYFLOATTYPE phi;
	};

	EntrancePupilLocation entrance;

	OpticalConfig(int numberOfSurfaces, float _wavelength = 555.0) 
		:numofsurfaces(numberOfSurfaces), wavelength(_wavelength)
	{
		surfaces = new mysurface<MYFLOATTYPE>*[numofsurfaces];
	}

	~OpticalConfig()
	{
		deleteImageChannel();
		freesiblings();
		for (int i = 0; i < numofsurfaces; i++)
		{
			delete surfaces[i];
		}
		delete[] surfaces;
	}

	mysurface<MYFLOATTYPE>* operator[](int i)
	{
		return surfaces[i];
	}

	void copytosiblings()
	{
		for (int i = 0; i < numofsurfaces; i++)
			surfaces[i]->copytosibling();
	}

	void freesiblings()
	{
		for (int i = 0; i < numofsurfaces; i++)
			surfaces[i]->freesibling();
	}

	void setEntrancePupil(EntrancePupilLocation newLocation)
	{
		entrance = newLocation;
	}

private:

	//disable both of them, OpticalConfig are meant to only be created and destroyed, not passing around
	OpticalConfig(const OpticalConfig& origin);
	OpticalConfig operator=(const OpticalConfig& origin);

};


class RayBundleColumn 
{
public:
	int numofsurfaces = 2;
	float wavelength = 555.0;

	raybundle<MYFLOATTYPE>** bundles = nullptr;

	RayBundleColumn(int numberOfSurfaces, float _wavelength)
		:numofsurfaces(numberOfSurfaces), wavelength(_wavelength)
	{
		bundles = new raybundle<MYFLOATTYPE>*[numofsurfaces + 1];
		for (int i = 0; i < numofsurfaces + 1; i++)
		{
			bundles[i] = new raybundle<MYFLOATTYPE>(32, wavelength); //initialize with 32 rays, for example
		}
	}

	~RayBundleColumn()
	{
		LOG2("column destructor called")
		for (int i = 0; i < numofsurfaces + 1; i++)
		{
			delete bundles[i];
		}
		delete[] bundles;
	}

	raybundle<MYFLOATTYPE>& operator[](int i)
	{
		return *(bundles[i]);
	}
private:
	//disable both until an appropriate use for them can be found
	RayBundleColumn(const RayBundleColumn& origin);
	RayBundleColumn operator=(const RayBundleColumn& origin);
};


class LuminousPoint
{
public:
	float x = 0.0;
	float y = 0.0;
	float z = 0.0;
	float wavelength = 555.0;
	float intensity = 1.0;

	LuminousPoint()
	{}

	bool operator==(const LuminousPoint& rhs) const
	{
		if (x == rhs.x &&y == rhs.y &&z == rhs.z &&wavelength == rhs.wavelength &&intensity == rhs.intensity)
			return true;
		else
			return false;
	}
};





