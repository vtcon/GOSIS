#pragma once

#include "mycommon.cuh"
#include "vec3.cuh"

//forward declaration

template<typename T = float>
class samplingpos
{
public:
	T u = 0;
	T v = 0;
	__host__ __device__ samplingpos(T u = 0, T v = 0)
		:u(u), v(v)
	{}
	__host__ __device__ ~samplingpos()
	{}
};

//the grammar is: device can manipulate both ray objects and pointers
template<typename T = float>
class raysegment
{
public:
	vec3<T> pos, dir;
	samplingpos<T> spos;
	T intensity = 0; //radiant intensity in W/sr

	//obsolete: 1 is active, 0 is deactivated, 2 is finised, more to come

	enum Status {active, deactivated, finished, inactive};
	Status status = active;

	__host__ __device__ raysegment(const vec3<T>& pos = vec3<T>(0, 0, 0), const vec3<T>& dir = vec3<T>(0, 0, -1), const samplingpos<T>& spos = samplingpos<T>(0, 0), T intensity = 0) :
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
	T wavelength = 555; //wavelength in nm
	raysegment<T>*prays = nullptr;
	samplingpos<T>*samplinggrid = nullptr;
	raybundle<T>* d_sibling = nullptr; //sibling to this bundle on the device
									   //one bundle can only have one sibling at a time
									   //attempt to create a new sibling will delete the existing one

	//constructor and destructor, creation is limited to host only, to transfer ray bundle to device...
	//...must create the device sibling of the bundle
	__host__ raybundle(int size = bundlesize, T wavelength = 555)
		:size(size), wavelength(wavelength)
	{
		prays = new raysegment<T>[size];
		samplinggrid = new samplingpos<T>[size];
	}
	__host__  ~raybundle()
	{
		if (d_sibling != nullptr) freesibling();
		delete[] prays;
		delete[] samplinggrid;
	}

	//copy constructor
	__host__ raybundle(const raybundle <T>& origin)
		:size(origin.size), wavelength(origin.wavelength)
	{
		prays = new raysegment<T>[size];
		samplinggrid = new samplingpos<T>[size];
		memcpy(prays, origin.prays, size * sizeof(raysegment<T>));
		memcpy(samplinggrid, origin.samplinggrid, size * sizeof(samplingpos<T>));
	}

	//copy assignment operator
	__host__ raybundle<T> operator=(const raybundle<T>& origin)
	{
		size = origin.size;
		wavelength = origin.wavelength;
		prays = new raysegment<T>[size];
		samplinggrid = new samplingpos<T>[size];
		memcpy(prays, origin.prays, size * sizeof(raysegment<T>));
		memcpy(samplinggrid, origin.samplinggrid, size * sizeof(samplingpos<T>));
		return *this;
	}

	//free the current sibling
	__host__ void freesibling()
	{
		cudaFree(d_prays);
		cudaFree(d_samplinggrid);
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
		CUDARUN(cudaMalloc((void**)&(d_samplinggrid), size * sizeof(samplingpos<T>)));

		//copy data to device
		CUDARUN(cudaMemcpy(d_sibling, this, sizeof(*this), cudaMemcpyHostToDevice));
		CUDARUN(cudaMemcpy(d_prays, prays, size * sizeof(raysegment<T>), cudaMemcpyHostToDevice));
		CUDARUN(cudaMemcpy(d_samplinggrid, samplinggrid, size * sizeof(samplingpos<T>), cudaMemcpyHostToDevice));
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
			CUDARUN(cudaMemcpy(samplinggrid, d_samplinggrid, size * sizeof(samplingpos<T>), cudaMemcpyDeviceToHost));
		}
		return *this;
	}

	//simplest initializer: generate 1D parallel ray fan along vertical direction
	__host__ raybundle<T>& init_1D_parallel(vec3<T> dir, T diam, T z_position)
	{
		float step = diam / size;
		float start = -(diam / 2) + (step / 2);
		for (int i = 0; i < size; i++)
		{
			prays[i] = raysegment<T>(vec3<T>(start + step * i, 0, z_position), dir);
			samplinggrid[i] = samplingpos<T>(i - size / 2, 0);
			printf("i = %d, (u,v) = (%f,%f), pos = (%f,%f,%f), dir = (%f,%f,%f) \n", i
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
		if (d_sibling != nullptr) freesibling();
		delete[] prays;
		delete[] samplinggrid;
		size = 0;

		//assign temporary memory
		raysegment<T>* temp_prays = new raysegment<T>[temp_size];
		samplingpos<T>* temp_samplinggrid = new samplingpos<T>[temp_size];

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
					temp_samplinggrid[size] = samplingpos<T>(i, j);
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
		samplinggrid = new samplingpos<T>[size];
		memcpy(prays, temp_prays, size * sizeof(raysegment<T>));
		memcpy(samplinggrid, temp_samplinggrid, size * sizeof(samplingpos<T>));
		delete[] temp_prays;
		delete[] temp_samplinggrid;

		//debugging: printout trace
#ifdef _DEBUGMODE2
		if (samplinggrid != nullptr && prays != nullptr)
		{
			for (int i = 0; i < size; i++)
			{
				printf("i = %d\t (u,v) = (%f,%f)\t at (%f,%f,%f)\t pointing (%f,%f,%f)\n", i,
					samplinggrid[i].u, samplinggrid[i].v,
					prays[i].pos.x, prays[i].pos.y, prays[i].pos.z,
					prays[i].dir.x, prays[i].dir.y, prays[i].dir.z);
			}
		}
		else printf("error: null ptr detected");
#endif

		return *this;
	}


private:
	
	raysegment<T>* d_prays = nullptr;
	samplingpos<T>* d_samplinggrid = nullptr;
};

template<typename T = float>
class mysurface
{
public:
	vec3<T> pos; // at first no rotation of axis
	T diameter; // default to 10 mm, see constructor

	//int type; 
	//0 is image surface, 1 is power surface, 2 is stop surface
	enum SurfaceTypes {image, refractive, stop};
	SurfaceTypes type;

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
			delete[] p_data;
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

	__host__ bool add_data(void* p_originaldata, int size)
	{
		if (size <= 0) return false;
		data_size = size;
		p_data = new char[size];
		memcpy(p_data, p_originaldata, size);
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

	__host__ __device__ quadricsurface(SurfaceTypes type, const quadricparam<T>& param = quadricparam<T>(1, 1, 1, 0, 0, 0, 0, 0, 0, -1),
		T n1 = 1, T n2 = 1, const vec3<T>& pos = vec3<T>(0, 0, 0), T diameter = 10) :
		mysurface(pos, diameter, type), param(param), n1(n1), n2(n2)
	{
		LOG1("quadric surface created")
	}

	__host__ __device__ ~quadricsurface()
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

