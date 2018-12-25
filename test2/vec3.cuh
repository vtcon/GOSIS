#pragma once

#include "mycommon.cuh"



//floating point comparison

#ifdef nothing
template <typename T = float>
inline __host__ __device__ bool operator<(T lhs, T rhs)
{
	return (lhs < rhs) ? true : false;
}

template <typename T = float>
inline __host__ __device__ bool operator>(T lhs, T rhs)
{
	return ((T)lhs > (T)rhs) ? true : false;
}

template <typename T = float>
inline __host__ __device__ bool operator<=(T lhs, T rhs)
{
	return ((T)lhs <= (T)rhs) ? true : false;
}

template <typename T = float>
inline __host__ __device__ bool operator>=(T lhs, T rhs)
{
	return ((T)lhs >= (T)rhs) ? true : false;
}


template <typename T = float>
inline __host__ __device__ bool operator==(T lhs, T rhs)
{
	return ((double)(lhs - rhs) 
		< MYZERO) ? true : false;
}
#endif

inline __host__ __device__ bool floatequal(float lhs, float rhs)
{
	return ((lhs - rhs)
		< MYZERO) ? true : false;
}

template<typename T = float>
class vec3
{
public:
	T x, y, z, t;
#ifdef nothing
	__host__ __device__ vec3(T x = 0, T y = 0, T z = 0) :
		x(x), y(y), z(z), t(1)
	{
		LOG1("vec 3 created")
	}
#endif

	__host__ __device__ vec3(T x = 0, T y = 0, T z = 0, T t = 1) :
		x(x), y(y), z(z), t(t)
	{
		LOG1("vec 3 created")
	}

	__host__ __device__ ~vec3()
	{
		LOG1("vec3 destructor called")
	}

	__host__ __device__ inline vec3<T> operator+(const vec3<T>& rhs) const
	{
		return vec3<T>(x + rhs.x, y + rhs.y, z + rhs.z);
	}


	__host__ __device__ inline vec3<T> operator-(const vec3<T>& rhs) const
	{
		return vec3<T>(x - rhs.x, y - rhs.y, z - rhs.z);
	}

	

	__host__ __device__ inline bool operator!=(const vec3<T>& rhs) const
	{
		return !(*this == rhs);
	}
};

template<typename T = float>
__host__ __device__ inline bool operator==(const vec3<T>& lhs, const vec3<T>& rhs)
{
	return ((lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.t == rhs.t)) ? true : false;
}

template<typename T = float>
__host__ __device__ T dot(const vec3<T>& lhs, const vec3<T>& rhs)
{
	return (lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z);
}

template<typename T = float>
__host__ __device__ vec3<T> cross(const vec3<T>& lhs, const vec3<T>& rhs)
{
	return vec3<T>(lhs.y*rhs.z - lhs.z*rhs.y, lhs.x*rhs.z - lhs.z*rhs.x, lhs.x*rhs.y - lhs.y*rhs.x);
}

// scalar pre multiplication
template<typename T = float>
__host__ __device__ vec3<T> operator*(T lhs, const vec3<T>& rhs)
{
	return vec3<T>(lhs*rhs.x, lhs*rhs.y, lhs*rhs.z);
}

// scalar post multiplication
template<typename T = float>
__host__ __device__ vec3<T> operator*(const vec3<T>& lhs, T rhs)
{
	return vec3<T>(lhs.x*rhs, lhs.y*rhs, lhs.z*rhs);
}

// scalar post division
template<typename T = float>
__host__ __device__ vec3<T> operator/(const vec3<T>& lhs, T rhs)
{
	if (rhs == 0) printf("Division by zero!");
	return (rhs == 0) ? vec3<T>(MYINFINITY, MYINFINITY, MYINFINITY)
		: vec3<T>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}

template<typename T = float>
__host__ __device__ T norm(const vec3<T>& rhs)
{
	return (T)sqrtf((T)(rhs.x*rhs.x + rhs.y*rhs.y + rhs.z*rhs.z));
}

template<typename T = float>
__host__ __device__ vec3<T> normalize(const vec3<T>& rhs)
{
	if (rhs == NULLVECTOR)
	{
		printf("Cannot normalize a null vector");
		return NULLVECTOR;
	}
	else
		return rhs / norm(rhs);
}

template<typename T = float>
__host__ __device__ vec3<T> operator-(const vec3<T>& rhs)
{
	return vec3<T>(-rhs.x, -rhs.y, -rhs.z);
}

template<typename T = float>
std::ostream& operator<<(std::ostream& os, const vec3<T>& vec)
{
	os << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
	return os;
}
