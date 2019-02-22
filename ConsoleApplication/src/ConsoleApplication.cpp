// ConsoleApplication.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "CppCommon.h"
#include "ThreadObject.h"

//#include <test2API.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "ImageFacilities.h"
#include "ProgramInterface.h"

int function2(int argc, char** argv)
{
	{
		std::lock_guard<std::mutex> lock(coutlock);
		std::cout << "thread" << std::this_thread::get_id() << " is running \n";
	}
	std::this_thread::sleep_for(std::chrono::seconds(1));
	return 0;
}

//auto pthreadobject1 = new ThreadObject(function2);
//auto pthreadobject2 = new ThreadObject(function2);

//TODO: ThreadManager class with a detached regulator thread
/*
+ unordered_map , key is the job type, value is the function pointer
+ manager scan the job repo, obtain the function pointer 
+ create a thread and pass (a number of jobs, pass by pointer) to them, how?
+ or else, each thread function should ask by themselves the data repo for a job
+ when job number is low, stop the thread
+ still need a data structure to manage the threads created...
... candidate: unordered_map, key is job type, value is the thread(s)
*/

//TODO: StorageManager class
/*
+ functions: job check in and check out with signature:
manager first call: bool jobCheckOut(p_input, p_output); returns whether a job can be fetched or not
manipulate the input and write to output;
bool jobCheckIn(p_input, p_output);
+ data management: job and job data inititator, destructor
+ data structure to hold the pointers to jobs
*/


#ifdef crap
//everything inside is static, as we have only one signal repo, used singleton pattern
class SignalRepo
{
public:
	bool initialized = false;

	static SignalRepo& instance()
	{
		static SignalRepo instance_;
		return instance_;
	}

	virtual ~SignalRepo() {}

	static bool stopOneInstanceOfFunction(int(*f)(int, char**))
	{
		//acquire the mutex
		
		//looking for an entry with similar function register
		//stop it
	}

	static bool stopAllInstancesOfFunction(int(*f)(int, char**))
	{
		//looking for an entry with similar function register
		//stop it
	}

private:
	SignalRepo() {};                   // prevent direct instantiation
	SignalRepo(const SignalRepo&);             // prevent copy construction
	SignalRepo& operator=(const SignalRepo&);  // prevent assignment
};
bool SignalRepo::initialized;
#endif

#ifdef nothing
class Test
{
public:
	static int a;
	static int geta()
	{
		return a;
	}

	static int getb()
	{
		return b;
	}

private:
	static int b;
};

int Test::a;
int Test::b;

extern int testfunction1();
extern int testfunction2();
#endif

//define all the externs
extern int GPUmanager(int argc = 0, char** argv = nullptr);
extern void testRenderer();
extern void testbenchGPU();

void roughUI()
{
	/*
	int count = 0;
	std::cout << "Please enter the number of surfaces\n";
	std::cin >> count;
	
	PI_Surface* surfaces = new PI_Surface[count];
	float angularResol;
	float angularExtend;

	for (int i = 0; i < count - 1; i++)
	{
		std::cout << "For surface " << i + 1 << ":\n";
		std::cout << "Please enter vertex position\n";
		std::cin >> surfaces[i].z;
		std::cout << "Please enter diameter\n";
		std::cin >> surfaces[i].diameter;
		std::cout << "Please enter curvature radius\n";
		std::cin >> surfaces[i].radius;
		std::cout << "Please enter refractive index\n";
		std::cin >> surfaces[i].refractiveIndex;
	}
	{
		int i = count - 1;
		std::cout << "For image surface:\n";
		std::cout << "Please enter diameter\n";
		std::cin >> surfaces[i].diameter;
		std::cout << "Please enter curvature radius\n";
		std::cin >> surfaces[i].radius;
	}
	{
		std::cout << "For the retina: \n";
		std::cout << "Please angular resolution\n";
		std::cin >> angularResol;
		std::cout << "Please angular extend\n";
		std::cin >> angularExtend;
	}
	*/

	//test data
	
	int count = 3;
	PI_Surface* surfaces = new PI_Surface[count];
	surfaces[0].z = 40.0; surfaces[0].diameter = 40.0; surfaces[0].radius = 40.0; surfaces[0].refractiveIndex = 2.0;
	surfaces[1].z = 10.0; surfaces[1].diameter = 40.0; surfaces[1].radius = -40.0; surfaces[1].refractiveIndex = 1.0;
	surfaces[2].diameter = 40.0; surfaces[2].radius = -60.0;
	float angularResol = 2.0;
	float angularExtend = 90.0;
	

	addOpticalConfigAt(555.0, count, surfaces, angularResol, angularExtend);

	
	std::cout << "Please enter the number of points\n";
	int pcount = 0;
	std::cin >> pcount;

	for (int i = 0; i < pcount; i++)
	{
		PI_LuminousPoint point;
		std::cout << "For point " << i + 1 << ":\n";
		std::cout << "Please enter X\n";
		std::cin >> point.x;
		std::cout << "Please enter Y\n";
		std::cin >> point.y;
		std::cout << "Please enter Z\n";
		std::cin >> point.z;
		addPoint(point);
	}
	

	checkData();
	trace();
	render();


	delete[] surfaces;
}

int main()
{
    std::cout << "Hello World!\n"; 
	//GPUmanager();

	//***********testing area***************
	//testbenchGPU();
	roughUI();
	//testRenderer();

	/*
	std::this_thread::sleep_for(std::chrono::seconds(5));
	delete pthreadobject1;
	std::this_thread::sleep_for(std::chrono::seconds(2));
	delete pthreadobject2;
	*/

	
	//testopencv();

#ifdef NDEBUG
	char t;
	std::cin >> t;
#endif

	return 0;
}