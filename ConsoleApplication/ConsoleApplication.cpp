// ConsoleApplication.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <test2API.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <algorithm>
#include <unordered_map>

//really should put these to a header file
std::mutex coutlock;
class ThreadObject;
typedef int(*fsignature)(int, char**);

#ifdef crap
//forward declaration
void threadFunctionWrapper(int(*f)(int, char**), int argc, char** argv);
#endif

//most elegant solution for threading of manager functions, RAII
class ThreadObject
{
private:
	std::atomic<bool> calltostop = false;
	std::atomic<bool> stillrunning = false;
	int argc = 0;
	char** argv = nullptr;

	void managerWrapper(fsignature f)
	{
		stillrunning = true;
		while (calltostop == false)
		{
			last_return = f(argc, argv);
		}
		stillrunning = false;
	}
	
public:
	std::thread::id threadid;
	fsignature currentfunction;
	int last_return = 0;

	ThreadObject(fsignature newf, int _argc = 0, char** _argv = nullptr) 
		:currentfunction(newf), argc(_argc), argv(_argv)
	{
		std::thread thread1(&ThreadObject::managerWrapper,this, currentfunction); //all this weirdness from INVOKE()
		calltostop = false;
		threadid = thread1.get_id();
		thread1.detach();
	}

	~ThreadObject()
	{
		calltostop = true;
		while (stillrunning == true);
		{
			std::lock_guard<std::mutex> lock(coutlock);
			std::cout << "thread" << threadid << " stopped \n";
		}
	}
};


int function2(int argc, char** argv)
{
	{
		std::lock_guard<std::mutex> lock(coutlock);
		std::cout << "thread" << std::this_thread::get_id() << " is running \n";
	}
	std::this_thread::sleep_for(std::chrono::seconds(1));
	return 0;
}


auto pthreadobject1 = new ThreadObject(function2);
auto pthreadobject2 = new ThreadObject(function2);

//TODO: ThreadManager class with a detach manager thread
/*
+ unordered_map , key is the job type, value is the function pointer
+ manager scan the job repo, obtain the function pointer 
+ create a thread and pass (a number of jobs, pass by pointer) to them, how?
+ when job number is low, stop the thread
+ still need a data structure to manage the threads created...
... candidate: unordered_map, key is job type, value is the thread(s)
*/

#ifdef crap
class ThreadSignal
{
public:
	bool quit = false;
	int last_return = 0;
};

//everything inside is static, as we have only one signal repo, used singleton pattern
class SignalRepo
{
public:
	static std::atomic<bool> allquit;

	static SignalRepo& instance()
	{
		static SignalRepo instance_;
		return instance_;
	}

	virtual ~SignalRepo() {}

	static bool allquitted()
	{
		std::lock_guard<std::mutex> lock(threadLedgerLock);
		return !(threadLedger.empty());
	}

	static bool shouldIQuit(std::thread::id checkingthread)
	{
		bool individualquit;
		{
			std::lock_guard<std::mutex> lock(threadLedgerLock);
			if (threadLedger.find(checkingthread) == threadLedger.end())
				return true; // unregistered thread will quit	
			individualquit = threadLedger[checkingthread].quit;
		}
		return allquit || individualquit;
	}

	static bool registerthread(std::thread::id newthreadid)
	{
		std::lock_guard<std::mutex> lock(threadLedgerLock);
		if (threadLedger.find(newthreadid) != threadLedger.end())
		{
			threadLedger.insert(std::make_pair(newthreadid, ThreadSignal()));
			{
				std::lock_guard<std::mutex> lock(coutlock);
				std::cout << "Thread " << newthreadid << " registered\n";
			}
			return true;
		}
		else
		{
			{
				std::lock_guard<std::mutex> lock(coutlock);
				std::cout << "Thread " << newthreadid << " already present!\n";
			}
			return false;
		}
	}

	static bool deregisterthread(std::thread::id clearthreadid)
	{
		std::lock_guard<std::mutex> lock(threadLedgerLock);
		if (threadLedger.find(clearthreadid) == threadLedger.end()) //thread not present
		{
			std::lock_guard<std::mutex> lock(coutlock);
			std::cout << "Thread " << clearthreadid << " not found \n";
			return false;
		}
		else
		{
			threadLedger.erase(clearthreadid);
			std::lock_guard<std::mutex> lock(coutlock);
			std::cout << "Thread" << clearthreadid << " deregistered \n";
			return true;
		}
	}

	static bool dispatchfunction(int(*f)(int, char**), int argc = 0, char** argv = nullptr)
	{
		std::thread thread1(threadFunctionWrapper, f, argc, argv);
		thread1.detach();
		return true;
	}

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
	//static std::mutex CPUThreadsLedgerLock;
	//static std::vector<std::thread::id> CPUThreadsLedger;

	static std::mutex threadLedgerLock;
	static std::unordered_map<std::thread::id, ThreadSignal> threadLedger;

	static bool initialized;
	
	SignalRepo() {};                   // prevent direct instantiation

	SignalRepo(const SignalRepo&);             // prevent copy construction
	SignalRepo& operator=(const SignalRepo&);  // prevent assignment
};

std::atomic<bool> SignalRepo::allquit = false;
//std::mutex SignalRepo::CPUThreadsLedgerLock;
//std::vector<std::thread::id> SignalRepo::CPUThreadsLedger;
std::mutex SignalRepo::threadLedgerLock;
std::unordered_map<std::thread::id, ThreadSignal> SignalRepo::threadLedger;
bool SignalRepo::initialized;

int function1(int argc = 0, char **argv = nullptr)
{
	{
		std::lock_guard<std::mutex> lock(coutlock);
		std::cout << "thread" << std::this_thread::get_id() << " is running \n";
	}
		std::this_thread::sleep_for(std::chrono::seconds(1));
		return 0;
}

void threadFunctionWrapper(int(*f)(int, char**), int argc = 0, char** argv = nullptr)
{
	std::thread::id threadid = std::this_thread::get_id();
	bool quit = false;
	int last_return;

	SignalRepo::registerthread(std::this_thread::get_id());
	{
		std::lock_guard<std::mutex> lock(coutlock);
		std::cout << "thread" << threadid << " is started \n";
	}
	while (quit == false)
	{
		last_return = f(argc, argv);
		//write result
		quit = SignalRepo::shouldIQuit(std::this_thread::get_id());
	}
	SignalRepo::deregisterthread(std::this_thread::get_id());
	{
		std::lock_guard<std::mutex> lock(coutlock);
		std::cout << "thread" << std::this_thread::get_id() << " quits \n";
	}
}
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

int main()
{
    std::cout << "Hello World!\n"; 

	//GPUmanager();

	//std::thread thread3(GPUmanager);
	//thread3.detach();

	std::this_thread::sleep_for(std::chrono::seconds(5));
	delete pthreadobject1;
	std::this_thread::sleep_for(std::chrono::seconds(2));
	delete pthreadobject2;
	/*
	SignalRepo::allquit = true;

	while (!SignalRepo::allquitted());
	std::this_thread::sleep_for(std::chrono::seconds(1));
	*/
	return 0;
}