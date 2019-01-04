#pragma once

#include "CppCommon.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>

typedef int(*fsignature)(int, char**);

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
		std::thread thread1(&ThreadObject::managerWrapper, this, currentfunction); //all this weirdness from INVOKE()
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