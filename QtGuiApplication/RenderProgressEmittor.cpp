#include "RenderProgressEmittor.h"
#include "../test2/ProgramInterface.h"


#include <thread>
RenderProgressEmittor::RenderProgressEmittor()
{
}


RenderProgressEmittor::~RenderProgressEmittor()
{
}

void RenderProgressEmittor::setProgress(int value)
{
	if (value != m_progress)
	{
		m_progress = value;
		emit progressChanges(m_progress);
	}
}

void RenderProgressEmittor::watchProgress()
{
	while (true)
	{
		int newvalue = (int)(100.0*tracer::PI_renderProgress);
		if (newvalue != m_progress)
		{
			m_progress = newvalue;
			emit progressChanges(m_progress);
		}
		using namespace std::literals::chrono_literals;
		std::this_thread::sleep_for(0.5s);
	}
}
