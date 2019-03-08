#include "pch.h"
#include "CppCommon.h"
#include "OutputImage.h"
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

#include "ImageFacilities.h"
#include "ColorimetricLookup.h"

//external variables
extern unsigned int PI_rgbStandard;

//external functions
extern void XYZtoBGR(cv::Mat& XYZmat, cv::Mat& BGRmat, unsigned int RGBoption);
extern void showOnWindow(std::string windowName, cv::Mat image);

class OutputImageChannel
{
public:
	unsigned short int dataType;
	union //not the most elegant solution, but it works...
	{
		float* fdata;
		double* ddata;
	};
	float wavelength;
	int rows;
	int columns;
	int offsetX = 0;
	int offsetY = 0;
	float scaling = 1.0;
};

class OutputImage::OutputImageImpl
{
public:
	long outputRows = 0;
	long outputCols = 0;
	cv::Mat CVoutputImage;
	double* raw_data = nullptr;

	~OutputImageImpl()
	{
		freeData();
	}
	
	bool allocateData()
	{
		freeData();
		try
		{
			raw_data = new double[outputRows*outputCols];
		}
		catch (std::exception e)
		{
			std::cout << "Malloc for output image raw data failed! \n";
			return false;
		}
		return true;
	}

	void freeData()
	{
		if (raw_data != nullptr)
			delete[] raw_data;
		raw_data = nullptr;
	}
private:
};

OutputImage::OutputImage()
	:pimpl{std::make_unique<OutputImageImpl>()}
{
}

OutputImage::~OutputImage()
{
}

void OutputImage::createOutputImage(unsigned short int outputFormat)
{
	pimpl->outputCols = 0;
	pimpl->outputRows = 0;

	//parse input
	unsigned short int designatorC1, designatorC2, designatorC3;
	switch (outputFormat)
	{
	case OIC_LMS:
		designatorC1 = CLU_LMS_L;
		designatorC2 = CLU_LMS_M;
		designatorC3 = CLU_LMS_S;
		break;
	case OIC_XYZ:
		designatorC1 = CLU_XYZ_X;
		designatorC2 = CLU_XYZ_Y;
		designatorC3 = CLU_XYZ_Z;
		break;
	default:
		std::cout << "Cannot create output image in this format!\n";
		return;
		break;
	}

	//find the final size by comparing the dimensions of sub images
	for (OutputImageChannel currentChannel : allChannels)
	{
		if (currentChannel.offsetX + currentChannel.scaling*currentChannel.columns > pimpl->outputCols)
			pimpl->outputCols = currentChannel.offsetX + currentChannel.scaling*currentChannel.columns;
		if (currentChannel.offsetY + currentChannel.scaling*currentChannel.rows > pimpl->outputRows)
			pimpl->outputRows = currentChannel.offsetY + currentChannel.scaling*currentChannel.rows;
	}

	//pimpl->allocateData();
	pimpl->CVoutputImage = cv::Mat::zeros(pimpl->outputRows, pimpl->outputCols, CV_64FC3);
	std::vector<cv::Mat> fullsizeChannels(3);
	fullsizeChannels[0] = cv::Mat::zeros(pimpl->outputRows, pimpl->outputCols, CV_64FC1);// X or L
	fullsizeChannels[1] = cv::Mat::zeros(pimpl->outputRows, pimpl->outputCols, CV_64FC1);// Y or M
	fullsizeChannels[2] = cv::Mat::zeros(pimpl->outputRows, pimpl->outputCols, CV_64FC1);// Z or S

	//copy and scale and color matching for each channel
	for (OutputImageChannel currentChannel : allChannels)
	{
		//convert channel sub_image into CV_64F format
		cv::Mat temp1, temp2; //bad coding...
		if (currentChannel.dataType == OIC_FLOAT)
		{
			cv::Mat temp2 = cv::Mat(currentChannel.rows, currentChannel.columns, CV_32FC1, currentChannel.fdata);
			temp2.convertTo(temp1, CV_64FC1);
		}
		else if (currentChannel.dataType == OIC_DOUBLE)
		{
			temp1 = cv::Mat(currentChannel.rows, currentChannel.columns, CV_64FC1, currentChannel.ddata);
		}

		
		//apply scaling
		cv::resize(temp1, temp2, cv::Size(), currentChannel.scaling, currentChannel.scaling, cv::INTER_NEAREST);

		//look up color matching function
		float multiplierC[3] = {
			static_cast<float>(ColorimetricLookup::lookup(currentChannel.wavelength, designatorC1)),
			static_cast<float>(ColorimetricLookup::lookup(currentChannel.wavelength, designatorC2)),
			static_cast<float>(ColorimetricLookup::lookup(currentChannel.wavelength, designatorC3))
		};

		//std::cout << pimpl->CVoutputImage << std::endl;

		//copy to the output channels
		for (int i = 0; i < 3; i++)
		{
			cv::Mat roiOnChannel = fullsizeChannels[i](cv::Rect(currentChannel.offsetX, currentChannel.offsetY, temp2.cols, temp2.rows));
			//std::cout << roiOnChannel << std::endl;
			cv::scaleAdd(temp2, multiplierC[i], roiOnChannel, roiOnChannel);
			//std::cout << roiOnChannel << std::endl;
		}
	}

	cv::merge(fullsizeChannels, pimpl->CVoutputImage);

	//std::cout << pimpl->CVoutputImage << std::endl;
}

bool OutputImage::saveRaw(std::string path)
{
	if (path.empty())
	{
		std::cout << "Path invalid!\n";
		return false;
	}
	/*
	cv::FileStorage newRawFile(path, cv::FileStorage::WRITE);
	newRawFile << "Rows: " << pimpl->CVoutputImage.rows << "\n";
	newRawFile << "Cols: " << pimpl->CVoutputImage.cols << "\n";
	newRawFile << "Format: CV_64FC3\n";
	newRawFile << pimpl->CVoutputImage;
	*/

	std::ofstream myfile;
	myfile.open(path, std::ios::out|std::ios::binary);
	myfile << pimpl->CVoutputImage.rows << " ";
	myfile << pimpl->CVoutputImage.cols << " ";
	myfile << (int)CV_64FC3 << " ";
	myfile << pimpl->CVoutputImage;

	myfile.close();

	return true;
}

bool OutputImage::saveRGB(std::string path)
{
	cv::Mat toSave = cv::Mat::zeros(pimpl->CVoutputImage.size(), CV_8UC3);
	XYZtoBGR(pimpl->CVoutputImage, toSave, PI_rgbStandard);

	if (path.empty())
	{
		std::cout << "Path invalid!\n";
		return false;
	}

	bool result = cv::imwrite(path, toSave);
	if (result != true)
	{
		std::cout << "Cannot save image to " << path << " \n";
		return false;
	}
	return true;
}

void OutputImage::displayRGB(int rows, int columns, int offsetX, int offsetY, float scaling)
{
	//TEMPORARY VERSION: ignoring input parameters

	cv::Mat toDisplay = cv::Mat::zeros(pimpl->CVoutputImage.size(), CV_8UC3);
	XYZtoBGR(pimpl->CVoutputImage, toDisplay, PI_rgbStandard);
	cv::resize(toDisplay, toDisplay, cv::Size(), 500.0 / cv::max(toDisplay.rows, toDisplay.cols), 500.0 / cv::max(toDisplay.rows, toDisplay.cols), cv::INTER_NEAREST);
	showOnWindow("test", toDisplay);
}

template<>
void OutputImage::pushNewChannel<double>(double * data, float wavelength, int rows, int columns, int offsetX, int offsetY, float scaling)
{
	if (data == nullptr || wavelength <= 0 || rows <= 0 || columns <= 0 || offsetX < 0 || offsetY < 0 || scaling <= 0)
		return;

	OutputImageChannel topush;
	topush.ddata = data;
	topush.wavelength = wavelength;
	topush.rows = rows;
	topush.columns = columns;
	topush.offsetX = offsetX;
	topush.offsetY = offsetY;
	topush.scaling = scaling;
	topush.dataType = OIC_DOUBLE;
	allChannels.push_back(topush);
}

template<>
void OutputImage::pushNewChannel<float>(float * data, float wavelength,int rows, int columns, int offsetX, int offsetY, float scaling)
{
	if (data == nullptr || wavelength <= 0 || rows <= 0 || columns <= 0 || offsetX < 0 || offsetY < 0 || scaling <= 0)
		return;

	OutputImageChannel topush;
	topush.fdata = data;
	topush.wavelength = wavelength;
	topush.rows = rows;
	topush.columns = columns;
	topush.offsetX = offsetX;
	topush.offsetY = offsetY;
	topush.scaling = scaling;
	topush.dataType = OIC_FLOAT;
	allChannels.push_back(topush);
}