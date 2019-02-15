#include "pch.h"
#include "CppCommon.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

#include "ImageFacilities.h"
#include "ColorimetricLookup.h"

#include "OutputImage.h"

using namespace cv; //only inside this file

//forward declaration of internally-used functions
void showOnWindow(std::string windowName, Mat image);

//specialize the templates before using them
template<> void quickDisplay<float>(float* rawData, int rows, int columns, int longEdge);
template<> void quickDisplay<double>(double* rawData, int rows, int columns, int longEdge);
template<> void quickSave<float>(float* rawData, int rows, int columns, std::string filename, std::string path);
template<> void quickSave<double>(double* rawData, int rows, int columns, std::string filename, std::string path);

void quickDisplay()
{
	double rawdata[] = { 0.3, 0.1, 0.3, 0.9, 1.2, 0.8, 0.4, 0.5 };
	Mat bigimg;
	Mat smallimg(2, 4, CV_64FC1, rawdata);
	float scaleFactor = 1000 / max(smallimg.rows, smallimg.cols);
	resize(smallimg, bigimg, Size(), scaleFactor, scaleFactor, INTER_NEAREST);
	Mat imgout, imgoutconverted;
	normalize(bigimg, imgout, 255.0, 0.0, NORM_INF,CV_8UC1);
	std::cout << smallimg << std::endl;
	//std::cout << imgout << std::endl;
	showOnWindow("test", imgout);
}


void testopencv()
{
	//quickDisplay();

	double rawdata[] = { 0.3, 0.1, 0.3, 0.9, 1.2, 0.8, 0.4, 0.5 };
	float rawdata2[] = { 0.5, 0.6, 0.2, 0.0, 0.1, 0.5, 0.4, 1.3 };
	//quickDisplay(rawdata, 4, 2);
	
	OutputImage image1;
	image1.pushNewChannel(rawdata, 555.0, 4, 2, 0, 0);
	image1.pushNewChannel(rawdata2, 556.0, 4, 2, 1, 2);
	image1.createOutputImage(OIC_XYZ);
	
	//quickSave(rawdata, 4, 2, "rawdata.bmp", "resources/image/");
}


void initiateCV()
{

}

void showOnWindow(std::string windowName, Mat image)
{
	namedWindow(windowName); // Create a window

	imshow(windowName, image); // Show our image inside the created window.

	resizeWindow(windowName, image.cols, image.rows);

	waitKey(0); // Wait for any keystroke in the window

	destroyWindow(windowName); //destroy the created window
}

template<typename T>
void quickDisplay(T* rawData, int rows, int columns, int longEdge)
{
	std::cout << "Data type not supported, please use float or double\n";
}

template<>
void quickDisplay<float>(float* rawData, int rows, int columns, int longEdge)
{
	Mat bigimg;
	Mat smallimg(rows, columns, CV_32FC1, rawData);
	float scaleFactor = (float)longEdge / max(smallimg.rows, smallimg.cols);
	resize(smallimg, bigimg, Size(), scaleFactor, scaleFactor, INTER_NEAREST);
	Mat imgout;
	normalize(bigimg, imgout, 255.0, 0.0, NORM_INF, CV_8UC1);
	showOnWindow("Display", imgout);
}

template<>
void quickDisplay<double>(double* rawData, int rows, int columns, int longEdge)
{
	Mat bigimg;
	Mat smallimg(rows, columns, CV_64FC1, rawData);
	float scaleFactor = (float)longEdge / max(smallimg.rows, smallimg.cols);
	resize(smallimg, bigimg, Size(), scaleFactor, scaleFactor, INTER_NEAREST);
	Mat imgout;
	normalize(bigimg, imgout, 255.0, 0.0, NORM_INF, CV_8UC1);
	showOnWindow("Display", imgout);
}

template<> 
void quickSave<float>(float* rawData, int rows, int columns, std::string filename, std::string path)
{
	Mat image(rows, columns, CV_32FC1, rawData);
	Mat imgout;
	normalize(image, imgout, 255.0, 0.0, NORM_INF, CV_8UC1);
	std::string fullpath = path + filename;
	imwrite(fullpath, image);
}

template<> 
void quickSave<double>(double* rawData, int rows, int columns, std::string filename, std::string path)
{
	Mat image(rows, columns, CV_64FC1, rawData);
	Mat imgout;
	normalize(image, imgout, 255.0, 0.0, NORM_INF, CV_8UC1);
	std::string fullpath = path + filename;
	imwrite(fullpath, image);
}