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

void update_map_projection_none(Mat& map_x, Mat& map_y, int rows, int cols)
{
	for (int j = 0; j < rows; j++)
	{
		for (int i = 0; i < cols; i++)
		{
			map_x.at<float>(j, i) = i;
			map_y.at<float>(j, i) = rows - j;
		}
	}
}

void update_map_projection_alongz(Mat& map_x, Mat& map_y, int rows, int columns, float thetaR, float R0, float maxTheta, float maxPhi) 
{
	for (int j = 0; j < rows; j++)
	{
		for (int i = 0; i < columns; i++)
		{
			float nxp = (2.0 * i - columns) / columns;
			float nyp = (2.0 * j - rows) / rows;
			if (nxp * nxp + nyp * nyp >= 1)
			{
				map_x.at<float>(j, i) = 0.0;
				map_y.at<float>(j, i) = 0.0;
				//printf("Map [%d, %d] to [%f, %f]\n", i, j, 0, 0);
			}
			else
			{
				float worldcoorX = nxp * R0*sin(maxPhi);
				float worldcoorY = nyp * R0*sin(maxTheta);

				float nyf = asin(worldcoorY / R0) / thetaR;
				float Rp = R0 * cos(nyf*thetaR);
				float nxf = asin(worldcoorX / Rp) / thetaR;

				//printf("Map [%d, %d] to [%f, %f]\n", i, j, nxf + columns / 2, nyf + rows / 2);
				map_x.at<float>(j, i) = nxf + columns / 2;
				map_y.at<float>(j, i) = -nyf + rows / 2;
			}
		}
	}
}

void update_map_projection_template(Mat& map_x, Mat& map_y, const Mat& src)
{
	int ind = 1;
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			switch (ind)
			{
			case 0:
				if (i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75)
				{
					map_x.at<float>(j, i) = 2 * (i - src.cols*0.25) + 0.5;
					map_y.at<float>(j, i) = 2 * (j - src.rows*0.25) + 0.5;
				}
				else
				{
					map_x.at<float>(j, i) = 0;
					map_y.at<float>(j, i) = 0;
				}
				break;
			case 1:
				map_x.at<float>(j, i) = i;
				map_y.at<float>(j, i) = src.rows - j;
				break;
			case 2:
				map_x.at<float>(j, i) = src.cols - i;
				map_y.at<float>(j, i) = j;
				break;
			case 3:
				map_x.at<float>(j, i) = src.cols - i;
				map_y.at<float>(j, i) = src.rows - j;
				break;
			} // end of switch
		}
	}
}


template<>
void quickDisplayv2<double>(double* rawData, int rows, int columns, void* map_x, void* map_y, int longEdge)
{
	//import image data
	Mat smallimg(rows, columns, CV_64FC1, rawData);

	//remapping
	Mat smallimg_remapped(smallimg.size(), smallimg.type());
	Mat* p_map_x = static_cast<Mat*>(map_x);
	Mat* p_map_y= static_cast<Mat*>(map_y);

	remap(smallimg, smallimg_remapped, *p_map_x, *p_map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

	//rescalling for display
	Mat bigimg;
	float scaleFactor = (float)longEdge / max(smallimg_remapped.rows, smallimg_remapped.cols);
	resize(smallimg_remapped, bigimg, Size(), scaleFactor, scaleFactor, INTER_NEAREST);

	//normalizing
	Mat imgout;
	normalize(bigimg, imgout, 255.0, 0.0, NORM_INF, CV_8UC1);

	//show
	showOnWindow("Display", imgout);
}

template<>
void quickDisplayv2<float>(float* rawData, int rows, int columns, void* map_x, void* map_y, int longEdge)
{
	//import image data
	Mat smallimg(rows, columns, CV_32FC1, rawData);

	//remapping
	Mat smallimg_remapped(smallimg.size(), smallimg.type());
	Mat* p_map_x = static_cast<Mat*>(map_x);
	Mat* p_map_y = static_cast<Mat*>(map_y);

	remap(smallimg, smallimg_remapped, *p_map_x, *p_map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

	//rescalling for display
	Mat bigimg;
	float scaleFactor = (float)longEdge / max(smallimg_remapped.rows, smallimg_remapped.cols);
	resize(smallimg_remapped, bigimg, Size(), scaleFactor, scaleFactor, INTER_NEAREST);

	//normalizing
	Mat imgout;
	normalize(bigimg, imgout, 255.0, 0.0, NORM_INF, CV_8UC1);

	//show
	showOnWindow("Display", imgout);
}

template<typename T>
void quickDisplayv2(T * rawData, int rows, int columns, void * map_x, void * map_y, int longEdge)
{
	std::cout << "Data type not supported, please use float or double\n";
}

template<typename T>
void generateProjectionMap(void *& mapX, void *& mapY, int rows, int columns, unsigned int projection, int argc, T * argv)
{
	Mat* p_mapX = static_cast<Mat*>(mapX);
	Mat* p_mapY = static_cast<Mat*>(mapY);
	p_mapX = new Mat(rows, columns, CV_32FC1);
	p_mapY = new Mat(rows, columns, CV_32FC1);

	switch (projection)
	{
	case IF_PROJECTION_NONE:
		update_map_projection_none(*p_mapX, *p_mapY, rows, columns);
		break;
	case IF_PROJECTION_ALONGZ:
		update_map_projection_alongz(*p_mapX, *p_mapY, rows, columns, (float)(argv[0]), (float)(argv[1]), (float)(argv[2]), (float)(argv[3]));
		break;
	case IF_PROJECTION_MECATOR:
		std::cout << "projection unimplemented\n";
		break;
	default:
		std::cout << "projection not found\n";
		break;
	}
	mapX = static_cast<void*>(p_mapX);
	mapY = static_cast<void*>(p_mapY);
	return;
}

template
void generateProjectionMap<float>(void *& mapX, void *& mapY, int rows, int columns, unsigned int projection, int argc, float* argv);

template
void generateProjectionMap<double>(void *& mapX, void *& mapY, int rows, int columns, unsigned int projection, int argc, double* argv);