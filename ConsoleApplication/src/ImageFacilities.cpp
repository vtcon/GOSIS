#include "pch.h"
#include "CppCommon.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

#include "ImageFacilities.h"
#include "ColorimetricLookup.h"

#include "OutputImage.h"


#ifndef MYPI
#define MYPI 3.14159265358979323846264338327950288419716939937510582097494 
#endif // ! MYPI



using namespace cv; //only inside this file

//forward declaration of internally-used functions
void showOnWindow(std::string windowName, Mat image);
void XYZtoBGR(Mat& XYZmat, Mat& BGRmat, unsigned int RGBoption = IF_SRGB);

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

void XYZtoBGR(Mat& XYZmat, Mat& BGRmat, unsigned int RGBoption)
{
	/*
	//normalize XYZ mat so that max Y = 1.0
	Mat channelY(XYZmat.size(), CV_64FC1);
	int fromtopair[] = { 1,0 };
	mixChannels(&XYZmat, 1, &channelY, 1, fromtopair, 1);
	double normY = norm(channelY, NORM_INF);
	XYZmat.convertTo(XYZmat, -1, 1.0 / normY, 0);
	*/

	//std::cout << "input img = \n" << XYZmat << "\n";

	double aWhite[3];
	double aBlack[3];
	double transformMat[9];
	double gammaLowerBound = 0.0;
	double gammaLowerBoundAlpha = 0.0;
	double gamma = 0.0;
	double gammaAlpha = 0.0;
	double gammaBeta = 0.0;

	if (RGBoption == IF_ADOBERGB)//AdobeRGB specification
	{
		aWhite[0] = 152.07; aWhite[1] = 160.00; aWhite[2] = 174.25;
		aBlack[0] = 0.5282; aBlack[1] = 0.5557; aBlack[2] = 0.6052;
		 
		transformMat[0] = 2.04159;	transformMat[1] = -0.56501; transformMat[2] = -0.34473;
		transformMat[3] = -0.96924; transformMat[4] = 1.87597;	transformMat[5] = 0.04156;
		transformMat[6] = 0.01344;	transformMat[7] = -0.11836; transformMat[8] = 1.01517;

		gammaLowerBound = 0.0;
		gammaLowerBoundAlpha = 0.0;
		gamma = 1.0 / 2.19921875;
		gammaAlpha = 1.0;
		gammaBeta = 0.0;
	}
	else //sRGB specification
	{
		aWhite[0] = 76.04; aWhite[1] = 80; aWhite[2] = 87.12;
		aBlack[0] = 0.1901; aBlack[1] = 0.2; aBlack[2] = 0.2178;

		transformMat[0] = 3.2406255;	transformMat[1] = -1.537208;	transformMat[2] = -0.4986286;
		transformMat[3] = -0.9689307;	transformMat[4] = 1.8757561;	transformMat[5] = 0.0415175;
		transformMat[6] = 0.0557101;	transformMat[7] = -0.2040211;	transformMat[8] = 1.0569959;

		gammaLowerBound = 0.0031308;
		gammaLowerBoundAlpha = 12.92;
		gamma = 1.0 / 2.4;
		gammaAlpha = 1.055;
		gammaBeta = -0.055;
	}

	//split the channels
	Mat channels[3];
	channels[0] = Mat::zeros(XYZmat.size(), CV_64FC1);
	channels[1] = Mat::zeros(XYZmat.size(), CV_64FC1);
	channels[2] = Mat::zeros(XYZmat.size(), CV_64FC1);
	split(XYZmat, channels);

	//find the brightest value in each of XYZ channels
	double normX = norm(channels[0], NORM_INF);
	double normY = norm(channels[1], NORM_INF);
	double normZ = norm(channels[2], NORM_INF);

	//std::cout << "normX =" << normX << " normY =" << normY << " normZ =" << normZ << "\n";

	//scale them to absolute value
	//by scaling each channel the max value set by reference absolute white
	double selectScale = aWhite[0] / normX;
	if ((normY*selectScale > aWhite[1]) || (normZ*selectScale > aWhite[2]))
	{
		selectScale = aWhite[1] / normY;
		if ((normX*selectScale > aWhite[0]) || (normZ*selectScale > aWhite[2]))
		{
			selectScale = aWhite[2] / normZ;
		}
	}
	for (int i = 0; i < 3; i++)
	{
		channels[i].convertTo(channels[i], -1, selectScale, 0.0);
	}

	//std::cout << "selectScale = " << selectScale << "\n";
	//std::cout << "scaled channels = \n" << channels[0] << "\n" << channels[1] << "\n" << channels[2] << "\n";

	//scale channels to normalized xyz value
	channels[0].convertTo(channels[0], -1, aWhite[0] / (aWhite[1] * (aWhite[0] - aBlack[0])), (-aBlack[0] * aWhite[0]) / (aWhite[1] * (aWhite[0] - aBlack[0])));
	channels[1].convertTo(channels[1], -1, 1 / (aWhite[1] - aBlack[1]), (-aBlack[1]) / (aWhite[1] - aBlack[1]));
	channels[2].convertTo(channels[2], -1, aWhite[2] / (aWhite[1] * (aWhite[2] - aBlack[2])), (-aBlack[2] * aWhite[2]) / (aWhite[1] * (aWhite[2] - aBlack[2])));

	//std::cout << "normalized channels = \n" << channels[0] << "\n" << channels[1] << "\n" << channels[2] << "\n";

	//create RGB channels
	Mat rgbs[3];
	rgbs[0] = Mat::zeros(XYZmat.size(), CV_64FC1); //red
	rgbs[1] = Mat::zeros(XYZmat.size(), CV_64FC1); //blue
	rgbs[2] = Mat::zeros(XYZmat.size(), CV_64FC1); //green

	scaleAdd(channels[0], transformMat[0], rgbs[0], rgbs[0]); //X to R
	scaleAdd(channels[1], transformMat[1], rgbs[0], rgbs[0]); //Y to R
	scaleAdd(channels[2], transformMat[2], rgbs[0], rgbs[0]); //Z to R
	scaleAdd(channels[0], transformMat[3], rgbs[1], rgbs[1]); //X to G
	scaleAdd(channels[1], transformMat[4], rgbs[1], rgbs[1]); //Y to G
	scaleAdd(channels[2], transformMat[5], rgbs[1], rgbs[1]); //Z to G
	scaleAdd(channels[0], transformMat[6], rgbs[2], rgbs[2]); //X to B
	scaleAdd(channels[1], transformMat[7], rgbs[2], rgbs[2]); //Y to B
	scaleAdd(channels[2], transformMat[8], rgbs[2], rgbs[2]); //Z to B

	//std::cout << "rgb channels = \n" << rgbs[0] << "\n" << rgbs[1] << "\n" << rgbs[2] << "\n";

	//clipping and do gamma transform
	for (int k = 0; k < 3; k++)
	{
		for (int i = 0; i < rgbs[k].cols; i++)
		{
			for (int j = 0; j < rgbs[k].rows; j++)
			{
				double temp = rgbs[k].at<double>(j, i);

				//gamma correction
				if (temp < gammaLowerBound) //for value lower than bound
				{
					temp = gammaLowerBoundAlpha * temp;
				}
				else
				{
					temp = gammaAlpha*pow(temp, gamma)+gammaBeta;
				}
				
				//clipping
				temp = (temp < 0) ? 0.0 : temp;
				temp = (temp > 1.0) ? 1.0 : temp;

				rgbs[k].at<double>(j, i) = temp;
			}
		}
	}
	//std::cout << "rgb channels = \n" << rgbs[0] << "\n" << rgbs[1] << "\n" << rgbs[2] << "\n";

	//assemble to BGR image
	Mat bgrs[3] = { rgbs[2], rgbs[1], rgbs[0] };
	BGRmat = Mat::zeros(XYZmat.size(), CV_64FC3);
	merge(bgrs, 3, BGRmat);


	//scale the output image to 255.0 and convert to 8UC3
	BGRmat.convertTo(BGRmat, -1, 255.0, 0.0);
	normalize(BGRmat, BGRmat, 255.0, 0.0, NORM_INF, CV_8UC3);
	//std::cout << "scaled bgr = \n" << BGRmat << "\n";
}

void testopencv()
{
	//quickDisplay();

	double rawdata[] = { 0.3, 0.1, 0.3, 0.9, 1.2, 0.8, 0.4, 0.5 };
	float rawdata2[] = { 0.5, 0.6, 0.2, 0.0, 0.1, 0.5, 0.4, 1.3 };
	double rawdata3[] = { // 3x3 of 3 components
		0.3, 0.1, 0.3, 0.9, 1.2, 0.8, 0.4, 0.5, 0.4, 
		0.3, 0.6, 0.3, 0.9, 0.1, 0.8, 0.4, 0.5, 0.6, 
		0.3, 0.1, 0.3, 0.4, 0.5, 0.4, 0.6, 0.3, 0.9
		//0.5, 0.6, 0.2, 0.0, 0.1, 0.5, 0.4, 1.3, 0.6, 0.3, 0.1, 0.3, 0.9, 1.2, 0.8, 0.4, 0.5, 0.4, 0.5, 0.6, 0.2, 0.4, 1.3, 0.6, 0.1, 0.3, 0.9,
		//0.3, 0.6, 0.3, 0.9, 0.1, 0.8, 0.4, 0.5, 0.6, 0.3, 0.1, 0.3, 0.9, 1.2, 0.8, 0.4, 0.5, 0.4, 0.3, 0.6, 0.3, 0.4, 0.5, 0.6, 0.1, 0.3, 0.9
	};

	/*
	Mat testimg(3, 3, CV_64FC3, rawdata3);
	Mat outimg;
	XYZtoBGR(testimg, outimg, IF_ADOBERGB);
	resize(outimg, outimg, Size(), 500 / max(outimg.rows, outimg.cols), 500 / max(outimg.rows, outimg.cols), INTER_NEAREST);
	showOnWindow("test", outimg);
	XYZtoBGR(testimg, outimg, IF_SRGB);
	resize(outimg, outimg, Size(), 500 / max(outimg.rows, outimg.cols), 500 / max(outimg.rows, outimg.cols), INTER_NEAREST);
	showOnWindow("test", outimg);
	*/

	/*
	Mat channelY(testimg.size(), CV_64FC1);
	std::cout << channelY << "\n";
	int fromtopair[] = { 1,0 };
	mixChannels(&testimg, 1, &channelY, 1, fromtopair, 1);
	std::cout << channelY << "\n";
	double normY = norm(channelY, NORM_INF);
	std::cout << normY << "\n";
	testimg.convertTo(testimg, -1, 1.0 / normY, 0);
	std::cout << testimg << "\n";
	*/
	//quickDisplay(rawdata, 4, 2);

	
	OutputImage image1;
	image1.pushNewChannel(rawdata, 555.0, 4, 2, 0, 0);
	image1.pushNewChannel(rawdata2, 556.0, 4, 2, 1, 2);
	image1.createOutputImage(OIC_XYZ);
	image1.displayRGB();

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
			//normalized, centered coordinates
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
				//float nxf = asin(worldcoorX / Rp) / thetaR;
				float nxf = asin(worldcoorX / Rp) * Rp / (thetaR * R0);

				//printf("Map [%d, %d] to [%f, %f]\n", i, j, nxf + columns / 2, nyf + rows / 2);
				map_x.at<float>(j, i) = nxf + columns / 2;
				map_y.at<float>(j, i) = -nyf + rows / 2;
			}
		}
	}
}

void update_map_projection_plate_carree(Mat& map_x, Mat& map_y, int rows, int columns, float thetaR, float R0, float maxTheta, float maxPhi)
{
	for (int j = 0; j < rows; j++)
	{
		for (int i = 0; i < columns; i++)
		{
			//centered coordinates
			float nxp = (i - columns/2.0);
			float nyp = (j - rows / 2.0);

			float nxf = nxp * cos(nyp*thetaR);

			//printf("Map [%d, %d] to [%f, %f]\n", i, j, nxf + columns / 2, nyf + rows / 2);
			map_x.at<float>(j, i) = nxf + columns / 2;
			map_y.at<float>(j, i) = rows - j; //simply invert
			
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
	case IF_PROJECTION_PLATE_CARREE:
		update_map_projection_plate_carree(*p_mapX, *p_mapY, rows, columns, (float)(argv[0]), (float)(argv[1]), (float)(argv[2]), (float)(argv[3]));
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

void clearProjectionMap(void *& mapX, void *& mapY)
{
	Mat* p_mapX = static_cast<Mat*>(mapX);
	Mat* p_mapY = static_cast<Mat*>(mapY);
	
	if (p_mapX != nullptr)
	{
		delete p_mapX;
		p_mapX = nullptr;
	}
	
	if (p_mapY != nullptr)
	{
		delete p_mapY;
		p_mapY = nullptr;
	}
	
	mapX = static_cast<void*>(p_mapX);
	mapY = static_cast<void*>(p_mapY);
	return;
}



//external variable needed to do inverse scale of rgb and xyz channels
extern unsigned short int PI_rawFormat;

bool importImageCV(std::vector<tracer::PI_LuminousPoint>& outputvec, std::string path, float posX, float posY, float posZ, float sizeHorz, float sizeVert, float rotX, float rotY, float rotZ, float wavelengthR, float wavelengthG, float wavelengthB, float brightness)
{
	//read in the image
	Mat inputimg = imread(path);
	if (inputimg.data == NULL)
	{
		return false;
	}
	inputimg.convertTo(inputimg, CV_32FC3);
	Mat channelBGR[3]; //BGR is openCV's order
	split(inputimg, channelBGR);
	
	//calculate the stepping vectors
	//horizontal unit vector is (-1,0,0), vertical unit vector is (0,-1,0)
	//..., as images are read from topleft to bottomright facing the world coordinate system

	//step1: rotate the vectors
	float thetaX = rotX / 180.0 * MYPI;
	float thetaY = rotY / 180.0 * MYPI;
	float thetaZ = rotZ / 180.0 * MYPI;

	float horzrotated[3];
	float vertrotated[3];

	horzrotated[0] = -cos(thetaY)*cos(thetaZ);
	horzrotated[1] = -cos(thetaX)*sin(thetaZ) - cos(thetaZ)*sin(thetaX)*sin(thetaY);
	horzrotated[2] = cos(thetaX)*cos(thetaZ)*sin(thetaY) - sin(thetaX)*sin(thetaZ);

	vertrotated[0] = cos(thetaY)*sin(thetaZ);
	vertrotated[1] = sin(thetaX)*sin(thetaY)*sin(thetaZ) - cos(thetaX)*cos(thetaZ);
	vertrotated[2] = -cos(thetaZ)*sin(thetaX) - cos(thetaX)*sin(thetaY)*sin(thetaZ);

	//step2: scale the vectors
	float pixelPitchHorz = sizeHorz / inputimg.cols;
	float pixelPitchVert = sizeVert / inputimg.rows;

	float stepHorz[3] = { 
		horzrotated[0] * pixelPitchHorz,
		horzrotated[1] * pixelPitchHorz,
		horzrotated[2] * pixelPitchHorz
	};

	float stepVert[3] = {
		vertrotated[0] * pixelPitchVert,
		vertrotated[1] * pixelPitchVert,
		vertrotated[2] * pixelPitchVert
	};

	//step3: find the origin vectors at topleft corner and translate it half a pixel pitch
	float oldorigin[3] = { posX, posY, posZ };
	
	auto step = [&stepHorz, &stepVert](float* inputvec, float* outputvec, float countHorz, float countVert)
	{
		outputvec[0] = inputvec[0] + countHorz * stepHorz[0] + countVert * stepVert[0];
		outputvec[1] = inputvec[1] + countHorz * stepHorz[1] + countVert * stepVert[1];
		outputvec[2] = inputvec[2] + countHorz * stepHorz[2] + countVert * stepVert[2];
	};

	float origin[3];
	step(oldorigin, origin, 0.5, 0.5);
	
	//lookup the inverse scaling
	unsigned short int designatorC1, designatorC2, designatorC3;
	switch (PI_rawFormat)
	{
	case OIC_LMS:
		designatorC1 = CLU_LMS_L;
		designatorC2 = CLU_LMS_M;
		designatorC3 = CLU_LMS_S;
		break;
	case OIC_XYZ:
	default:
		designatorC1 = CLU_XYZ_X;
		designatorC2 = CLU_XYZ_Y;
		designatorC3 = CLU_XYZ_Z;
		break;
	}
	
	float inversescale[3] = {
			static_cast<float>(ColorimetricLookup::lookup(wavelengthB, designatorC2)),
			static_cast<float>(ColorimetricLookup::lookup(wavelengthG, designatorC2)),
			static_cast<float>(ColorimetricLookup::lookup(wavelengthR, designatorC2))
	};
	
	for (int i = 0; i < 3; i++)
	{
		inversescale[i] = inversescale[1] / inversescale[i];
	}

	//rasterizing the images
	outputvec.clear();

	for (int i = 0; i < inputimg.rows; i++)
	{
		for (int j = 0; j < inputimg.cols; j++)
		{
			float pixelpos[3];
			step(origin, pixelpos, j, i);
			tracer::PI_LuminousPoint pointB, pointG, pointR;

			pointB.x = pixelpos[0];
			pointB.y = pixelpos[1];
			pointB.z = pixelpos[2];
			pointB.wavelength = wavelengthB;
			pointB.intensity = channelBGR[0].at<float>(i, j) * brightness * inversescale[0];

			pointG.x = pixelpos[0];
			pointG.y = pixelpos[1];
			pointG.z = pixelpos[2];
			pointG.wavelength = wavelengthG;
			pointG.intensity = channelBGR[1].at<float>(i, j) * brightness * inversescale[1];

			pointR.x = pixelpos[0];
			pointR.y = pixelpos[1];
			pointR.z = pixelpos[2];
			pointR.wavelength = wavelengthR;
			pointR.intensity = channelBGR[2].at<float>(i, j) * brightness * inversescale[2];

			outputvec.push_back(pointB);
			outputvec.push_back(pointG);
			outputvec.push_back(pointR);
		}
	}

	return true;
}

bool importCustomApo(double *& p_customApoData, int & customApoDataSize, std::string path)
{
	//just to be safe, or not...
	if (p_customApoData != nullptr)
	{
		delete p_customApoData;
	}
	p_customApoData = nullptr;
	customApoDataSize = 0;

	//try open the image
	if (path.empty()) return false;

	Mat inputimg = imread(path, IMREAD_GRAYSCALE);
	if (inputimg.empty()) return false;

	//convert to  double, rescale from 0 to 1
	Mat rescaledimg;
	normalize(inputimg, rescaledimg, 1.0, 0.0, NORM_INF, CV_64FC1);

	//allocate the pointer with new[], write the data (using memcpy is probably more safe)
	int customApoDataCount = (rescaledimg.rows*rescaledimg.cols + 2);
	customApoDataSize = customApoDataCount * sizeof(double);
	p_customApoData = new double[customApoDataCount];

	p_customApoData[0] = static_cast<double>(rescaledimg.rows);
	p_customApoData[1] = static_cast<double>(rescaledimg.cols);

	memcpy(&(p_customApoData[2]), rescaledimg.data, rescaledimg.rows*rescaledimg.cols * sizeof(double));

	return true;
}
