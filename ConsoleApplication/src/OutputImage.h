#pragma once
#include "SolutionGlobalInclude.h"
#include <vector>
#include <memory>
#include <string>

#define OIC_DOUBLE 2
#define OIC_FLOAT 1
#define OIC_LMS 3
#define OIC_XYZ 4

class OutputImageChannel;

class OutputImage
{
public:
	std::vector<OutputImageChannel> allChannels;

	OutputImage();
	~OutputImage();

	template<typename T>
	void pushNewChannel(T* data, float wavelength, int rows, int columns, int offsetX = 0, int offsetY = 0, float scaling = 1.0);
	// After pushing a channel to the output image, please DO NOT delete or modify the size
	// otherwise, please use the provided method to remove the channel from output image

	void createOutputImage(unsigned short int outputFormat);

	void saveRaw(std::string path, std::string filename);

	void saveRGB(std::string path, std::string filename);

	void displayRGB(int rows = -1, int columns = -1, int offsetX = 0, int offsetY = 0, float scaling = 1.0);
	//if value of rows or column == -1, the function will display all pixels from offset until end of image


private:
	class OutputImageImpl; //PIMPL design pattern to reduce compilation dependency
	std::unique_ptr<OutputImageImpl> pimpl;
};

template<typename T>
inline void OutputImage::pushNewChannel(T * data, float wavelength, int rows, int columns, int offsetX, int offsetY, float scaling)
{
	std::cout << "Output Channel Format not supported!\n";
}

template<>
void OutputImage::pushNewChannel<double>(double* data, float wavelength, int rows, int columns, int offsetX, int offsetY, float scaling);

template<>
void OutputImage::pushNewChannel<float>(float* data, float wavelength, int rows, int columns, int offsetX, int offsetY, float scaling);

