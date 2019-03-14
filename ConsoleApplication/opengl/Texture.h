#pragma once

#include <string>

class Texture
{
private:
	unsigned int m_renderedID = 0;
	std::string m_filePath;
	unsigned char* m_localBuffer = nullptr;
	int m_width = 0, m_height = 0, m_nrChannel = 0;

public:
	Texture(const std::string& filePath);
	~Texture();

	void bind(unsigned int slot = 0) const;
	void unbind() const;

	inline int getWidth() { return m_width; }
	inline int getHeight() { return m_height; }
};

