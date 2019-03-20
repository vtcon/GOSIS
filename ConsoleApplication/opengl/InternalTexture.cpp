#include "InternalTexture.h"


#include "Renderer.h"
#include "stb_image/stb_image.h"

InternalTexture::InternalTexture()
{
}

InternalTexture::InternalTexture(unsigned char* data, int rows, int cols) :m_width(cols), m_height(rows)
{
	GLCall(glGenTextures(1, &m_renderedID));
	glBindTexture(GL_TEXTURE_2D, m_renderedID);

	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

	if (data != nullptr || rows <= 0 || cols <= 0)
	{
		GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, m_width, m_height, 0,
			GL_RGB, GL_UNSIGNED_BYTE, data));
		GLCall(glGenerateMipmap(GL_TEXTURE_2D));
	}
	else
	{
		std::cout << "Unable to load texture" << std::endl;
	}
	
	unbind();
}

InternalTexture::~InternalTexture()
{
	GLCall(glDeleteTextures(1, &m_renderedID));
}

void InternalTexture::bind(unsigned int slot) const
{
	GLCall(glActiveTexture(GL_TEXTURE0 + slot));
	GLCall(glBindTexture(GL_TEXTURE_2D, m_renderedID));
	boundSlot = slot;
}

void InternalTexture::unbind() const
{
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));
	boundSlot = 0;
}

