#include "Texture.h"
#include "Renderer.h"
#include "stb_image/stb_image.h"

Texture::Texture(const std::string & filePath)
{
	GLCall(glGenTextures(1, &m_renderedID));
	glBindTexture(GL_TEXTURE_2D, m_renderedID);

	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

	stbi_set_flip_vertically_on_load(1);
	m_localBuffer = stbi_load(filePath.c_str(), &m_width, &m_height, &m_nrChannel, 4);

	if (m_localBuffer)
	{
		GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, 
			GL_RGBA, GL_UNSIGNED_BYTE, m_localBuffer));
		GLCall(glGenerateMipmap(GL_TEXTURE_2D));
	}
	else
	{
		std::cout << "Unable to load texture " << filePath << std::endl;
	}
	stbi_image_free(m_localBuffer);
	unbind();
}

Texture::~Texture()
{
	GLCall(glDeleteTextures(1, &m_renderedID));
}

void Texture::bind(unsigned int slot) const
{
	GLCall(glActiveTexture(GL_TEXTURE0 + slot));
	GLCall(glBindTexture(GL_TEXTURE_2D, m_renderedID));
}

void Texture::unbind() const
{
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}
