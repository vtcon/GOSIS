#pragma once

#include <string>

class InternalTexture
{
private:
	unsigned int m_renderedID = 0;
	int m_width = 0, m_height = 0;
	mutable unsigned int boundSlot = 0;

public:
	InternalTexture();
	InternalTexture(unsigned char* data, int rows, int cols);
	~InternalTexture();

	void bind(unsigned int slot = 0) const;
	void unbind() const;

	inline int getWidth() { return m_width; }
	inline int getHeight() { return m_height; }
	inline unsigned int getBoundSlot() { return boundSlot; }
};

