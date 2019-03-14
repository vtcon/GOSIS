#pragma once

#include <vector>
#include "glew.h"

struct VertexBufferLayoutElement
{
	unsigned int type;
	unsigned int count;
	unsigned char normalized;

	inline unsigned int getSize()
	{
		switch (type){
			case GL_FLOAT:			return 4;
			case GL_UNSIGNED_INT:	return 4;
			case GL_UNSIGNED_BYTE:	return 1;
		}
		return 0;
	}
};

class VertexBufferLayout
{
private:
	std::vector<VertexBufferLayoutElement> m_elements;
	unsigned int m_stride = 0;
public:
	VertexBufferLayout();
	~VertexBufferLayout();

	template<typename T>
	void push(unsigned int count)
	{
		static_assert(false);
	}

	template<>
	void push<float>(unsigned int count)
	{
		VertexBufferLayoutElement element = { GL_FLOAT, count, GL_FALSE };
		m_elements.push_back(element);
		m_stride += count * element.getSize();
	}

	template<>
	void push<unsigned int>(unsigned int count)
	{
		VertexBufferLayoutElement element = { GL_UNSIGNED_INT, count, GL_FALSE };
		m_elements.push_back(element);
		m_stride += count * element.getSize();
	}

	template<>
	void push<unsigned char>(unsigned int count)
	{
		VertexBufferLayoutElement element = { GL_UNSIGNED_BYTE, count, GL_TRUE };
		m_elements.push_back(element);
		m_stride += count * element.getSize();
	}

	inline const std::vector<VertexBufferLayoutElement>& getElements() const { return m_elements; }

	inline unsigned int getStride() const { return m_stride; }

};

