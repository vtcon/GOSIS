#pragma once

#include "VertexBuffer.h"
#include "VertexBufferLayout.h"

class VertexArray
{
private:
	unsigned int m_rendererID;
public:
	VertexArray();
	~VertexArray();

	void bind() const;
	void unbind() const;

	void AddBuffer(const VertexBuffer& vb, const VertexBufferLayout& layout);
};

