#include "VertexArray.h"
#include "Renderer.h"


VertexArray::VertexArray()
{
	GLCall(glGenVertexArrays(1, &m_rendererID));
	GLCall(glBindVertexArray(m_rendererID));
}


VertexArray::~VertexArray()
{
	GLCall(glDeleteVertexArrays(1, &m_rendererID));
}

void VertexArray::bind() const
{
	GLCall(glBindVertexArray(m_rendererID));
}

void VertexArray::unbind() const
{
	GLCall(glBindVertexArray(0));
}

void VertexArray::AddBuffer(const VertexBuffer & vb, const VertexBufferLayout & layout)
{
	vb.bind();
	const auto& elements = layout.getElements();
	VertexBufferLayoutElement element;
	unsigned int offset = 0;
	for (int i = 0; i < elements.size(); i++)
	{
		element = elements[i];
		GLCall(glEnableVertexAttribArray(i));
		GLCall(glVertexAttribPointer(i, element.count , element.type, element.normalized,
			layout.getStride(), (const void*)offset));
		offset += element.count*element.getSize();
	}
}
