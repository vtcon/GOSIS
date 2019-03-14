#include "Renderer.h"



Renderer::Renderer()
{
}


Renderer::~Renderer()
{
}

void Renderer::clear() const
{
	glClear(GL_COLOR_BUFFER_BIT);
}

void Renderer::draw(const VertexArray & va, const IndexBuffer & ib, const Shader & shader) const
{
	shader.bind();
	va.bind();
	ib.bind();
	GLCall(glDrawElements(GL_TRIANGLES, ib.getCount(), GL_UNSIGNED_INT, nullptr));
}

void Renderer::draw(const VertexArray & va, const IndexBuffer & ib, const InternalShader & shader) const
{
	shader.bind();
	va.bind();
	ib.bind();
	GLCall(glDrawElements(GL_TRIANGLES, ib.getCount(), GL_UNSIGNED_INT, nullptr));
}
