#pragma once

#include <iostream>
#include "glew.h"

#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"
#include "InternalShader.h"

#define ASSERT(x) if (!(x)) __debugbreak();

#ifdef DEBUG
#define GLCall(x) while (glGetError() != GL_NO_ERROR);\
x;\
while (GLenum error = glGetError()) {\
std::cout << "[OpenGL error " << error << "] " << #x <<" "<< __FILE__ << ":" << __LINE__ << std::endl;\
__debugbreak();\
}
#else
#define GLCall(x) x;
#define ASSERT(x)
#endif


class Renderer
{
public:
	Renderer();
	~Renderer();

	void clear() const;

	void draw(const VertexArray& va, const IndexBuffer& ib, const Shader& shader) const;

	void draw(const VertexArray& va, const IndexBuffer& ib, const InternalShader& shader) const;
};

