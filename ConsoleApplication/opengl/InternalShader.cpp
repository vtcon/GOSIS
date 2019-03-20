#include "InternalShader.h"
#include "Renderer.h"
#include "glm/gtc/type_ptr.hpp"
#include <sstream>

InternalShader::InternalShader()
{
}

InternalShader::InternalShader(const std::string & shadersource)
{
	ShaderProgramSource source = parseShader(shadersource);
	std::cout << "Vertex shader source: \n" << source.vertexSource << std::endl;
	std::cout << "Fragment shader source: \n" << source.fragmentSource << std::endl;

	m_rendererID = createShader(source.vertexSource, source.fragmentSource);
	bind();
}

InternalShader::~InternalShader()
{
	GLCall(glDeleteProgram(m_rendererID));
}

void InternalShader::initialize(const std::string & shadersource)
{
	if (m_rendererID != 0)
	{
		GLCall(glDeleteProgram(m_rendererID));
	}

	ShaderProgramSource source = parseShader(shadersource);
#ifdef DEBUG
	std::cout << "Vertex shader source: \n" << source.vertexSource << std::endl;
	std::cout << "Fragment shader source: \n" << source.fragmentSource << std::endl;
#endif
	m_rendererID = createShader(source.vertexSource, source.fragmentSource);
	bind();
}

void InternalShader::bind() const
{
	GLCall(glUseProgram(m_rendererID));
}

void InternalShader::unbind() const
{
	GLCall(glUseProgram(0));
}

void InternalShader::setUniform(const std::string & uniformName, float v0, float v1, float v2, float v3)
{
	bind();
	int location = getUniformLocation(uniformName);
	GLCall(glUniform4f(location, v0, v1, v2, v3));
}

void InternalShader::setUniform(const std::string & uniformName, int value)
{
	bind();
	int location = getUniformLocation(uniformName);
	GLCall(glUniform1i(location, value));
}

void InternalShader::setUniform(const std::string & uniformName, const glm::mat4 & matrix)
{
	bind();
	int location = getUniformLocation(uniformName);
	glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
}

int InternalShader::getUniformLocation(const std::string & uniformName)
{
	if (m_UniformLocationCache.find(uniformName) != m_UniformLocationCache.end())
		return m_UniformLocationCache[uniformName];

	GLCall(int uniformLocation = glGetUniformLocation(m_rendererID, uniformName.c_str()));
	if (uniformLocation == -1)
		std::cout << "Uniform " << uniformName << " location error!\n";
	else
		m_UniformLocationCache[uniformName] = uniformLocation;

	return uniformLocation;
}

InternalShader::ShaderProgramSource InternalShader::parseShader(const std::string& shadersource)
{
	std::stringstream instream(shadersource);
	std::string line;
	std::stringstream sources[2];

	enum class ShaderType { NONE = -1, VERTEX = 0, FRAGMENT = 1 };
	ShaderType currentmode = ShaderType::NONE;
	while (getline(instream,line))
	{
		if (line.find("#shader") != std::string::npos)
		{
			if (line.find("vertex") != std::string::npos)
				currentmode = ShaderType::VERTEX;
			else
				currentmode = ShaderType::FRAGMENT;
		}
		else
		{
			sources[(int)currentmode] << line << "\n";
		}
	}

	return { sources[0].str(), sources[1].str() };
}

unsigned int InternalShader::compileShader(unsigned int type, const std::string & source)
{
	unsigned int id = glCreateShader(type);
	const char * src = source.c_str();
	glShaderSource(id, 1, &src, NULL);
	glCompileShader(id);

	//error handling
	int result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE)
	{
		int length;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
		char * message = (char *)alloca(length * sizeof(char));
		glGetShaderInfoLog(id, length, &length, message);
		std::cout << (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment") << " shader compiling error: \n";
		std::cout << message << std::endl;
		glDeleteShader(id);
		return 0;
	}

	return id;
}

unsigned int InternalShader::createShader(const std::string & vertexShader, const std::string & fragmentShader)
{
	unsigned int prog = glCreateProgram();
	unsigned int vs = compileShader(GL_VERTEX_SHADER, vertexShader);
	unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragmentShader);

	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);
	glValidateProgram(prog);

	glDeleteShader(vs);
	glDeleteShader(fs);

	return prog;
}
