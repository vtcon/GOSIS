#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include "glm/glm.hpp"
#include <unordered_map>

struct ShaderProgramSource
{
	std::string vertexSource;
	std::string fragmentSource;
};

class Shader
{
private:
	unsigned int m_rendererID = 0;
	std::string m_filepath;
	std::unordered_map<std::string, int> m_UniformLocationCache;

public:
	Shader();
	Shader(const std::string& filepath);
	~Shader();

	void bind() const;
	void unbind() const;

	void setUniform(const std::string& uniformName, float v0, float v1, float v2, float v3);
	void setUniform(const std::string& uniformName, int value);
	void setUniform(const std::string& uniformName, const glm::mat4& matrix);

private:
	int getUniformLocation(const std::string& uniformName);
	ShaderProgramSource parseShader(std::string filepath);
	unsigned int compileShader(unsigned int type, const std::string& source);
	unsigned int createShader(const std::string& vertexShader, const std::string& fragmentShader);
};

