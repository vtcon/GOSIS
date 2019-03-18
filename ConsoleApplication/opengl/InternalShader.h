#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include "glm/glm.hpp"
#include <unordered_map>



class InternalShader
{
private:
	unsigned int m_rendererID = 0;
	std::unordered_map<std::string, int> m_UniformLocationCache;

public:
	InternalShader();
	InternalShader(const std::string& shadersource);
	~InternalShader();

	void initialize(const std::string& shadersource);

	void bind() const;
	void unbind() const;

	void setUniform(const std::string& uniformName, float v0, float v1, float v2, float v3);
	void setUniform(const std::string& uniformName, int value);
	void setUniform(const std::string& uniformName, const glm::mat4& matrix);

private:
	struct ShaderProgramSource
	{
		std::string vertexSource;
		std::string fragmentSource;
	};

	int getUniformLocation(const std::string& uniformName);
	ShaderProgramSource parseShader(const std::string& shadersource);
	unsigned int compileShader(unsigned int type, const std::string& source);
	unsigned int createShader(const std::string& vertexShader, const std::string& fragmentShader);
};

