#include <iostream>
#include "glew.h"
#include "glfw3.h"

#include <fstream>
#include <string>
#include <sstream>
#include <list>

#include "Renderer.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArray.h"

#include "Shader.h"
#include "Texture.h"
#include "Renderer.h"
#include "Camera.h"
#include "InternalShader.h"
#include "InternalTexture.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "../src/GLDrawFacilities.h"

static Camera camera; //it must be a global variable, as function callbacks depends on it

static bool lbutton_down = false;
static double lastCursorPos_x = 0;
static double lastCursorPos_y = 0;

void mouseScrollCallback(GLFWwindow * window, double xoffset, double yoffset);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow *window, float deltaTime);

static int m_windowWidth = 1280;
static int m_windowHeight = 720;
static int m_windowDepth = 720;

static int windowCounter = 0;

/************************Shader Sources*************************************/

#define USE_SHADER_PURE_COLOR 0
#define USE_SHADER_PURE_TEXTURE 1
#define USE_SHADER_MIXED 2

class ShaderCaliber
{
public:
	ShaderCaliber()
	{
		vectorShaders.push_back(InternalShader());
		vectorShaders.push_back(InternalShader());
		vectorShaders.push_back(InternalShader());

		vectorShaders[0].initialize(BasicShaderNoTex);
		vectorShaders[1].initialize(BasicShaderOnlyTex);
		vectorShaders[2].initialize(BasicShaderMixed);
	}

	~ShaderCaliber()
	{

	}

	void setUniform(const std::string & uniformName, int value)
	{
		vectorShaders[currentShader].bind();
		vectorShaders[currentShader].setUniform(uniformName, value);
	}

	void setUniform(const std::string& uniformName, float v0, float v1, float v2, float v3)
	{
		vectorShaders[currentShader].bind();
		vectorShaders[currentShader].setUniform(uniformName, v0, v1, v2, v3);
	}

	void setUniform(const std::string& uniformName, const glm::mat4& matrix)
	{
		vectorShaders[currentShader].bind();
		vectorShaders[currentShader].setUniform(uniformName, matrix);
	}

	void unbind()
	{
		vectorShaders[0].unbind();
	}

	void useShader(unsigned int shaderCode)
	{
		switch (shaderCode)
		{
		case USE_SHADER_PURE_TEXTURE:
			vectorShaders[1].bind();
			currentShader = 1;
			break;
		case USE_SHADER_MIXED:
			vectorShaders[2].bind();
			currentShader = 2;
			break;
		case USE_SHADER_PURE_COLOR:
		default:
			vectorShaders[0].bind();
			currentShader = 0;
			break;
		}
	}

	InternalShader& getCurrentShader()
	{
		return vectorShaders[currentShader];
	}

private:
	unsigned int currentShader = 0;
	std::vector<InternalShader> vectorShaders;

	std::string BasicShader = "#shader vertex\n\
		#version 330 core\n\
		\
		layout(location = 0) in vec4 position;\n\
		layout(location = 1) in vec2 texCoor;\n\
		\
		uniform mat4 u_MVP;\n\
		\
		out vec2 v_texCoor;\n\
		\
		void main()\n\
		{\n\
			gl_Position = u_MVP * position;\n\
			v_texCoor = texCoor;\n\
		};\n\
		\
		#shader fragment\n\
		#version 330 core\n\
		\
		layout(location = 0) out vec4 color;\n\
		\
		in vec2 v_texCoor;\n\
		\
		uniform vec4 u_color;\n\
		uniform sampler2D u_textureSlot0;\n\
		uniform sampler2D u_textureSlot1;\n\
		\
		void main()\n\
		{\n\
			vec4 texColor = mix(texture(u_textureSlot0, v_texCoor), texture(u_textureSlot1, v_texCoor), 0.8);\n\
			color = texColor * u_color;\n\
		}; ";

	std::string BasicShaderMixed = "#shader vertex\n\
		#version 330 core\n\
		\
		layout(location = 0) in vec4 position;\n\
		layout(location = 1) in vec2 texCoor;\n\
		\
		uniform mat4 u_MVP;\n\
		\
		out vec2 v_texCoor;\n\
		\
		void main()\n\
		{\n\
			gl_Position = u_MVP * position;\n\
			v_texCoor = texCoor;\n\
		};\n\
		\
		#shader fragment\n\
		#version 330 core\n\
		\
		layout(location = 0) out vec4 color;\n\
		\
		in vec2 v_texCoor;\n\
		\
		uniform vec4 u_color;\n\
		uniform sampler2D u_textureSlot0;\n\
		\
		void main()\n\
		{\n\
			vec4 texColor = texture(u_textureSlot0, v_texCoor);\n\
			color = texColor * u_color;\n\
		}; ";

	std::string BasicShaderOnlyTex = "#shader vertex\n\
		#version 330 core\n\
		\
		layout(location = 0) in vec4 position;\n\
		layout(location = 1) in vec2 texCoor;\n\
		\
		uniform mat4 u_MVP;\n\
		\
		out vec2 v_texCoor;\n\
		\
		void main()\n\
		{\n\
			gl_Position = u_MVP * position;\n\
			v_texCoor = texCoor;\n\
		};\n\
		\
		#shader fragment\n\
		#version 330 core\n\
		\
		layout(location = 0) out vec4 color;\n\
		\
		in vec2 v_texCoor;\n\
		\
		uniform vec4 u_color;\n\
		uniform sampler2D u_textureSlot0;\n\
		\
		void main()\n\
		{\n\
			vec4 texColor = texture(u_textureSlot0, v_texCoor);\n\
			color = texColor;\n\
		}; ";

	std::string BasicShaderNoTex = "#shader vertex\n\
		#version 330 core\n\
		\
		layout(location = 0) in vec4 position;\n\
		\
		uniform mat4 u_MVP;\n\
		\
		void main()\n\
		{\n\
			gl_Position = u_MVP * position;\n\
		};\n\
		\
		#shader fragment\n\
		#version 330 core\n\
		\
		layout(location = 0) out vec4 color;\n\
		\
		uniform vec4 u_color;\n\
		void main()\n\
		{\n\
			color = u_color;\n\
		}; ";
};


/*************************End of Shader Sources*****************************/

int GLInfoPrint()
{
	GLFWwindow* window;

	if (!glfwInit())
	{
		std::cout << "Cannot initialize OpenGL!\n";
		return -1;
	}
		

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(m_windowWidth, m_windowHeight, "Graphical Output", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	//set up callbacks
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetScrollCallback(window, mouseScrollCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, mouseMoveCallback);

	if (glewInit() != GLEW_OK)
	{
		std::cout << "Cannot initialize OpenGL!\n";
		return -1;
	}

	std::cout << " Vendor: " << glGetString(GL_VENDOR) << std::endl;
	std::cout << " Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << " Renderer: " << glGetString(GL_RENDERER) << std::endl;

	glfwTerminate();
	return 0;
}

int GLDrawer(const std::vector<std::pair<glm::vec3, glm::vec3>>& poArray, 
			const std::vector<std::vector<float>>& vaoArray,
			const std::vector<std::vector<unsigned int>>& iboArray,
			const std::vector<TextureIndicator>& tiArray)
{
	GLFWwindow* window;

	/* Initialize the library */
	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	/* Create a windowed mode window and its OpenGL context */
	if (windowCounter != 0)
	{
		std::cout << "Please first close the already opened graphical window!\n";
		return -1;
	}
	windowCounter++;
	std::string title = "Graphical Output" + std::to_string(windowCounter);
	window = glfwCreateWindow(m_windowWidth, m_windowHeight, title.c_str(), NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	//set up callbacks
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetScrollCallback(window, mouseScrollCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, mouseMoveCallback);

	//enable Vsync
	glfwSwapInterval(1);

	if (glewInit() != GLEW_OK)
		std::cerr << "Glew init error!" << std::endl;

	
	//setup blending
	GLCall(glEnable(GL_BLEND));
	GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
	GLCall(glBlendEquation(GL_FUNC_ADD));

	//setup Depth testing
	GLCall(glEnable(GL_DEPTH_TEST));

	//setup culling
	GLCall(glEnable(GL_CULL_FACE));
	GLCall(glFrontFace(GL_CW));	

	{
		//initial MVP mat
		glm::mat4 projmat = glm::perspective(glm::radians(45.0f), (float)m_windowWidth / (float)m_windowHeight, 0.1f, 100.0f);
		glm::mat4 viewmat = glm::lookAt(
			glm::vec3(0, 0, 5),
			glm::vec3(0, 0, 0),
			glm::vec3(0, 1, 0)
		);
		glm::mat4 modelmat = glm::mat4(1.0f);
		glm::mat4 MVPmat = projmat * viewmat * modelmat;

//#ifdef  nothing
		//loading models
		int surfaceCount = iboArray.size();
		std::list<VertexArray> ivaArray;
		std::list<VertexBuffer> ivbArray;
		std::list<VertexBufferLayout> ivblArray;
		std::list<IndexBuffer> iibArray;
		std::list<unsigned int> shaderCodeArray;
		std::list<InternalTexture> texArray;
		ShaderCaliber shaderCaliber;
		//PLEASE NEVER change these lists into vectors, because no matter how you construct next object,...
		//...the previous objects in the vector get pushed to front and thus destructors are called,...
		//...which will instruct GL to destroy created objects (shader, vao, ibo etc.)

		for (int si = 0; si < surfaceCount; si++)
		{
			ivaArray.emplace_back(); //don't use push_back either, as the destructor will then get called...
			ivblArray.emplace_back();
			ivblArray.back().push<float>(3);
			if (tiArray[si].hasTexture) ivblArray.back().push<float>(2);
			ivbArray.emplace_back(vaoArray[si].data(), vaoArray[si].size() * sizeof(float));
			ivaArray.back().AddBuffer(ivbArray.back(), ivblArray.back());
			iibArray.emplace_back(iboArray[si].data(), iboArray[si].size());
			if (tiArray[si].hasTexture && si != surfaceCount -1)
			{
				shaderCodeArray.emplace_back(USE_SHADER_MIXED);
				texArray.emplace_back(tiArray[si].p_tex, tiArray[si].rows, tiArray[si].cols);
				texArray.back().bind((si <= 15) ? si : 0);
			}
			else if (tiArray[si].hasTexture && si == surfaceCount - 1)
			{
				shaderCodeArray.emplace_back(USE_SHADER_PURE_TEXTURE);
				texArray.emplace_back(tiArray[si].p_tex, tiArray[si].rows, tiArray[si].cols);
				texArray.back().bind((si <= 15) ? si : 0);
			}
			else
			{
				shaderCodeArray.emplace_back(USE_SHADER_PURE_COLOR);
				texArray.emplace_back();
			}
			shaderCaliber.setUniform("u_MVP", MVPmat);
			shaderCaliber.setUniform("u_color", 0.5f, 0.6f, 0.7f, 1.0f);

			shaderCaliber.unbind();
			ivaArray.back().unbind();
			ivbArray.back().unbind();
			iibArray.back().unbind();
		}

		//InternalShader shader(BasicShaderNoTex);
		//shader.bind();
		//shader.unbind();
		//shader.bind();
		//GLCall(glUseProgram(0));
		
		//iisArray.front().bind();
//#endif //  nothing

		//InternalShader shader(BasicShaderNoTex);

#ifdef nothing
		int si = 0;
		//setup VAO, VBO, IBO
		VertexArray va;
		VertexBufferLayout layout;
		layout.push<float>(3);
		if (tiArray[si].hasTexture) layout.push<float>(2);
		VertexBuffer vb(vaoArray[si].data(), vaoArray[si].size() * sizeof(float));
		va.AddBuffer(vb, layout);
		IndexBuffer ib(iboArray[si].data(), iboArray[si].size());

		InternalShader shader(BasicShaderNoTex);

		//Texture texture("res/textures/awesomeface.png");
		//Texture texture1("res/textures/container.jpg");

		//texture.bind(0);
		//texture1.bind(1);

		//shader.setUniform("u_textureSlot0", 0);
		//shader.setUniform("u_textureSlot1", 1);
		shader.setUniform("u_MVP", MVPmat);

		//unbind everything
		shader.unbind();
		va.unbind();//should unbind va before vb and ib
		vb.unbind();
		ib.unbind();
#endif // nothing''
		
		//create a renderer
		Renderer renderer;

		//setup the camera
		camera.reset();
		camera.useMouse(true);

		//setup IMGUI context
		ImGui::CreateContext();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		const char* glsl_version = "#version 330";
		ImGui_ImplOpenGL3_Init(glsl_version);

		//setup IMGUI variables
		ImVec4 clear_color = ImVec4(0.2f, 0.2f, 0.2f, 1.00f);
		ImVec4 gui_sourceColor = ImVec4(1.0f, 1.0f, 1.0f, 1.00f);
		std::stringstream versionText;
		versionText << "OpenGL version " << glGetString(GL_VERSION);
		glm::vec3 gui_objectTranslation(0.0f, 0.0f, 0.0f);
		glm::vec3 gui_sourceTranslation(1.2f, 1.0f, 2.0f);
		
		/* Loop until the user closes the window */
		while (!glfwWindowShouldClose(window))
		{
			//calculate frametime
			static float lastFrameTime;
			float thisFrameTime = float(glfwGetTime());
			float deltaTime = thisFrameTime - lastFrameTime;
			lastFrameTime = thisFrameTime;

			//process the inputs
			processInput(window, deltaTime);
			camera.onUpdate(window, deltaTime);

			//clear screen
			GLCall(glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w));
			renderer.clear();

			//clear depth buffer
			GLCall(glClear(GL_DEPTH_BUFFER_BIT));

			//draw IMGUI frame
			static bool discoMode = false; //have some changing color
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			{
				static float f = 0.0f;
				static bool useMouseState = true;
				//window name
				ImGui::Begin("Control Panel");
				//version text
				ImGui::Text(versionText.str().c_str());
				//window size text
				ImGui::Text("Window size: %dx%d ", m_windowWidth, m_windowHeight);
				//delta text
				ImGui::Text("Delta Time: %f ", deltaTime);
				//camera position text
				ImGui::Text("Camera position: (%f, %f, %f)", camera.m_position.x,
					camera.m_position.y, camera.m_position.z);
				ImGui::Text("Camera front: (%f, %f, %f)", camera.m_front.x,
					camera.m_front.y, camera.m_front.z);
				ImGui::Text("Camera yaw: %f, pitch: %f, fov: %f", camera.m_yaw,
					camera.m_pitch, camera.m_zoom);
				//translation slider
				ImGui::ColorEdit3("Clear color", (float*)&clear_color);
				//light source color pick
				ImGui::ColorEdit3("Light color", (float*)&gui_sourceColor);
				// Buttons return true when clicked (most widgets return true when edited/activated)
				if (ImGui::Button("Use Mouse"))
				{
					useMouseState = !useMouseState;
				}
				camera.useMouse(useMouseState);
				ImGui::SameLine();
				ImGui::Text(useMouseState ? "Yes" : "No");
				// Buttons return true when clicked (most widgets return true when edited/activated)
				if (ImGui::Button("Disco mode"))
				{
					discoMode = !discoMode;
				}
				ImGui::SameLine();
				ImGui::Text(discoMode ? "On" : "Off");
				//button for camera reset
				if (ImGui::Button("Camera Reset"))
					camera.reset();
				ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
				ImGui::End();
			}


			//set light color
			glm::vec3 lightColor;
			//color overlay
			//float redValue = (sin(glfwGetTime()) / (2.0f)) + 0.5f;
			//lightColor = glm::vec3(redValue, 1.0f, 1.0f);
			lightColor = glm::vec3(gui_sourceColor.x, gui_sourceColor.y, gui_sourceColor.z);

			//VP calculation, M depends on each object
			projmat = glm::perspective(camera.getFOV(), (float)m_windowWidth / (float)m_windowHeight, 0.01f, 10000.0f);
			/*viewmat = glm::lookAt(
				glm::vec3(0, 0, 5),
				glm::vec3(0, 0, 0),
				glm::vec3(0, 1, 0));*/
			viewmat = camera.getViewMatrix();

			//int si = 0;
			//draw the surface
#ifdef nothing
			{
				//bind shader
				shader.bind();
				
				//M Calculation
				modelmat = glm::mat4(1.0f);
				modelmat = glm::translate(modelmat, poArray[si].first);
				modelmat = glm::rotate(modelmat, poArray[si].second.z, glm::vec3(0, 0, 1));
				modelmat = glm::rotate(modelmat, poArray[si].second.y, glm::vec3(0, 1, 1));
				modelmat = glm::rotate(modelmat, poArray[si].second.x, glm::vec3(1, 0, 1));

				//set MVP matrix uniform
				MVPmat = projmat * viewmat * modelmat;
				shader.setUniform("u_MVP", MVPmat);

				//draw call
				GLCall(glCullFace(GL_FRONT));
				shader.setUniform("u_color", lightColor.x, lightColor.y, lightColor.z, 1.0f);
				renderer.draw(va, ib, shader);
				GLCall(glCullFace(GL_BACK));
				shader.setUniform("u_color", 0.8f*lightColor.x, 0.8f*lightColor.y, 0.8f*lightColor.z, 1.0f);
				renderer.draw(va, ib, shader);
			}
#endif
//#ifdef nothing
			//shader.bind();
			auto currentShaderCode = shaderCodeArray.begin();
			auto currentVA = ivaArray.begin();
			auto currentIB = iibArray.begin();
			auto currentTex = texArray.begin();
			for (int si = 0; si < surfaceCount; si++, currentShaderCode++, currentVA++, currentIB++, currentTex++)
			{
				//M Calculation
				modelmat = glm::mat4(1.0f);
				modelmat = glm::rotate(modelmat, poArray[si].second.z, glm::vec3(0, 0, 1));
				modelmat = glm::rotate(modelmat, poArray[si].second.y, glm::vec3(0, 1, 1));
				modelmat = glm::rotate(modelmat, poArray[si].second.x, glm::vec3(1, 0, 1));
				modelmat = glm::translate(modelmat, poArray[si].first);

				//set MVP matrix uniform
				MVPmat = projmat * viewmat * modelmat;
				
				//draw call
				GLCall(glCullFace(GL_FRONT));
				shaderCaliber.useShader(*currentShaderCode);
				shaderCaliber.setUniform("u_MVP", MVPmat);
				if (*currentShaderCode != USE_SHADER_PURE_TEXTURE)
				{
					if (!discoMode)
					{
						shaderCaliber.setUniform("u_color", sin(lightColor.y*(si + 1.0f)*3.3f + 1.0f), sin(lightColor.y*(si + 1.0f)*1.5f + 1.0f), sin(lightColor.z*(si + 1.0f)*2.7f + 1.0f), 1.0f);
					}
					else
					{
						shaderCaliber.setUniform("u_color", sin(lightColor.y*(si + 1.0f)*3.3f + thisFrameTime), sin(lightColor.y*(si + 1.0f)*1.5f + thisFrameTime), sin(lightColor.z*(si + 1.0f)*2.7f + thisFrameTime), 1.0f);
					}
				}
				if (tiArray[si].hasTexture) shaderCaliber.setUniform("u_textureSlot0", currentTex->getBoundSlot());
				renderer.draw(*currentVA, *currentIB, shaderCaliber.getCurrentShader());

				GLCall(glCullFace(GL_BACK));
				shaderCaliber.useShader(USE_SHADER_PURE_COLOR);
				shaderCaliber.setUniform("u_MVP", MVPmat);
				shaderCaliber.setUniform("u_color", 0.8f*lightColor.x, 0.8f*lightColor.y, 0.8f*lightColor.z, 1.0f);
				renderer.draw(*currentVA, *currentIB, shaderCaliber.getCurrentShader());
			}
//#endif
			
			
			//draw GUI on top
			ImGui::Render();
			glfwMakeContextCurrent(window);
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			glfwMakeContextCurrent(window);

			/* Swap front and back buffers */
			glfwSwapBuffers(window);
			
			/* Poll for and process events */
			glfwPollEvents();
		}
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
	windowCounter--;
	return 0;
}

void mouseScrollCallback(GLFWwindow * window, double xoffset, double yoffset)
{
	camera.scroll(float(yoffset));
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT)
	{
		if (GLFW_PRESS == action)
		{
			lbutton_down = true;
			glfwGetCursorPos(window, &lastCursorPos_x, &lastCursorPos_y);
		}
		else if (GLFW_RELEASE == action)
			lbutton_down = false;
	}
}

void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (lbutton_down)
	{
		float xoffset = float(-xpos + lastCursorPos_x);
		float yoffset = float(-lastCursorPos_y + ypos);

		lastCursorPos_x = xpos;
		lastCursorPos_y = ypos;

		camera.cursorDrag(xoffset, yoffset);
	}
}

void processInput(GLFWwindow *window, float deltaTime)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	GLCall(glViewport(0, 0, width, height));
	m_windowWidth = width;
	m_windowHeight = height;
}


