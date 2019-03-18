#include <iostream>
#include "glew.h"
#include "glfw3.h"

#include <fstream>
#include <string>
#include <sstream>

#include "Renderer.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArray.h"

#include "Shader.h"
#include "Texture.h"
#include "Renderer.h";
#include "Camera.h"
#include "InternalShader.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

static Camera camera; //it must be a global variable, as function callbacks depends on it

static bool lbutton_down = false;
static double lastCursorPos_x = 0;
static double lastCursorPos_y = 0;

static void mouseScrollCallback(GLFWwindow * window, double xoffset, double yoffset);
static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
static void processInput(GLFWwindow *window, float deltaTime);

static int m_windowWidth = 1280;
static int m_windowHeight = 720;
static int m_windowDepth = 720;

/************************Shader Sources*************************************/

static std::string BasicShader = "#shader vertex\n\
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

/*************************End of Shader Sources*****************************/


int GLmain(void)
{
	GLFWwindow* window;

	/* Initialize the library */
	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(m_windowWidth, m_windowHeight, "Hello World", NULL, NULL);
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

	std::cout << glGetString(GL_VERSION) << std::endl;
	{
		float positions[] = {
			-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
			 0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
			 0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
			 0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
			-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
			-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

			-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
			 0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
			 0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
			 0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
			-0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
			-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

			-0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
			-0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
			-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
			-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
			-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
			-0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

			 0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
			 0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
			 0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
			 0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
			 0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
			 0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

			-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
			 0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
			 0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
			 0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
			-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
			-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

			-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
			 0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
			 0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
			 0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
			-0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
			-0.5f,  0.5f, -0.5f,  0.0f, 1.0f
		};

		unsigned int indices[] = {
			0, 1, 2,
			3, 4, 5,
			6, 7, 8,
			9, 10, 11,
			12, 13, 14,
			15, 16, 17,
			18, 19, 20,
			21, 22, 23,
			24, 25, 26,
			27, 28, 29,
			30, 31, 32,
			33, 34, 35
		};

		//setup blending
		GLCall(glEnable(GL_BLEND));
		GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
		GLCall(glBlendEquation(GL_FUNC_ADD));

		//setup Depth testing
		GLCall(glEnable(GL_DEPTH_TEST));

		//setup VAO, VBO, IBO
		VertexArray va;
		VertexBufferLayout layout;
		layout.push<float>(3);
		layout.push<float>(2);
		VertexBuffer vb(positions, sizeof(positions));
		va.AddBuffer(vb, layout);
		IndexBuffer ib(indices, sizeof(indices)/sizeof(unsigned int));
		
		/*glm::mat4 projimat = glm::ortho(-(float)(m_windowWidth/2), (float)(m_windowWidth / 2), 
			-(float)(m_windowHeight / 2), (float)(m_windowHeight / 2),
			-(float)(m_windowDepth / 2), (float)(m_windowDepth / 2));*/
		glm::mat4 projmat = glm::perspective(glm::radians(45.0f), (float)m_windowWidth / (float)m_windowHeight, 0.1f, 100.0f);
		glm::mat4 viewmat = glm::lookAt(
			glm::vec3(0, 0, 5),
			glm::vec3(0, 0, 0),
			glm::vec3(0, 1, 0)
		);
		glm::mat4 modelmat = glm::mat4(1.0f);

		glm::mat4 MVPmat = projmat * viewmat * modelmat;

		//Shader shader("res/shaders/Basic.shader");
		InternalShader shader(BasicShader);
		Shader lightshader("res/shaders/SimpleLightSource.shader");

		Texture texture("res/textures/awesomeface.png");
		Texture texture1("res/textures/container.jpg");

		texture.bind(0);
		texture1.bind(1);

		shader.setUniform("u_textureSlot0", 0);
		shader.setUniform("u_textureSlot1", 1);
		shader.setUniform("u_MVP", MVPmat);

		//unbind everything
		shader.unbind();
		va.unbind();//should unbind va before vb and ib
		vb.unbind();
		ib.unbind();

		//create a renderer
		Renderer renderer;

		//setup IMGUI context
		ImGui::CreateContext();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		const char* glsl_version = "#version 330";
		ImGui_ImplOpenGL3_Init(glsl_version);

		//setup IMGUI variables
		ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
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
			float thisFrameTime = glfwGetTime();
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
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			{
				static float f = 0.0f;
				static bool useMouseState = false;
				//window name
				ImGui::Begin("Debug Window");
				//version text
				ImGui::Text(versionText.str().c_str());
				//window size text
				ImGui::Text("Window size: %dx%d ", m_windowWidth, m_windowHeight);
				//delta text
				ImGui::Text("Delta Time: %f ",deltaTime);
				//camera position text
				ImGui::Text("Camera position: (%f, %f, %f)", camera.m_position.x, 
					camera.m_position.y, camera.m_position.z);
				ImGui::Text("Camera yaw: %f, pitch: %f, fov: %f", -camera.m_yaw-90.0f,
					-camera.m_pitch, camera.m_zoom);
				//translation slider
				ImGui::SliderFloat3("Object Translation", &(gui_objectTranslation.x), -4.0f, 4.0f);
				//translation slider
				ImGui::SliderFloat3("Light Source Translation", &(gui_sourceTranslation.x), -4.0f, 4.0f);
				//clear color pick, will update next frame
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
				ImGui::Text(useMouseState? "Yes" : "No");
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
			projmat = glm::perspective(camera.getFOV(), (float)m_windowWidth / (float)m_windowHeight, 0.1f, 100.0f);
			/*viewmat = glm::lookAt(
				glm::vec3(0, 0, 5),
				glm::vec3(0, 0, 0),
				glm::vec3(0, 1, 0));*/
			viewmat = camera.getViewMatrix();

			//draw the object box
			{

				//bind shader
				shader.bind();
				shader.setUniform("u_color", lightColor.x, lightColor.y, lightColor.z, 1.0f);

				//M Calculation
				modelmat = glm::mat4(1.0f);
				modelmat = glm::translate(modelmat, gui_objectTranslation);
				modelmat = glm::rotate(modelmat, (float)glfwGetTime(), glm::vec3(1, 0, 0));

				//set MVP matrix uniform
				MVPmat = projmat * viewmat * modelmat;
				shader.setUniform("u_MVP", MVPmat);

				//draw call
				renderer.draw(va, ib, shader);
			}

			//draw the light source
			{
				lightshader.bind();
				lightshader.setUniform("u_color", lightColor.x, lightColor.y, lightColor.z, 1.0f);

				//M Calculation
				modelmat = glm::mat4(1.0f);
				modelmat = glm::translate(modelmat, gui_sourceTranslation);
				modelmat = glm::scale(modelmat, glm::vec3(0.2f));

				//set MVP matrix uniform
				MVPmat = projmat * viewmat * modelmat;
				lightshader.setUniform("u_MVP", MVPmat);

				//draw call
				renderer.draw(va, ib, lightshader);
			}
			//draw GUI on top
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

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
	return 0;
}

static void mouseScrollCallback(GLFWwindow * window, double xoffset, double yoffset) 
{
	camera.scroll(yoffset);
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
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

static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (lbutton_down)
	{
		float xoffset = - xpos + lastCursorPos_x;
		float yoffset = - lastCursorPos_y + ypos;

		lastCursorPos_x = xpos;
		lastCursorPos_y = ypos;

		camera.cursorDrag(xoffset, yoffset);
	}
}

static void processInput(GLFWwindow *window, float deltaTime)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	GLCall(glViewport(0, 0, width, height));
	m_windowWidth = width;
	m_windowHeight = height;
}

bool runTestOpenGL = false;
void GLtest()
{
	GLmain();
}

