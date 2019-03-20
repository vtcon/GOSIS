#pragma once
#include "glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Camera
{
public:
	// Camera Attributes
	glm::vec3 m_position = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 m_front = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 m_up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 m_right;
	glm::vec3 m_worldUp;
	float m_zoom = 45.0f;
	// Euler Angles
	float m_yaw = 0.0f;
	float m_pitch = 0.0f;
	// Camera options
	float m_movementSpeed = 25.0f;
	float m_mouseSensitivity = 0.1f;
	float m_zoomStep = 0.5;
	bool m_useMouse = false;

	Camera(glm::vec3 position = glm::vec3(200.0f, 0.0f, 0.0f),
		glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
		float yaw = -180.0f, float pitch = 0.0f);
	~Camera();
	void reset();
	void onUpdate(GLFWwindow *window, float deltaTime);
	void cursorDrag(float xoffset, float yoffset, GLboolean constrainPitch = true);
	void scroll(float yoffset);
	void useMouse(bool usemouse);

	float getFOV();
	glm::mat4 getViewMatrix();

private:
	void update();
};