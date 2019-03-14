#include "Camera.h"

Camera::Camera(glm::vec3 position, glm::vec3 front, glm::vec3 up)
	:m_position(position), m_front(front), m_up(up)
{
	m_worldUp = m_up;
	update();
}

Camera::~Camera()
{
}

void Camera::reset()
{
	m_position = glm::vec3(0.0f, 0.0f, 0.0f);
	m_front = glm::vec3(0.0f, 0.0f, -1.0f);
	m_up = glm::vec3(0.0f, 1.0f, 0.0f);
	m_worldUp = m_up;
	m_zoom = 45.0f;
	
	m_yaw = -90.0f;
	m_pitch = 0.0f;

	update();
}

void Camera::onUpdate(GLFWwindow * window, float deltaTime)
{
	float movementStep = m_movementSpeed * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		m_position += movementStep * m_front;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		m_position -= movementStep * m_front;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		m_position -= m_right * movementStep;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		m_position += m_right * movementStep;
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		m_position += m_up * movementStep;
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		m_position -= m_up * movementStep;

	update();
}

void Camera::cursorDrag(float xoffset, float yoffset, GLboolean constrainPitch)
{
	if (m_useMouse)
	{
		xoffset *= m_mouseSensitivity;
		yoffset *= m_mouseSensitivity;

		m_yaw += xoffset;
		m_pitch += yoffset;

		// Make sure that when pitch is out of bounds, screen doesn't get flipped
		if (constrainPitch)
		{
			if (m_pitch > 89.0f)
				m_pitch = 89.0f;
			if (m_pitch < -89.0f)
				m_pitch = -89.0f;
		}

		// Update Front, Right and Up Vectors using the updated Euler angles
		update();
	}
}

void Camera::scroll(double yoffset)
{
	if (m_zoom >= 1.0f && m_zoom <= 45.0f)
		m_zoom -= yoffset;
	if (m_zoom <= 1.0f)
		m_zoom = 1.0f;
	if (m_zoom >= 45.0f)
		m_zoom = 45.0f;

	//update();
}

void Camera::useMouse(bool usemouse)
{
	m_useMouse = usemouse;
}

float Camera::getFOV()
{
	return glm::radians(m_zoom);
}

glm::mat4 Camera::getViewMatrix()
{
	return glm::lookAt(m_position, m_position + m_front, m_up);
}

void Camera::update()
{
	glm::vec3 newfront;
	newfront.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
	newfront.y = sin(glm::radians(m_pitch));
	newfront.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
	m_front = glm::normalize(newfront);
	// Also re-calculate the Right and Up vector
	m_right = glm::normalize(glm::cross(m_front, m_worldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
	m_up = glm::normalize(glm::cross(m_right, m_front));
}
