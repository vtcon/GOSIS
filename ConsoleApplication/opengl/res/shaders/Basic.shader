#shader vertex
#version 330 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texCoor;

uniform mat4 u_MVP;

out vec2 v_texCoor;

void main()
{
	gl_Position = u_MVP * position;
	v_texCoor = texCoor;
};

#shader fragment
#version 330 core

layout(location = 0) out vec4 color;

in vec2 v_texCoor;

uniform vec4 u_color;
uniform sampler2D u_textureSlot0;
uniform sampler2D u_textureSlot1;

void main()
{
	vec4 texColor = mix(texture(u_textureSlot0, v_texCoor), texture(u_textureSlot1, v_texCoor), 0.8);
	color = texColor*u_color;
};