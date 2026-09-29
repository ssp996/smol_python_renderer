#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

uniform mat4 model;
uniform mat4 projection;

out vec3 frag_color;
out vec3 frag_normal;

void main()
{
    gl_Position = projection * model * vec4(position, 1.0);

    frag_color = color;
    frag_normal = normal;
}