#version 330 core

uniform vec4 rect;   

out vec2 uv;

void main()
{
    vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));

    uv = c * 2.0 - 1.0;
    
    gl_Position = vec4(rect.xy + c * rect.zw, 0.0, 1.0);
}