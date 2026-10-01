#version 330 core

in vec2 uv;

uniform sampler2D tex2D;
uniform samplerCube texCube;
uniform int mode;                
uniform vec3 faceForward;
uniform vec3 faceRight;
uniform vec3 faceUp;
uniform float scale;

out vec4 outColor;

void main()
{
    float d;

    if (mode == 0)
        d = texture(tex2D, uv * 0.5 + 0.5).r;
    else
        d = texture(texCube, faceForward + uv.x * faceRight + uv.y * faceUp).r;

    outColor = vec4(vec3(clamp(d * scale, 0.0, 1.0)), 1.0);
}