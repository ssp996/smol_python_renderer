#version 330 core

in vec3 fragColor;
in vec3 fragNormal;
in vec3 fragPos;

uniform vec3 light_pos;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 norm = normalize(fragNormal);

    vec3 lightDir = normalize(light_pos - fragPos);

    float diff = max(dot(norm, lightDir), 0.0);
    
    float ambient = 0.1;
    vec3 final_light = fragColor * (diff + ambient);

    outColor = vec4(final_light, 1.0);
}