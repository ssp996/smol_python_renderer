#version 330 core

#define MAX_DIR_LIGHTS 2
#define MAX_POINT_LIGHTS 4

struct DirLight {
    vec3 direction;
    vec3 color;
    float intensity;
    mat4 lightSpaceMatrix;
};

struct PointLight {
    vec3 position;
    vec3 color;
    float intensity;
    float farPlane;
};

in vec3 fragColor;
in vec3 fragNormal;
in vec3 fragPos;

uniform vec3 cameraPos;

uniform int activeDirLights;
uniform DirLight dirLights[MAX_DIR_LIGHTS];
uniform sampler2D dirShadowMaps[MAX_DIR_LIGHTS];

uniform int activePointLights;
uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform samplerCube pointShadowMaps[MAX_POINT_LIGHTS];

layout(location = 0) out vec4 outColor;

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir, sampler2D shadowMap)
{
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec4 fragPosLightSpace = light.lightSpaceMatrix * vec4(fragPos, 1.0);
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float shadow = 0.0;
    if(projCoords.z <= 1.0) {
        float closestDepth = texture(shadowMap, projCoords.xy).r; 
        float currentDepth = projCoords.z;
        float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.001);
        shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    }
    
    return light.color * light.intensity * (diff * (1.0 - shadow));
}

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, samplerCube shadowMap)
{
    vec3 lightDir = normalize(light.position - fragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * (distance * distance));    
    

    float cosTheta = clamp(dot(normal, lightDir), 0.0, 1.0);
    vec3 samplePos = fragPos + normal * (0.05 + 0.1 * (1.0 - cosTheta));
    vec3 fragToLight = samplePos - light.position;
    float closestDepth = texture(shadowMap, fragToLight).r;
    closestDepth *= light.farPlane; 
    float currentDepth = length(fragToLight);

    float bias = max(0.1 * (1.0 - cosTheta), 0.03);
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;

    return light.color * light.intensity * attenuation * (diff * (1.0 - shadow));
}

void main()
{
    vec3 norm = normalize(fragNormal);
    vec3 viewDir = normalize(cameraPos - fragPos);
    
    vec3 result = vec3(0.1); 

    for(int i = 0; i < activeDirLights; i++)
        result += CalcDirLight(dirLights[i], norm, viewDir, dirShadowMaps[i]);
        
    for(int i = 0; i < activePointLights; i++)
        result += CalcPointLight(pointLights[i], norm, fragPos, viewDir, pointShadowMaps[i]);

    outColor = vec4(result * fragColor, 1.0);
}