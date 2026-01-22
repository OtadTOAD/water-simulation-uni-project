#version 450

layout(location = 0) in vec2 worldUV;
layout(location = 1) in float lodScale;
layout(location = 2) in float sssScaleFactor;
layout(location = 3) in vec3 viewVector;
layout(location = 4) in vec4 screenPos;

layout(set = 0, binding = 0) uniform sampler2D displacement;
layout(set = 0, binding = 1) uniform sampler2D derivatives;
layout(set = 0, binding = 2) uniform sampler2D turbulence;
layout(set = 0, binding = 3) uniform sampler2D cameraDepthTexture;
layout(set = 0, binding = 4) uniform sampler2D foamTexture;

layout(set = 1, binding = 0) uniform OceanParams {
    float lengthScale;
    float lodScale;
    float sssBase;
    float sssScale;
} params;

layout(set = 1, binding = 1) uniform MaterialParams {
    vec4 color;
    vec4 foamColor;
    vec4 sssColor;
    float sssStrength;
    float roughness;
    float roughnessScale;
    float maxGloss;
    float foamBias;
    float foamScale;
    float contactFoam;
    float time;
    vec3 lightDir;
} material;

layout(location = 0) out vec4 outColor;

float pow5(float f) {
    return f * f * f * f * f;
}

float linearEyeDepth(float depth) {
    float near = 0.1;
    float far = 1000.0;
    return (2.0 * near) / (far + near - depth * (far - near));
}

void main() {
    vec4 derivs = texture(derivatives, worldUV / params.lengthScale);
    
    vec2 slope = vec2(
        derivs.x / (1.0 + derivs.z),
        derivs.y / (1.0 + derivs.w)
    );
    vec3 worldNormal = normalize(vec3(-slope.x, 1.0, -slope.y));
    
    // Calculate foam/turbulence (jacobian)
    float jacobian = texture(turbulence, worldUV / params.lengthScale).x;
    jacobian = clamp((-jacobian + material.foamBias) * material.foamScale, 0.0, 1.0);
    
    // Contact foam (depth-based)
    vec2 screenUV = (screenPos.xy / screenPos.w) * 0.5 + 0.5;
    float backgroundDepth = linearEyeDepth(texture(cameraDepthTexture, screenUV).r);
    float surfaceDepth = screenPos.z / screenPos.w;
    float depthDifference = max(0.0, backgroundDepth - surfaceDepth - 0.1);
    
    float foam = texture(foamTexture, worldUV * 0.5 + material.time).r;
    jacobian += material.contactFoam * clamp(max(0.0, foam - depthDifference) * 5.0, 0.0, 1.0) * 0.9;
    
    // Albedo (base color with foam)
    vec3 albedo = mix(vec3(0.0), material.foamColor.rgb, jacobian);
    
    // Smoothness/roughness calculation
    float distanceGloss = mix(
        1.0 - material.roughness,
        material.maxGloss,
        1.0 / (1.0 + length(viewVector) * material.roughnessScale)
    );
    float smoothness = mix(distanceGloss, 0.0, jacobian);
    
    // Subsurface scattering
    vec3 viewDir = normalize(viewVector);
    vec3 H = normalize(-worldNormal + material.lightDir);
    float viewDotH = pow5(clamp(dot(viewDir, -H), 0.0, 1.0)) * 30.0 * material.sssStrength;
    vec3 baseColor = clamp(material.color.rgb + material.sssColor.rgb * viewDotH * sssScaleFactor, 0.0, 1.0);
    
    // Fresnel
    float fresnel = dot(worldNormal, viewDir);
    fresnel = clamp(1.0 - fresnel, 0.0, 1.0);
    fresnel = pow5(fresnel);
    
    // Foam(basically a mask where white is foam and black is water)
    vec3 emission = mix(baseColor * (1.0 - fresnel), vec3(0.0), jacobian);
    
    // Dot diffuse light
    float ndotl = max(0.0, dot(worldNormal, material.lightDir));
    vec3 diffuse = albedo * (0.2 + ndotl * 0.8);
    
    // Specular (simplified Blinn-Phong)
    vec3 halfVec = normalize(viewDir + material.lightDir);
    float ndoth = max(0.0, dot(worldNormal, halfVec));
    float specPower = exp2(smoothness * 10.0 + 1.0);
    vec3 specular = vec3(pow(ndoth, specPower)) * smoothness;
    
    outColor = vec4(diffuse + specular + emission, 1.0);
}