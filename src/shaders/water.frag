#version 450

layout(location = 0) in vec2 worldUV;
layout(location = 1) in float lodScale;
layout(location = 2) in float sssScaleFactor;
layout(location = 3) in vec3 viewVector;
layout(location = 4) in vec4 screenPos;

const int CASCADES = 3;

layout(set = 0, binding = 0) uniform sampler2D displacement[CASCADES];
layout(set = 0, binding = 1) uniform sampler2D derivatives[CASCADES];
layout(set = 0, binding = 2) uniform sampler2D turbulence[CASCADES];
layout(set = 0, binding = 3) uniform sampler2D cameraDepthTexture;
layout(set = 0, binding = 4) uniform sampler2D foamTexture;
layout(set = 0, binding = 5) uniform sampler2D skyTexture;

layout(set = 1, binding = 0) uniform OceanParams {
    vec4 lengthScales; // world-space patch size of each cascade (xyz used)
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

const float PI = 3.14159265359;

float pow5(float f) {
    return f * f * f * f * f;
}

vec2 equirectUV(vec3 dir) {
    return vec2(
        atan(dir.z, dir.x) / (2.0 * PI) + 0.5,
        acos(clamp(dir.y, -1.0, 1.0)) / PI
    );
}

// Must match SKY_EXPOSURE in sky.frag
const float SKY_EXPOSURE = 0.05;
// Ambient light level for the water's own (non-reflected) shading.
const float WATER_LIGHT = 0.5;

vec3 sampleSky(vec3 dir, float lod) {
    vec3 c = textureLod(skyTexture, equirectUV(dir), lod).rgb * SKY_EXPOSURE;
    c = c / (c + vec3(1.0));
    return pow(c, vec3(1.0 / 2.2));
}

float linearEyeDepth(float depth) {
    float near = 0.1;
    float far = 1000.0;
    return (2.0 * near) / (far + near - depth * (far - near));
}

void main() {
    float viewDist = length(viewVector);

    // Average multiple cascades since now we don't have only 1
    vec4 derivs = vec4(0.0);
    float jacobian = 0.0;
    for (int c = 0; c < CASCADES; c++) {
        float scale = params.lengthScales[c];
        vec2 uv = worldUV / scale;
        float fade = min(params.lodScale * scale / viewDist, 1.0);
        derivs += texture(derivatives[c], uv) * fade;
        jacobian += texture(turbulence[c], uv).x;
    }
    jacobian /= float(CASCADES);

    vec2 slope = vec2(
        derivs.x / (1.0 + derivs.z),
        derivs.y / (1.0 + derivs.w)
    );
    vec3 worldNormal = normalize(vec3(-slope.x, 1.0, -slope.y));

    // Calculate foam/turbulence (jacobian)
    jacobian = clamp((-jacobian + material.foamBias) * material.foamScale, 0.0, 1.0);
    
    // Contact foam (depth-based)
    vec2 screenUV = (screenPos.xy / screenPos.w) * 0.5 + 0.5;
    float backgroundDepth = linearEyeDepth(texture(cameraDepthTexture, screenUV).r);
    float surfaceDepth = screenPos.z / screenPos.w;
    float depthDifference = max(0.0, backgroundDepth - surfaceDepth - 0.1);
    
    float foam = texture(foamTexture, worldUV * 0.5 + material.time).r;
    jacobian += material.contactFoam * clamp(max(0.0, foam - depthDifference) * 5.0, 0.0, 1.0) * 0.9;
    
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
    
    // Fresnel (Schlick) water has a low base reflectance (~0.02).
    float ndotv = max(dot(worldNormal, viewDir), 0.0);
    float fresnel = 0.02 + 0.98 * pow5(1.0 - ndotv);

    // Reflect the sky environment off the surface. Fold the ray back up so
    // grazing reflections never sample the (empty) lower hemisphere.
    vec3 reflDir = reflect(-viewDir, worldNormal);
    reflDir.y = abs(reflDir.y);
    vec3 skyReflection = sampleSky(reflDir, 0.0);

    // Blend the deep-water body colour (with SSS tint) and the reflected sky.
    // The body colour is dimmed to the night ambient; the reflection already
    // carries the sky's own (dark) brightness.
    vec3 surface = mix(baseColor * WATER_LIGHT, skyReflection, fresnel);

    // Specular sun highlight (simplified Blinn-Phong).
    vec3 halfVec = normalize(viewDir + material.lightDir);
    float ndoth = max(0.0, dot(worldNormal, halfVec));
    float specPower = exp2(smoothness * 10.0 + 1.0);
    vec3 specular = vec3(pow(ndoth, specPower)) * smoothness * WATER_LIGHT;

    // Foam sits on top as a simple diffuse-lit layer.
    float ndotl = max(0.0, dot(worldNormal, material.lightDir));
    vec3 foamLit = material.foamColor.rgb * (0.3 + ndotl * 0.7) * WATER_LIGHT;

    vec3 color = mix(surface + specular, foamLit, jacobian);

    outColor = vec4(color, 1.0);
}