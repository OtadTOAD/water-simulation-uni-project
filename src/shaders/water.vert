#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in mat4 instance_model;
layout(location = 6) in mat4 instance_normal;
layout(location = 10) in vec4 lod_morph;

const int CASCADES = 3;

layout(set = 0, binding = 0) uniform sampler2D displacement[CASCADES];
layout(set = 0, binding = 1) uniform sampler2D derivatives[CASCADES];
layout(set = 0, binding = 2) uniform sampler2D turbulence[CASCADES];

layout(set = 1, binding = 0) uniform OceanParams {
    vec4 lengthScales; // world-space patch size of each cascade (xyz used)
    float lodScale;
    float sssBase;
    float sssScale;
} params;

layout(push_constant) uniform Camera {
    mat4 proj;
    mat4 view;
    vec3 pos;
} cam;

layout(location = 0) out vec2 worldUV;
layout(location = 1) out float lodScale;
layout(location = 2) out float sssScaleFactor;
layout(location = 3) out vec3 viewVector;
layout(location = 4) out vec4 screenPos;

void main() {
    vec4 worldPos = instance_model * vec4(position, 1.0);

    float morphDist = max(abs(worldPos.x - cam.pos.x), abs(worldPos.z - cam.pos.z));
    float morphK = clamp((morphDist - lod_morph.x) / (lod_morph.y - lod_morph.x), 0.0, 1.0);
    float cell = lod_morph.z;
    vec2 fracPart = fract(worldPos.xz / cell * 0.5) * 2.0 * cell;
    worldPos.xz -= fracPart * morphK;

    worldUV = worldPos.xz;

    viewVector = cam.pos - worldPos.xyz;
    float viewDist = length(viewVector);

    // Sum the displacement of every cascade.
    // Take LOD into account to avoid aliasing
    vec3 displacementVec = vec3(0.0);
    for (int c = 0; c < CASCADES; c++) {
        float scale = params.lengthScales[c];
        float fade = min(params.lodScale * scale / viewDist, 1.0);
        displacementVec += textureLod(displacement[c], worldUV / scale, 0).xyz * fade;
        if (c == 0) {
            lodScale = fade;
        }
    }
    worldPos.xyz += displacementVec;

    sssScaleFactor = max(displacementVec.y - params.sssBase, 0.0) / params.sssScale;
    
    gl_Position = cam.proj * cam.view * worldPos;
    gl_Position.z += lod_morph.w * 0.0001 * gl_Position.w;

    screenPos = gl_Position;
}