#version 450

layout(location = 0) in vec3 viewRay;

layout(set = 0, binding = 0) uniform sampler2D skyTexture;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

vec2 equirectUV(vec3 dir) {
    return vec2(
        atan(dir.z, dir.x) / (2.0 * PI) + 0.5,
        acos(clamp(dir.y, -1.0, 1.0)) / PI
    );
}

vec3 tonemap(vec3 c) {
    // Reinhard tonemap + gamma correction for the HDR sky.
    c = c / (c + vec3(1.0));
    return pow(c, vec3(1.0 / 2.2));
}

void main() {
    vec3 dir = normalize(viewRay);
    vec3 color = texture(skyTexture, equirectUV(dir)).rgb;
    outColor = vec4(tonemap(color), 1.0);
}
