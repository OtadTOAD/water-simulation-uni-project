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

const float SKY_EXPOSURE = 0.05;
vec3 tonemap(vec3 c) {
    // Reinhard tonemap + gamma correction for the HDR sky.
    c *= SKY_EXPOSURE;
    c = c / (c + vec3(1.0));
    return pow(c, vec3(1.0 / 2.2));
}

void main() {
    vec3 dir = normalize(viewRay);

    vec3 ddx = dFdx(dir);
    vec3 ddy = dFdy(dir);

    // Compute uvs from direction
    float r2 = max(dir.x * dir.x + dir.z * dir.z, 1e-8);
    float du_dx = (dir.x * ddx.z - dir.z * ddx.x) / r2 / (2.0 * PI);
    float du_dy = (dir.x * ddy.z - dir.z * ddy.x) / r2 / (2.0 * PI);

    // v = acos(y)/PI, so dv = -dy / (PI * sqrt(1 - y^2)).
    float s = max(sqrt(1.0 - dir.y * dir.y), 1e-4);
    float dv_dx = -ddx.y / (PI * s);
    float dv_dy = -ddy.y / (PI * s);

    vec2 uv = equirectUV(dir);
    vec2 gradX = vec2(du_dx, dv_dx);
    vec2 gradY = vec2(du_dy, dv_dy);

    vec3 color = textureGrad(skyTexture, uv, gradX, gradY).rgb;
    outColor = vec4(tonemap(color), 1.0);
}
