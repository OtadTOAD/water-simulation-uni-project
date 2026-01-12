#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_tex_coords;
layout(location = 2) in vec3 in_world_pos;
layout(location = 3) in vec4 in_tangent;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple debug shader to visualize normals
    vec3 colorBlue = vec3(35, 137, 218) / 255.0;
    outColor = vec4(colorBlue, 1.0);
}