#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 2) in mat4 instance_model;
layout(location = 6) in mat4 instance_normal;

layout(location = 0) out vec4 out_albedo;

layout(set = 0, binding = 0) uniform Camera {
    mat4 proj;
    mat4 view;
    vec3 camPos;
} cam;

void main() {
    vec4 world_pos = instance_model * vec4(position, 1.0);
    gl_Position    = cam.proj * cam.view * world_pos;

    out_albedo = vec4(1.0);
}