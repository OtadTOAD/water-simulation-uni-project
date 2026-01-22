#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 2) in mat4 instance_model;
layout(location = 6) in mat4 instance_normal;

layout(set = 0, binding = 0) uniform sampler2D noise_texture;

layout(location = 0) out vec4 out_albedo;

layout(push_constant) uniform Camera {
    mat4 proj;
    mat4 view;
    // Pretty sure mat4 is 64 bytes so two mat4 fit in 128 bytes limit
    // vec3 camPos; Removed cam pos to fit in push_constant size limits(128 bytes)
} cam;

void main() {
    vec3 noise = texture(noise_texture, uv).rgb;

    vec4 world_pos = instance_model * vec4(position, 1.0);
    gl_Position    = cam.proj * cam.view * world_pos;

    out_albedo = vec4(noise, 1.0);
}