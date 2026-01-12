#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 uv;

/* WARNING: FOR INTANCE RENDERING, UNCOMMENT BELOW
layout(location = 4) in mat4 instance_model;
layout(location = 8) in mat4 instance_normal;
*/

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_tex_coords;
layout(location = 2) out vec3 out_world_pos;
layout(location = 3) out vec4 out_tangent;

layout(set = 0, binding = 0) uniform Camera {
    mat4 invProj;
    mat4 invView;
    vec3 camPos;
} cam;

void main() {
    vec4 world_pos = vec4(position, 1.0);
    //instance_model * vec4(position, 1.0); This is for instance rendering
    gl_Position    = cam.invProj * cam.invView * world_pos;

    out_normal     = normal;
    //normalize(mat3(instance_normal) * normal); This is for instance rendering
    out_tangent    =  tangent;
    //vec4(normalize(mat3(instance_normal) * tangent.xyz), tangent.w); This is for instance rendering
    out_tex_coords = uv;
    out_world_pos  = world_pos.xyz;
}