#version 450

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform Camera {
    mat4 invProj;
    mat4 invView;
    vec3 camPos;
} cam;

void main() {
    vec3 ro = cam.camPos;
    vec3 rd = (cam.invProj * vec4(uv*2.-1., 0, 1)).xyz;
    rd = (cam.invView * vec4(rd, 0)).xyz;
    rd = normalize(rd);

    outColor = vec4(rd*0.5 + 0.5, 1.0);
}