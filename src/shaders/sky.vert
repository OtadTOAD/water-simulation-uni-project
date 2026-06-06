#version 450

layout(push_constant) uniform SkyCamera {
    mat4 invViewProj;
    vec3 pos;
} cam;

layout(location = 0) out vec3 viewRay;

void main() {
    // Fullscreen triangle generated from gl_VertexIndex (no vertex buffer)
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 ndc = p * 2.0 - 1.0;

    // Place at the far plane so the sky sits behind all geometry.
    gl_Position = vec4(ndc, 1.0, 1.0);

    vec4 world = cam.invViewProj * vec4(ndc, 1.0, 1.0);
    viewRay = world.xyz / world.w - cam.pos;
}
