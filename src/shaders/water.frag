#version 450

layout(location = 0) in vec4 in_albedo;

layout(location = 0) out vec4 outColor;

void main() {
    //vec3 colorBlue = vec3(35, 137, 218) / 255.0;
    outColor = in_albedo;
}