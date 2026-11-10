#version 460

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float time;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

float freq = 3.0f;
float speed = 10.0f;
float amp = .50f;

void main() {
    float waveHeight = sin(inPosition.x * freq + ubo.time * speed + inPosition.y * freq) * amp;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition.x, waveHeight, inPosition.y, 1.0);
    fragColor = inColor;
}