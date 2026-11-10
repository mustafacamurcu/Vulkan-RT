#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan_raii.hpp>

#define NUM_WAVES 10

#define MAX_FRAMES_IN_FLIGHT 2

struct BufferData {
  vk::raii::Buffer buffer = nullptr;
  vk::raii::DeviceMemory memory = nullptr;
  void* mapped = nullptr;

  BufferData(vk::raii::Buffer& _buffer, vk::raii::DeviceMemory& _memory,
             void* mapped_)
      : buffer(std::move(_buffer)),
        memory(std::move(_memory)),
        mapped(mapped_) {}

  BufferData() : buffer(nullptr), memory(nullptr), mapped(nullptr) {}
};

struct Camera {
  glm::vec3 pos{50.0, -70.0, -80.0};
  glm::vec3 front{0.0, 0.0, 0.0};
  glm::vec3 up{0.0f, -1.0f, 0.0f};

  glm::mat4 getViewMatrix() const {
    return glm::lookAt(pos, pos + front, up); }

  glm::mat4 getProjectionMatrix(float aspectRatio) const {
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
    proj[1][1] *= -1; // Invert Y for Vulkan
    return proj;
  }
};

  
struct PerFrameVariables {
  glm::float32 time;
  alignas(16) glm::mat4 view;
  glm::mat4 proj;
  glm::vec4 cameraPos;
};

// Inverse camera matrices so the raygen shader can turn pixels into rays
struct RayTracingPushConstants {
  glm::mat4 invView;
  glm::mat4 invProj;
};

// Camera + tile info for the brute force compute ray tracer (96 bytes)
struct BruteForcePushConstants {
  glm::mat4 invViewProj;
  glm::vec4 cameraPos;
  glm::uvec4 tileInfo;  // xy = tile offset in pixels, z = cube count, w = unused
};

struct alignas(32) Wave {
  float amp;
  float phase;
  float dir_x;
  float dir_z;
  float freq;
  float sharpness;
};

struct Waves {
  Wave waves[NUM_WAVES];
};