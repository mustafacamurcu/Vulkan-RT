#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan_raii.hpp>

// Typed, scoped compile-time constant (unlike a #define)
inline constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

// A buffer and the memory backing it. Move-only, because the vk::raii handles
// inside can't be copied. Build one with BufferData{std::move(buffer), std::move(memory)}
// so the transfer of ownership is visible at the call site.
struct BufferData {
  vk::raii::Buffer buffer = nullptr;
  vk::raii::DeviceMemory memory = nullptr;
};

struct Camera {
  glm::vec3 pos{50.0, -70.0, -80.0};
  glm::vec3 front{0.0, 0.0, 0.0};
  glm::vec3 up{0.0f, -1.0f, 0.0f};

  glm::mat4 getViewMatrix() const {
    return glm::lookAt(pos, pos + front, up);
  }

  glm::mat4 getProjectionMatrix(float aspectRatio) const {
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
    proj[1][1] *= -1; // Invert Y for Vulkan
    return proj;
  }
};

// Inverse camera matrices so the raygen shader can turn pixels into rays
struct RayTracingPushConstants {
  glm::mat4 invView;
  glm::mat4 invProj;
};

// Camera + scene info for the compute ray tracers. 128 bytes: the most push
// constant space every Vulkan GPU is guaranteed to have.
struct GridPushConstants {
  glm::mat4 invViewProj;
  glm::vec4 cameraPos;
  glm::uvec4 params;  // x = unused, y = shading mode, z = unused, w = grid resolution
  glm::vec4 gridMin;  // xyz = grid's min corner, w = cell size
  glm::vec4 gridPad;  // unused (was the cell padding of the removed grid rejection method)
};
static_assert(sizeof(GridPushConstants) <= 128, "push constants over the guaranteed limit");
