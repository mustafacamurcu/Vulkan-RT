#pragma once
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "benchmark.h"
#include "brute_force_material.h"
#include "rt_material.h"
#include "structs.h"
#include "volkan.h"
#include "window.h"

enum class RenderMode {
  HardwareRayTracing,  // BLAS/TLAS + ray tracing pipeline (driver builds a BVH)
  BruteForceCompute    // compute shader, every pixel tests every cube
};

class Engine {
 public:
  Engine();
  ~Engine();

  // GLFW holds a pointer to this object, so it must never be copied or moved
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  void run();
  void handleFramebufferResized();

  // ---- Configuration (compile-time constants, one copy shared by all instances) ----
  static constexpr uint32_t WIDTH = 1920;
  static constexpr uint32_t HEIGHT = 1080;

  static constexpr RenderMode RENDER_MODE = RenderMode::BruteForceCompute;
  static constexpr const char* BENCHMARK_LABEL = "00-brute-force-10k";
  static constexpr uint32_t BENCHMARK_WARMUP_FRAMES = 2;
  static constexpr uint32_t BENCHMARK_TIMED_FRAMES = 30;

  // Brute force over 1M cubes takes ~65 s per frame, far past the ~2 s Windows
  // GPU timeout for a single dispatch. Render fewer cubes and project the cost.
  static constexpr uint32_t NUM_CUBES = 10000;
  static constexpr uint32_t PROJECTED_CUBE_COUNT = 1000000;

  // The full scene: 1M cubes in a box of half extent 200. Smaller cube counts
  // shrink the box so the cube density (and so the look of the scene) stays the same.
  static constexpr float FULL_SCENE_HALF_EXTENT = 200.0f;
  inline static const glm::vec3 SCENE_CENTER{50.0f, 0.0f, 0.0f};

 private:
  void createRayTracingResources();
  void createBruteForceResources();
  std::vector<glm::mat4> createRandomCubeTransforms(uint32_t count) const;
  static float sceneHalfExtent();

  void drawFrame();
  void recordFrame(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex);
  void collectFrameTiming(uint32_t frameIndex);
  void finishBenchmark();

  void updateCameraFront();
  void processInput();

  float yaw_ = 90.0f;
  float pitch_ = 40.0f;

  // Declaration order matters: members are destroyed in reverse, so the window
  // outlives Vulkan, Vulkan outlives every GPU resource, and the materials
  // (which reference the buffers) are destroyed first.
  Window window_;
  Volkan volkan_;
  Camera camera_;

  // Hardware ray tracing geometry
  struct Vertex {
    float x, y, z;
  };
  BufferData cubeVertexBuffer_;
  BufferData cubeIndexBuffer_;
  BufferData blasBuffer_;
  vk::raii::AccelerationStructureKHR blas_ = nullptr;
  BufferData tlasBuffer_;
  vk::raii::AccelerationStructureKHR tlas_ = nullptr;

  // Brute force geometry
  BufferData cubeBuffer_;

  // Benchmarking: a pair of GPU timestamps per frame in flight
  vk::raii::QueryPool timestampQueryPool_ = nullptr;
  bool timestampsPending_[MAX_FRAMES_IN_FLIGHT] = {};
  double frameIntervalMs_[MAX_FRAMES_IN_FLIGHT] = {};
  std::chrono::steady_clock::time_point lastSubmitTime_;
  uint32_t framesMeasured_ = 0;
  std::vector<FrameTiming> frameTimings_;
  BenchmarkInfo benchmarkInfo_;
  bool benchmarkDone_ = false;

  std::unique_ptr<RayTracingMaterial> rtMaterial_;
  std::unique_ptr<BruteForceMaterial> bruteForceMaterial_;
};
