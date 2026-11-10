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

  static constexpr const char* BENCHMARK_LABEL = "04-grid-dda";

  // Cubes are stored in a GRID_RESOLUTION^3 grid that each ray walks with 3D-DDA.
  // Kept at 16 (tuned for the previous step) so this step measures only the traversal.
  static constexpr uint32_t GRID_RESOLUTION = 16;

  static constexpr uint32_t NUM_CUBES = 1000000;
  // Brute force results with fewer cubes are scaled up to this count (lower bound)
  static constexpr uint32_t PROJECTED_CUBE_COUNT = 1000000;

  // TILE_SIZE > 0 renders the frame in TILE_SIZE x TILE_SIZE tiles, one submission
  // each, for frames too slow for the ~2 s Windows GPU timeout (brute force took
  // tens of seconds). 0 = one dispatch per frame. Tiled mode ignores input, so its
  // images are deterministic: use it for pixel-exact correctness checks.
  static constexpr uint32_t TILE_SIZE = 0;
  static constexpr bool BRUTE_FORCE_TILED =
      RENDER_MODE == RenderMode::BruteForceCompute && TILE_SIZE > 0;

  // A tiled frame already averages over ~40 tiles, so one is enough (no warm-up)
  static constexpr uint32_t BENCHMARK_WARMUP_FRAMES = BRUTE_FORCE_TILED ? 0 : 2;
  static constexpr uint32_t BENCHMARK_TIMED_FRAMES = BRUTE_FORCE_TILED ? 1 : 30;

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
  BruteForcePushConstants makeBruteForcePushConstants() const;
  bool renderTiledFrame(FrameTiming& timing);
  void runTiledBenchmark();
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
  // One entry per (cube, overlapped cell) pair, sorted by cell
  BufferData cubeBuffer_;    // inverse transform (64 B)
  BufferData sphereBuffer_;  // bounding sphere (16 B)
  BufferData cellBuffer_;    // where each cell's entries start (GRID_RESOLUTION^3 + 1)
  BufferData cubeIdBuffer_;  // original cube index (4 B), for its color
  glm::vec4 gridMin_{0.0f};  // xyz = grid's min corner, w = cell size

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
