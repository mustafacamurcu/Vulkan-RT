#pragma once
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "benchmark.h"
#include "grid.h"
#include "grid_material.h"
#include "rt_material.h"
#include "structs.h"
#include "volkan.h"
#include "window.h"

enum class RenderMode {
  HardwareRayTracing,  // BLAS/TLAS + ray tracing pipeline (driver builds a BVH); no benchmark
  GridCompute          // compute shader over a uniform grid; runs the benchmark suite
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
  static constexpr RenderMode RENDER_MODE = RenderMode::GridCompute;
  static constexpr const char* BENCHMARK_LABEL = "05-views-and-grid-sweep";

  static constexpr uint32_t NUM_CUBES = 1000000;
  inline static const glm::vec3 SCENE_CENTER{50.0f, 0.0f, 0.0f};

  // Benchmark viewpoints. The cost of grid traversal depends on how far rays
  // travel before hitting something, so it is measured in an easy, a medium
  // and a hard case.
  struct BenchmarkView {
    const char* name;
    float sceneHalfExtent;  // cubes fill a box of this half size around SCENE_CENTER
    glm::vec3 cameraPos;
    float yaw;    // degrees
    float pitch;  // degrees
  };
  inline static const BenchmarkView VIEWS[] = {
    // Camera inside the cloud: rays hit something within ~17 units
    {"dense-inside", 200.0f, {50.0f, -70.0f, -80.0f}, 90.0f, 40.0f},
    // Camera outside, the whole cloud in view: rays cross empty space, then hit its surface
    {"dense-outside", 200.0f, {50.0f, 0.0f, -800.0f}, 90.0f, 0.0f},
    // Same 1M cubes in a 125x larger volume: about half the rays escape without a hit
    {"sparse-inside", 1000.0f, {50.0f, -70.0f, -80.0f}, 90.0f, 40.0f},
  };

  // Each method and the grid resolutions to sweep. Rejection tests every cell
  // for every ray, so large grids get slow quickly; traversal can go much finer.
  struct GridMethodConfig {
    const char* name;
    const char* shader;
    GridLayout layout;
    std::vector<uint32_t> resolutions;
  };
  inline static const GridMethodConfig METHODS[] = {
    {"grid-rejection", "shaders/grid_rejection.spv", GridLayout::ByCenter, {8, 12, 16, 24, 32}},
    {"grid-traversal", "shaders/grid_traversal.spv", GridLayout::ByOverlap,
     {8, 16, 24, 32, 48, 64, 96, 128}},
  };

  // Per configuration: warm-up frames, then timed frames until either limit is reached
  static constexpr uint32_t WARMUP_FRAMES = 2;
  static constexpr uint32_t MIN_TIMED_FRAMES = 5;
  static constexpr uint32_t MAX_TIMED_FRAMES = 30;
  static constexpr double MAX_TIMED_GPU_MS = 3000.0;

  // Inverse transform + sphere + id per grid entry; skip grids larger than this
  static constexpr uint32_t BYTES_PER_ENTRY = sizeof(glm::mat4) + sizeof(glm::vec4) + sizeof(uint32_t);
  static constexpr uint64_t MAX_GRID_BYTES = 1ull << 30;

 private:
  void createRayTracingResources();

  void runBenchmarkSuite();
  void setView(const BenchmarkView& view);
  void loadGrid(const CubeScene& scene, const GridMethodConfig& method, uint32_t resolution,
                BenchmarkInfo& info);
  bool measureFrames(std::vector<FrameTiming>& timings);  // false if the window was closed
  void discardPendingTimings();

  void drawFrame();
  void recordFrame(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex);
  GridPushConstants makeGridPushConstants() const;
  void collectFrameTiming(uint32_t frameIndex);

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

  // The grid currently on the GPU: one entry per (cube, cell) pair, sorted by cell
  BufferData cubeBuffer_;    // inverse transform (64 B)
  BufferData sphereBuffer_;  // bounding sphere (16 B)
  BufferData cellBuffer_;    // where each cell's entries start (resolution^3 + 1)
  BufferData cubeIdBuffer_;  // original cube index (4 B), for its color
  glm::vec4 gridMin_{0.0f};  // xyz = grid's min corner, w = cell size
  float gridPad_ = 0.0f;     // how much to grow each cell when testing it (rejection only)
  uint32_t gridResolution_ = 0;

  // Benchmarking: a pair of GPU timestamps per frame in flight
  vk::raii::QueryPool timestampQueryPool_ = nullptr;
  bool timestampsPending_[MAX_FRAMES_IN_FLIGHT] = {};
  double frameIntervalMs_[MAX_FRAMES_IN_FLIGHT] = {};
  std::chrono::steady_clock::time_point lastSubmitTime_;
  bool collecting_ = false;
  uint32_t framesSeen_ = 0;
  std::vector<FrameTiming> collected_;
  bool benchmarkDone_ = false;

  std::unique_ptr<RayTracingMaterial> rtMaterial_;
  std::unique_ptr<GridMaterial> gridMaterial_;
};
