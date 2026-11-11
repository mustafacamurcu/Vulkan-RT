#pragma once
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "benchmark.h"
#include "grid.h"
#include "grid_material.h"
#include "rt_material.h"
#include "triangle_scene.h"
#include "structs.h"
#include "volkan.h"
#include "window.h"

// How each hit is colored (must match SHADING_* in grid_traversal.comp)
enum class ShadingMode : uint32_t {
  Flat = 0,     // the cube's color, no lighting
  Diffuse = 1,  // Lambert lighting from a directional light (no extra rays)
  Shadows = 2,  // diffuse + a shadow ray toward the light from every lit hit
};

// Command-line options. No options = open the viewer on the sparse scene.
struct StartupConfig {
  bool benchmark = false;       // --benchmark: run the benchmark suite instead of the viewer
  std::string scene;            // --scene sparse|quadrants (viewer only)
  uint32_t gridResolution = 0;  // --grid N (viewer only; 0 = the scene's default)
  std::string shading;          // --shading flat|diffuse|shadows (viewer only)
};

enum class RenderMode {
  HardwareRayTracing,  // BLAS/TLAS + ray tracing pipeline (driver builds a BVH); no benchmark
  GridCompute          // compute shader over a uniform grid; runs the benchmark suite
};

class Engine {
 public:
  explicit Engine(const StartupConfig& startup = {});
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
  static constexpr const char* BENCHMARK_LABEL = "07-quadrants-shading";

  struct CameraView {
    const char* name;
    glm::vec3 position;
    float yaw;    // degrees
    float pitch;  // degrees (positive looks down: -y is up)
  };

  // A scene: where the cubes are, where the camera starts, and its best grid size
  struct SceneConfig {
    const char* name;
    std::vector<CubeRegion> regions;
    CameraView camera;
    uint32_t defaultGridResolution;
  };

  // Viewer default: 1M cubes thinly spread through a 2000-unit box, seen from
  // just outside it. Most rays pass between cubes, so you see deep into the cloud.
  inline static const SceneConfig SPARSE_SCENE = {
    "sparse",
    {{"sparse", {-1000.0f, -1000.0f, -1000.0f}, {1000.0f, 1000.0f, 1000.0f}, 1000000}},
    {"sparse-afar", {0.0f, -800.0f, -4000.0f}, 90.0f, 11.3f},
    96,
  };

  // Benchmark scene: one box split into four quadrants (on x and z), each a
  // different challenge for the ray tracer. 1M cubes in total. The camera is high
  // above, looking down at ~55 degrees, so the tops of all four are in view.
  inline static const SceneConfig QUADRANT_SCENE = {
    "quadrants",
    {
      // Like the old dense scene: rays hit something within ~17 units
      {"dense", {-250.0f, -125.0f, -250.0f}, {0.0f, 125.0f, 0.0f}, 300000},
      // Mostly empty: rays cross many cells and often escape without a hit
      {"sparse", {0.0f, -125.0f, -250.0f}, {250.0f, 125.0f, 0.0f}, 10000},
      // 20 tight clusters of 20k cubes: very uneven density
      {"clustered", {-250.0f, -125.0f, 0.0f}, {0.0f, 125.0f, 250.0f}, 400000, 20, 12.0f},
      // Sizes from 0.2 to 8 (log-uniform): mostly tiny cubes, a few big ones that
      // span many grid cells and cast large shadows
      {"mixed-sizes", {0.0f, -125.0f, 0.0f}, {250.0f, 125.0f, 250.0f}, 290000, 0, 0.0f, 0.2f, 8.0f, true},
    },
    {"quadrants", {0.0f, -650.0f, -450.0f}, 90.0f, 53.5f},
    64,
  };

  // Viewer controls
  static constexpr float MOVE_SPEED = 100.0f;      // units per second for the cube scenes (x5 with Shift)

  // Amazon Lumberyard Bistro (exterior), CC-BY 4.0, from NVIDIA ORCA. Not in git:
  // download it into assets/bistro/ (see OPTIMIZING_RT.md).
  static constexpr const char* BISTRO_PATH = "assets/bistro/BistroExterior.fbx";
  static constexpr const char* TRIANGLE_SHADER = "shaders/triangle_traversal.spv";
  static constexpr uint32_t BISTRO_GRID_RESOLUTION = 256;  // cells along the longest side
  static constexpr float TURN_SPEED = 60.0f;       // degrees per second (arrow keys)

  // Benchmark sweep: every shading mode at every grid resolution
  struct ShadingConfig {
    const char* name;
    ShadingMode mode;
  };
  inline static const ShadingConfig SHADING_MODES[] = {
    {"flat", ShadingMode::Flat},
    {"diffuse", ShadingMode::Diffuse},
    {"shadows", ShadingMode::Shadows},
  };
  inline static const std::vector<uint32_t> GRID_RESOLUTIONS = {32, 48, 64, 96, 128};
  static constexpr const char* GRID_SHADER = "shaders/grid_traversal.spv";
  static constexpr ShadingMode DEFAULT_SHADING = ShadingMode::Shadows;

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
  void loadViewer();
  void loadBistroViewer(ShadingMode shading);
  void setCameraLookAt(const glm::vec3& position, const glm::vec3& target);
  void setView(const CameraView& view);
  void loadGrid(const CubeScene& scene, uint32_t resolution, BenchmarkInfo& info);
  void setShading(ShadingMode mode);
  bool measureFrames(std::vector<FrameTiming>& timings);  // false if the window was closed
  void discardPendingTimings();

  void drawFrame();
  void recordFrame(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex);
  GridPushConstants makeGridPushConstants() const;
  void collectFrameTiming(uint32_t frameIndex);

  void updateCameraFront();
  void processInput();

  StartupConfig startup_;
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

  // The triangle scene currently on the GPU (Bistro), when triangleMode_ is set
  bool triangleMode_ = false;
  BufferData triangleBuffer_;  // 3 vec4 per triangle
  BufferData entryBuffer_;     // triangle indices, grouped by cell
  BufferData materialBuffer_;  // color per material
  glm::uvec3 gridDims_{0};     // cells per axis (triangle grid)
  float moveSpeed_ = MOVE_SPEED;
  glm::vec4 gridMin_{0.0f};  // xyz = grid's min corner, w = cell size
  uint32_t gridResolution_ = 0;
  ShadingMode shading_ = DEFAULT_SHADING;
  std::string gridLabel_;    // perf marker name for the dispatch, e.g. "grid-traversal N=64 shadows"

  // Benchmarking: a pair of GPU timestamps per frame in flight
  vk::raii::QueryPool timestampQueryPool_ = nullptr;
  bool timestampsPending_[MAX_FRAMES_IN_FLIGHT] = {};
  double frameIntervalMs_[MAX_FRAMES_IN_FLIGHT] = {};
  std::chrono::steady_clock::time_point lastSubmitTime_;
  bool collecting_ = false;
  uint32_t framesSeen_ = 0;
  std::vector<FrameTiming> collected_;
  std::chrono::steady_clock::time_point lastInputTime_;

  std::unique_ptr<RayTracingMaterial> rtMaterial_;
  std::unique_ptr<GridMaterial> gridMaterial_;
};
