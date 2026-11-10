#pragma once
#include <GLFW/glfw3.h>
#include "volkan.h"
#include "structs.h"
#include "rt_material.h"
#include "brute_force_material.h"
#include "benchmark.h"

enum class RenderMode {
  HardwareRayTracing,  // BLAS/TLAS + ray tracing pipeline (driver builds a BVH)
  BruteForceCompute    // compute shader, every pixel tests every cube
};

class Engine {
 public:
  Engine();
  ~Engine();

  void run();

  void handleFramebufferResized();

  float yaw = 90.0f;
  float pitch = 40.0f;

  const uint32_t WIDTH = 1920;
  const uint32_t HEIGHT = 1080;

  // Which renderer to run, and how to label its benchmark results
  const RenderMode RENDER_MODE = RenderMode::BruteForceCompute;
  const char* BENCHMARK_LABEL = "00-brute-force-baseline";
  const uint32_t BENCHMARK_WARMUP_FRAMES = 1;
  const uint32_t BENCHMARK_TIMED_FRAMES = 3;

  // Each tile is a separate GPU submission so no single one hits the ~2s Windows TDR
  const uint32_t TILE_SIZE = 256;

 private:
  void createWindow(int width, int height, const char* title);

  void createRayTracingResources();
  void createBruteForceResources();
  std::vector<glm::mat4> createRandomCubeTransforms(uint32_t count);

  bool renderBruteForceFrame(FrameTiming& timing);
  void runBenchmark();

  void updateCameraFront();
  void processInput(GLFWwindow* window);
  void drawFrame();

  std::unique_ptr<RayTracingMaterial> rtMaterial_;
  std::unique_ptr<BruteForceMaterial> bruteForceMaterial_;

  // Ray tracing geometry
  struct Vertex {
    float x, y, z;
  };
  BufferData cubeVertexBuffer_;
  BufferData cubeIndexBuffer_;

  // Random cube scene
  const uint32_t NUM_CUBES = 1000000;
  const glm::vec3 SCENE_CENTER{50.0f, 0.0f, 0.0f};
  const float SCENE_HALF_EXTENT = 200.0f;
  vk::raii::AccelerationStructureKHR blas_ = nullptr;
  BufferData blasBuffer_;
  vk::raii::AccelerationStructureKHR tlas_ = nullptr;
  BufferData tlasBuffer_;

  // Brute force compute resources
  BufferData cubeBuffer_;
  vk::raii::QueryPool timestampQueryPool_ = nullptr;
  BenchmarkInfo benchmarkInfo_;

  GLFWwindow* window_;
  Volkan volkan_;

  Camera camera_;

};
