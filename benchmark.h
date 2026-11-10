#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Timing for one rendered frame
struct FrameTiming {
  double gpuMs = 0.0;            // GPU timestamps around the render work
  double frameIntervalMs = 0.0;  // CPU time since the previous frame was submitted
};

// Everything needed to describe a benchmark run so steps can be compared
struct BenchmarkInfo {
  std::string label;
  std::string renderer;
  std::string buildConfig;
  std::string deviceName;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t cubeCount = 0;
  uint32_t bytesPerCube = 0;
  std::string cubeMemory;
  uint32_t workgroupSize = 0;
  uint32_t tileSize = 0;  // 0 = one dispatch per frame

  // Brute force cost grows linearly with cube count, so a run with fewer
  // cubes can be scaled up to estimate the full scene. Not valid for a BVH,
  // whose cost grows roughly with log(cube count).
  bool linearInCubeCount = false;
  uint32_t projectedCubeCount = 0;

  double sceneGenMs = 0.0;     // generating random transforms on the CPU
  double cubePrepMs = 0.0;     // inverse transforms (brute force) or BLAS/TLAS build (hardware RT)
  double uploadMs = 0.0;       // creating + filling the cube buffer
  double pipelineMs = 0.0;     // shader module + pipeline creation
};

void printBenchmarkReport(const BenchmarkInfo& info, const std::vector<FrameTiming>& frames);
void appendBenchmarkCsv(const std::string& path, const BenchmarkInfo& info,
                        const std::vector<FrameTiming>& frames);
