#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Timing for one rendered frame
struct FrameTiming {
  double gpuMs = 0.0;            // GPU timestamps around the render work
  double frameIntervalMs = 0.0;  // CPU time since the previous frame was submitted
};

struct FrameStats {
  size_t frames = 0;
  double avgGpuMs = 0.0;
  double medianGpuMs = 0.0;  // robust to the occasional slow frame
  double stddevGpuMs = 0.0;  // how noisy the measurement is
  double minGpuMs = 0.0;
  double maxGpuMs = 0.0;
  double avgFrameIntervalMs = 0.0;
};

FrameStats computeStats(const std::vector<FrameTiming>& frames);

// Everything that identifies one benchmark configuration, so rows stay comparable
struct BenchmarkInfo {
  std::string label;
  std::string renderer;  // e.g. grid-rejection, grid-traversal
  std::string view;      // e.g. dense-inside
  std::string buildConfig;
  std::string deviceName;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t cubeCount = 0;
  uint32_t bytesPerEntry = 0;
  uint32_t gridResolution = 0;
  uint64_t gridEntries = 0;  // (cube, cell) pairs stored in the grid
  uint64_t imageHash = 0;    // hash of the rendered image (identical images, identical hash)

  double sceneGenMs = 0.0;   // generating the random cubes on the CPU
  double gridBuildMs = 0.0;  // sorting the cubes into the grid on the CPU
  double uploadMs = 0.0;     // copying the grid to VRAM
  double pipelineMs = 0.0;   // shader module + pipeline creation
};

void appendBenchmarkCsv(const std::string& path, const BenchmarkInfo& info, const FrameStats& stats);

// FNV-1a hash of a byte array: any change in the pixels changes it
uint64_t hashBytes(const std::vector<uint8_t>& bytes);

// How two RGBA8 images of the same size differ
struct ImageDiff {
  uint64_t differentPixels = 0;   // any color change
  uint64_t hitChangedPixels = 0;  // black in one, a cube in the other: a hit was gained or lost
};
ImageDiff compareImages(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b);
