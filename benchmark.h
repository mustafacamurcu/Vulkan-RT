#pragma once
#include <string>
#include <vector>

// Timing for one fully rendered frame
struct FrameTiming {
  double gpuMs = 0.0;          // sum of GPU timestamps over all tiles
  double wallMs = 0.0;         // CPU wall clock for the whole frame (includes submit/wait overhead)
  double maxTileGpuMs = 0.0;   // slowest single tile (must stay well under the ~2s Windows TDR limit)
  uint32_t tileCount = 0;
};

// Everything needed to describe a benchmark run so steps can be compared
struct BenchmarkInfo {
  std::string label;
  std::string deviceName;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t cubeCount = 0;
  uint32_t bytesPerCube = 0;
  std::string cubeMemory;
  uint32_t tileSize = 0;
  uint32_t workgroupSize = 0;
  double sceneGenMs = 0.0;     // generating random transforms on the CPU
  double cubePrepMs = 0.0;     // converting transforms into the GPU layout
  double uploadMs = 0.0;       // creating + filling the cube buffer
  double pipelineMs = 0.0;     // shader module + pipeline creation
};

void printBenchmarkReport(const BenchmarkInfo& info, const std::vector<FrameTiming>& frames);
void appendBenchmarkCsv(const std::string& path, const BenchmarkInfo& info,
                        const std::vector<FrameTiming>& frames);
