#include "benchmark.h"

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {

struct FrameStats {
  double avgGpuMs = 0.0;
  double minGpuMs = 0.0;
  double maxGpuMs = 0.0;
  double avgWallMs = 0.0;
  double maxTileGpuMs = 0.0;
};

FrameStats computeStats(const std::vector<FrameTiming>& frames) {
  FrameStats stats;
  if (frames.empty()) {
    return stats;
  }
  stats.minGpuMs = frames[0].gpuMs;
  stats.maxGpuMs = frames[0].gpuMs;
  for (const FrameTiming& frame : frames) {
    stats.avgGpuMs += frame.gpuMs;
    stats.avgWallMs += frame.wallMs;
    stats.minGpuMs = std::min(stats.minGpuMs, frame.gpuMs);
    stats.maxGpuMs = std::max(stats.maxGpuMs, frame.gpuMs);
    stats.maxTileGpuMs = std::max(stats.maxTileGpuMs, frame.maxTileGpuMs);
  }
  stats.avgGpuMs /= frames.size();
  stats.avgWallMs /= frames.size();
  return stats;
}

}  // namespace

void printBenchmarkReport(const BenchmarkInfo& info, const std::vector<FrameTiming>& frames) {
  FrameStats stats = computeStats(frames);

  double rays = double(info.width) * info.height;
  double tests = rays * info.cubeCount;
  double gpuSeconds = stats.avgGpuMs / 1000.0;
  double bufferMiB = double(info.cubeCount) * info.bytesPerCube / (1024.0 * 1024.0);
  // Every thread in a 32-wide warp reads the same cube at the same time, so the
  // hardware fetches it once per warp. This is the cube data the warps request.
  double warpBytes = (rays / 32.0) * info.cubeCount * info.bytesPerCube;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n==================== BENCHMARK ====================\n";
  std::cout << "Step            : " << info.label << "\n";
  std::cout << "GPU             : " << info.deviceName << "\n";
  std::cout << "Resolution      : " << info.width << "x" << info.height
            << " (" << uint64_t(rays) << " primary rays)\n";
  std::cout << "Cubes           : " << info.cubeCount << " x " << info.bytesPerCube
            << " B = " << bufferMiB << " MiB\n";
  std::cout << "Cube memory     : " << info.cubeMemory << "\n";
  std::cout << "Tiles           : " << info.tileSize << "x" << info.tileSize
            << " px, workgroup " << info.workgroupSize << "x" << info.workgroupSize << "\n";
  std::cout << "\n-- Setup (CPU wall clock) --\n";
  std::cout << "Scene generation: " << info.sceneGenMs << " ms\n";
  std::cout << "Cube prep       : " << info.cubePrepMs << " ms\n";
  std::cout << "Buffer upload   : " << info.uploadMs << " ms\n";
  std::cout << "Pipeline create : " << info.pipelineMs << " ms\n";
  std::cout << "\n-- Frames --\n";
  for (size_t i = 0; i < frames.size(); i++) {
    std::cout << "Frame " << i + 1 << "         : GPU " << frames[i].gpuMs << " ms, wall "
              << frames[i].wallMs << " ms, " << frames[i].tileCount << " tiles, slowest tile "
              << frames[i].maxTileGpuMs << " ms\n";
  }
  std::cout << "\n-- Summary (" << frames.size() << " timed frames) --\n";
  std::cout << "GPU frame time  : avg " << stats.avgGpuMs << " ms (min " << stats.minGpuMs
            << ", max " << stats.maxGpuMs << ")\n";
  std::cout << "Wall frame time : avg " << stats.avgWallMs << " ms (submit/wait overhead "
            << stats.avgWallMs - stats.avgGpuMs << " ms)\n";
  std::cout << "FPS (GPU)       : " << std::setprecision(4) << 1000.0 / stats.avgGpuMs
            << std::setprecision(2) << "\n";
  std::cout << "Slowest tile    : " << stats.maxTileGpuMs << " ms (Windows TDR ~2000 ms)\n";
  std::cout << "Rays/s          : " << rays / gpuSeconds / 1e6 << " M\n";
  std::cout << "Ray-cube tests/s: " << tests / gpuSeconds / 1e9 << " G\n";
  std::cout << "Time per test   : " << std::setprecision(4) << gpuSeconds / tests * 1e12
            << std::setprecision(2) << " ps\n";
  std::cout << "Cube data/warp  : " << warpBytes / gpuSeconds / 1e9 << " GB/s requested\n";
  std::cout << "===================================================\n" << std::endl;
}

void appendBenchmarkCsv(const std::string& path, const BenchmarkInfo& info,
                        const std::vector<FrameTiming>& frames) {
  FrameStats stats = computeStats(frames);
  double rays = double(info.width) * info.height;
  double gpuSeconds = stats.avgGpuMs / 1000.0;

  bool writeHeader = !std::ifstream(path).good();
  std::ofstream file(path, std::ios::app);
  if (!file) {
    std::cerr << "Could not write benchmark CSV to " << path << std::endl;
    return;
  }
  if (writeHeader) {
    file << "timestamp,label,gpu,width,height,cubes,bytes_per_cube,tile_size,"
            "avg_gpu_ms,min_gpu_ms,max_gpu_ms,avg_wall_ms,max_tile_gpu_ms,"
            "mrays_per_s,gtests_per_s,scene_gen_ms,cube_prep_ms,upload_ms,pipeline_ms\n";
  }

  std::time_t now = std::time(nullptr);
  char timeText[32];
  std::strftime(timeText, sizeof(timeText), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

  file << std::fixed << std::setprecision(3)
       << timeText << "," << info.label << "," << info.deviceName << ","
       << info.width << "," << info.height << "," << info.cubeCount << ","
       << info.bytesPerCube << "," << info.tileSize << ","
       << stats.avgGpuMs << "," << stats.minGpuMs << "," << stats.maxGpuMs << ","
       << stats.avgWallMs << "," << stats.maxTileGpuMs << ","
       << rays / gpuSeconds / 1e6 << "," << rays * info.cubeCount / gpuSeconds / 1e9 << ","
       << info.sceneGenMs << "," << info.cubePrepMs << "," << info.uploadMs << ","
       << info.pipelineMs << "\n";
}
