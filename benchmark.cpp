#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

struct FrameStats {
  double avgGpuMs = 0.0;
  double medianGpuMs = 0.0;  // robust to the occasional slow frame
  double stddevGpuMs = 0.0;  // how noisy the measurement is
  double minGpuMs = 0.0;
  double maxGpuMs = 0.0;
  double avgFrameIntervalMs = 0.0;
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
    stats.avgFrameIntervalMs += frame.frameIntervalMs;
    stats.minGpuMs = std::min(stats.minGpuMs, frame.gpuMs);
    stats.maxGpuMs = std::max(stats.maxGpuMs, frame.gpuMs);
  }
  stats.avgGpuMs /= frames.size();
  stats.avgFrameIntervalMs /= frames.size();

  std::vector<double> sorted;
  double squaredDiffs = 0.0;
  for (const FrameTiming& frame : frames) {
    sorted.push_back(frame.gpuMs);
    squaredDiffs += (frame.gpuMs - stats.avgGpuMs) * (frame.gpuMs - stats.avgGpuMs);
  }
  std::sort(sorted.begin(), sorted.end());
  size_t middle = sorted.size() / 2;
  stats.medianGpuMs = sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2.0;
  stats.stddevGpuMs = std::sqrt(squaredDiffs / frames.size());
  return stats;
}

// Scale a measured time to the projected cube count (only meaningful when linear)
double projectMs(const BenchmarkInfo& info, double measuredMs) {
  return measuredMs * double(info.projectedCubeCount) / info.cubeCount;
}

std::string currentTimestamp() {
  std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_s(&local, &now);  // thread-safe version of std::localtime
  std::ostringstream text;
  text << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
  return text.str();
}

}  // namespace

void printBenchmarkReport(const BenchmarkInfo& info, const std::vector<FrameTiming>& frames) {
  FrameStats stats = computeStats(frames);

  double rays = double(info.width) * info.height;
  double tests = rays * info.cubeCount;
  double gpuSeconds = stats.medianGpuMs / 1000.0;
  double bufferMiB = double(info.cubeCount) * info.bytesPerCube / (1024.0 * 1024.0);

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n==================== BENCHMARK ====================\n";
  std::cout << "Step            : " << info.label << "\n";
  std::cout << "Renderer        : " << info.renderer << " (" << info.buildConfig << " build)\n";
  std::cout << "GPU             : " << info.deviceName << "\n";
  std::cout << "Resolution      : " << info.width << "x" << info.height
            << " (" << uint64_t(rays) << " primary rays)\n";
  std::cout << "Cubes           : " << info.cubeCount;
  if (info.bytesPerCube > 0) {
    std::cout << " x " << info.bytesPerCube << " B = " << bufferMiB << " MiB";
  }
  std::cout << "\n";
  if (!info.cubeMemory.empty()) {
    std::cout << "Cube memory     : " << info.cubeMemory << "\n";
  }
  if (info.workgroupSize > 0) {
    std::cout << "Workgroup       : " << info.workgroupSize << "x" << info.workgroupSize << "\n";
  }
  std::cout << "\n-- Setup (CPU wall clock) --\n";
  std::cout << "Scene generation: " << info.sceneGenMs << " ms\n";
  std::cout << "Cube prep       : " << info.cubePrepMs << " ms\n";
  std::cout << "Buffer upload   : " << info.uploadMs << " ms\n";
  std::cout << "Pipeline create : " << info.pipelineMs << " ms\n";
  std::cout << "\n-- Measured (" << frames.size() << " frames) --\n";
  std::cout << "GPU frame time  : median " << stats.medianGpuMs << " ms, avg " << stats.avgGpuMs
            << " ms\n";
  std::cout << "Spread          : stddev " << stats.stddevGpuMs << " ms ("
            << 100.0 * stats.stddevGpuMs / stats.avgGpuMs << "%), min " << stats.minGpuMs
            << ", max " << stats.maxGpuMs << "\n";
  std::cout << "Frame interval  : avg " << stats.avgFrameIntervalMs << " ms (CPU, submit to submit)\n";
  std::cout << "FPS (GPU)       : " << 1000.0 / stats.medianGpuMs << "\n";
  std::cout << "(Throughput below uses the median)\n";
  std::cout << "Rays/s          : " << rays / gpuSeconds / 1e6 << " M\n";
  if (info.linearInCubeCount) {
    std::cout << "Ray-cube tests/s: " << tests / gpuSeconds / 1e9 << " G\n";
    std::cout << "Time per test   : " << std::setprecision(4) << gpuSeconds / tests * 1e12
              << std::setprecision(2) << " ps\n";
  }
  if (info.linearInCubeCount && info.projectedCubeCount != info.cubeCount) {
    double projectedMs = projectMs(info, stats.medianGpuMs);
    std::cout << "\n-- Projected to " << info.projectedCubeCount << " cubes (linear scaling) --\n";
    std::cout << "Assumes the cost per ray-cube test stays the same. A small scene can fit in\n"
                 "the GPU's cache and a large one can't, so treat this as a lower bound.\n";
    std::cout << "GPU frame time  : ~" << projectedMs << " ms (~" << projectedMs / 1000.0
              << " s)\n";
    std::cout << "FPS (GPU)       : ~" << std::setprecision(4) << 1000.0 / projectedMs
              << std::setprecision(2) << "\n";
  }
  std::cout << "===================================================\n" << std::endl;
}

void appendBenchmarkCsv(const std::string& path, const BenchmarkInfo& info,
                        const std::vector<FrameTiming>& frames) {
  FrameStats stats = computeStats(frames);
  double rays = double(info.width) * info.height;
  double gpuSeconds = stats.medianGpuMs / 1000.0;

  bool writeHeader = !std::ifstream(path).good();
  std::ofstream file(path, std::ios::app);
  if (!file) {
    std::cerr << "Could not write benchmark CSV to " << path << std::endl;
    return;
  }
  if (writeHeader) {
    file << "timestamp,label,renderer,build,gpu,width,height,cubes,bytes_per_cube,"
            "avg_gpu_ms,median_gpu_ms,stddev_gpu_ms,min_gpu_ms,max_gpu_ms,avg_frame_interval_ms,"
            "mrays_per_s,gtests_per_s,"
            "projected_cubes,projected_gpu_ms,scene_gen_ms,cube_prep_ms,upload_ms,pipeline_ms\n";
  }

  // Columns that only make sense for brute force are left empty otherwise
  std::string testsPerSecond;
  std::string projectedCubes;
  std::string projectedGpuMs;
  if (info.linearInCubeCount) {
    std::ostringstream value;
    value << std::fixed << std::setprecision(3);
    value << rays * info.cubeCount / gpuSeconds / 1e9;
    testsPerSecond = value.str();
    projectedCubes = std::to_string(info.projectedCubeCount);
    value.str("");
    value << projectMs(info, stats.medianGpuMs);
    projectedGpuMs = value.str();
  }

  file << std::fixed << std::setprecision(3)
       << currentTimestamp() << "," << info.label << "," << info.renderer << ","
       << info.buildConfig << "," << info.deviceName << ","
       << info.width << "," << info.height << "," << info.cubeCount << ","
       << info.bytesPerCube << ","
       << stats.avgGpuMs << "," << stats.medianGpuMs << "," << stats.stddevGpuMs << ","
       << stats.minGpuMs << "," << stats.maxGpuMs << ","
       << stats.avgFrameIntervalMs << "," << rays / gpuSeconds / 1e6 << ","
       << testsPerSecond << "," << projectedCubes << "," << projectedGpuMs << ","
       << info.sceneGenMs << "," << info.cubePrepMs << "," << info.uploadMs << ","
       << info.pipelineMs << "\n";
}
