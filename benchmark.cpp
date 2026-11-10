#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

FrameStats computeStats(const std::vector<FrameTiming>& frames) {
  FrameStats stats;
  stats.frames = frames.size();
  if (frames.empty()) {
    return stats;
  }

  std::vector<double> sorted;
  for (const FrameTiming& frame : frames) {
    sorted.push_back(frame.gpuMs);
    stats.avgGpuMs += frame.gpuMs;
    stats.avgFrameIntervalMs += frame.frameIntervalMs;
  }
  stats.avgGpuMs /= frames.size();
  stats.avgFrameIntervalMs /= frames.size();

  std::sort(sorted.begin(), sorted.end());
  size_t middle = sorted.size() / 2;
  stats.medianGpuMs = sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2.0;
  stats.minGpuMs = sorted.front();
  stats.maxGpuMs = sorted.back();

  double squaredDiffs = 0.0;
  for (double gpuMs : sorted) {
    squaredDiffs += (gpuMs - stats.avgGpuMs) * (gpuMs - stats.avgGpuMs);
  }
  stats.stddevGpuMs = std::sqrt(squaredDiffs / frames.size());
  return stats;
}

uint64_t hashBytes(const std::vector<uint8_t>& bytes) {
  uint64_t hash = 14695981039346656037ull;
  for (uint8_t byte : bytes) {
    hash = (hash ^ byte) * 1099511628211ull;
  }
  return hash;
}

ImageDiff compareImages(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  ImageDiff diff;
  for (size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4) {
    bool same = a[i] == b[i] && a[i + 1] == b[i + 1] && a[i + 2] == b[i + 2];
    if (same) {
      continue;
    }
    diff.differentPixels++;
    bool aBlack = a[i] == 0 && a[i + 1] == 0 && a[i + 2] == 0;
    bool bBlack = b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 0;
    if (aBlack != bBlack) {
      diff.hitChangedPixels++;
    }
  }
  return diff;
}

namespace {

std::string currentTimestamp() {
  std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_s(&local, &now);  // thread-safe version of std::localtime
  std::ostringstream text;
  text << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
  return text.str();
}

const char* CSV_HEADER =
    "timestamp,label,renderer,build,gpu,width,height,cubes,bytes_per_cube,"
    "avg_gpu_ms,median_gpu_ms,stddev_gpu_ms,min_gpu_ms,max_gpu_ms,avg_frame_interval_ms,"
    "mrays_per_s,gtests_per_s,projected_cubes,projected_gpu_ms,scene_gen_ms,cube_prep_ms,"
    "upload_ms,pipeline_ms,view,grid_resolution,grid_entries,image_hash";

}  // namespace

void appendBenchmarkCsv(const std::string& path, const BenchmarkInfo& info, const FrameStats& stats) {
  bool writeHeader = !std::ifstream(path).good();
  std::ofstream file(path, std::ios::app);
  if (!file) {
    std::cerr << "Could not write benchmark CSV to " << path << std::endl;
    return;
  }
  if (writeHeader) {
    file << CSV_HEADER << "\n";
  }

  double rays = double(info.width) * info.height;
  std::ostringstream hash;
  hash << std::hex << std::setw(16) << std::setfill('0') << info.imageHash;

  // gtests_per_s and the projection columns only applied to brute force: left empty
  file << std::fixed << std::setprecision(3)
       << currentTimestamp() << "," << info.label << "," << info.renderer << ","
       << info.buildConfig << "," << info.deviceName << ","
       << info.width << "," << info.height << "," << info.cubeCount << ","
       << info.bytesPerEntry << ","
       << stats.avgGpuMs << "," << stats.medianGpuMs << "," << stats.stddevGpuMs << ","
       << stats.minGpuMs << "," << stats.maxGpuMs << "," << stats.avgFrameIntervalMs << ","
       << rays / (stats.medianGpuMs / 1000.0) / 1e6 << ",,,,"
       << info.sceneGenMs << "," << info.gridBuildMs << "," << info.uploadMs << ","
       << info.pipelineMs << "," << info.view << "," << info.gridResolution << ","
       << info.gridEntries << "," << hash.str() << "\n";
}
