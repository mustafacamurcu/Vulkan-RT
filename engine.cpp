#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

// steady_clock never jumps (unlike system time), which is what you want for measuring
using Clock = std::chrono::steady_clock;

double msBetween(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double msSince(Clock::time_point start) {
  return msBetween(start, Clock::now());
}

void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
  auto engine = static_cast<Engine*>(glfwGetWindowUserPointer(window));
  engine->handleFramebufferResized();
}

}  // namespace

Engine::Engine() : window_(WIDTH, HEIGHT, "Vulkan") {
  glfwSetWindowUserPointer(window_.handle(), this);
  glfwSetFramebufferSizeCallback(window_.handle(), framebufferResizeCallback);

  volkan_.init(window_.handle());
  setView(VIEWS[0]);
  timestampQueryPool_ = volkan_.createTimestampQueryPool(2 * MAX_FRAMES_IN_FLIGHT);

  if (RENDER_MODE == RenderMode::HardwareRayTracing) {
    createRayTracingResources();
  }
}

Engine::~Engine() {
  // The GPU may still be using resources from the last frames. Everything
  // else is released by member destructors, in reverse declaration order.
  volkan_.device().waitIdle();
}

void Engine::createRayTracingResources() {
  CubeScene scene = createCubeScene(NUM_CUBES, SCENE_CENTER, VIEWS[0].sceneHalfExtent);

  // Unit cube centered at the origin
  std::vector<Vertex> vertices = {
    {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
    {-0.5f, -0.5f, 0.5f},  {0.5f, -0.5f, 0.5f},  {0.5f, 0.5f, 0.5f},  {-0.5f, 0.5f, 0.5f}
  };

  // 2 triangles per face, 6 faces
  std::vector<uint32_t> indices = {
    0, 2, 1, 0, 3, 2,  // -Z
    4, 5, 6, 4, 6, 7,  // +Z
    0, 4, 7, 0, 7, 3,  // -X
    1, 2, 6, 1, 6, 5,  // +X
    0, 1, 5, 0, 5, 4,  // -Y
    3, 7, 6, 3, 6, 2   // +Y
  };

  vk::BufferUsageFlags geometryUsage =
    vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
  cubeVertexBuffer_ = volkan_.createDeviceLocalBuffer(
    vertices.data(), sizeof(Vertex) * vertices.size(), geometryUsage);
  cubeIndexBuffer_ = volkan_.createDeviceLocalBuffer(
    indices.data(), sizeof(uint32_t) * indices.size(), geometryUsage);

  // One cube BLAS shared by every TLAS instance
  blas_ = volkan_.createBLAS(blasBuffer_,
                             volkan_.getBufferAddress(cubeVertexBuffer_),
                             static_cast<uint32_t>(vertices.size()),
                             volkan_.getBufferAddress(cubeIndexBuffer_),
                             static_cast<uint32_t>(indices.size()));
  tlas_ = volkan_.createTLAS(tlasBuffer_, blas_, scene.transforms);

  rtMaterial_ = std::make_unique<RayTracingMaterial>(volkan_, tlas_, volkan_.storageImageView());
}

// ---------------------------------------------------------------------------
// Benchmark suite: every view x method x grid resolution
// ---------------------------------------------------------------------------

void Engine::runBenchmarkSuite() {
  struct Result {
    const char* view;
    const char* method;
    uint32_t resolution;
    FrameStats stats;
  };
  std::vector<Result> results;
  ImageDiff largestDiff;

  std::cout << "Benchmark suite: " << std::size(VIEWS) << " views, " << NUM_CUBES << " cubes, "
            << volkan_.extent().width << "x" << volkan_.extent().height << "\n"
            << std::left << std::setw(15) << "view" << std::setw(16) << "method" << std::right
            << std::setw(5) << "N" << std::setw(12) << "entries" << std::setw(11) << "median ms"
            << std::setw(10) << "stddev" << std::setw(8) << "frames"
            << "  image vs first config of the view" << std::endl;

  CubeScene scene;
  float sceneHalfExtent = -1.0f;
  double sceneGenMs = 0.0;

  for (const BenchmarkView& view : VIEWS) {
    if (view.sceneHalfExtent != sceneHalfExtent) {
      auto start = Clock::now();
      scene = createCubeScene(NUM_CUBES, SCENE_CENTER, view.sceneHalfExtent);
      sceneGenMs = msSince(start);
      sceneHalfExtent = view.sceneHalfExtent;
    }
    setView(view);
    std::vector<uint8_t> referenceImage;

    for (const GridMethodConfig& method : METHODS) {
      for (uint32_t resolution : method.resolutions) {
        uint64_t entries = countGridEntries(scene, method.layout, resolution);
        if (entries * BYTES_PER_ENTRY > MAX_GRID_BYTES) {
          std::cout << std::left << std::setw(15) << view.name << std::setw(16) << method.name
                    << std::right << std::setw(5) << resolution << std::setw(12) << entries
                    << "  skipped: grid would need " << entries * BYTES_PER_ENTRY / (1 << 20)
                    << " MiB" << std::endl;
          continue;
        }
        window_.setTitle(std::string("Vulkan - ") + view.name + ", " + method.name + ", N=" +
                         std::to_string(resolution));

        BenchmarkInfo info;
        info.label = BENCHMARK_LABEL;
        info.renderer = method.name;
        info.view = view.name;
        info.buildConfig = BUILD_CONFIG;
        info.deviceName = volkan_.getDeviceName();
        info.width = volkan_.extent().width;
        info.height = volkan_.extent().height;
        info.cubeCount = NUM_CUBES;
        info.bytesPerEntry = BYTES_PER_ENTRY;
        info.sceneGenMs = sceneGenMs;
        loadGrid(scene, method, resolution, info);

        std::vector<FrameTiming> timings;
        if (!measureFrames(timings)) {
          std::cout << "Window closed, benchmark aborted." << std::endl;
          return;
        }

        // Every configuration of a view should render the same image. Count the
        // pixels that differ from the view's first configuration, and how many of
        // them gained or lost a hit (black <-> cube), which is what a bug would do.
        std::vector<uint8_t> image = volkan_.readStorageImage();
        info.imageHash = hashBytes(image);
        if (referenceImage.empty()) {
          referenceImage = image;
        }
        ImageDiff diff = compareImages(referenceImage, image);
        if (diff.differentPixels > largestDiff.differentPixels) {
          largestDiff = diff;
        }

        FrameStats stats = computeStats(timings);
        appendBenchmarkCsv(BENCHMARK_CSV_PATH, info, stats);
        results.push_back({view.name, method.name, resolution, stats});

        std::cout << std::left << std::setw(15) << view.name << std::setw(16) << method.name
                  << std::right << std::setw(5) << resolution << std::setw(12) << entries
                  << std::fixed << std::setprecision(2) << std::setw(11) << stats.medianGpuMs
                  << std::setw(10) << stats.stddevGpuMs << std::setw(8) << stats.frames << "  ";
        if (diff.differentPixels == 0) {
          std::cout << "identical" << std::endl;
        } else {
          std::cout << diff.differentPixels << " px differ (" << diff.hitChangedPixels
                    << " gained/lost a hit)" << std::endl;
        }
      }
    }
  }

  // Best resolution per view and method
  std::cout << "\nFastest grid resolution (median GPU ms per frame):\n";
  for (const BenchmarkView& view : VIEWS) {
    for (const GridMethodConfig& method : METHODS) {
      const Result* best = nullptr;
      for (const Result& result : results) {
        if (result.view == view.name && result.method == method.name &&
            (!best || result.stats.medianGpuMs < best->stats.medianGpuMs)) {
          best = &result;
        }
      }
      if (best) {
        std::cout << "  " << std::left << std::setw(15) << view.name << std::setw(16) << method.name
                  << std::right << "N=" << std::setw(3) << best->resolution << std::fixed
                  << std::setprecision(2) << std::setw(10) << best->stats.medianGpuMs << " ms\n";
      }
    }
  }
  uint64_t pixelCount = uint64_t(volkan_.extent().width) * volkan_.extent().height;
  std::cout << "Largest image difference within a view: " << largestDiff.differentPixels << " of "
            << pixelCount << " px (" << std::setprecision(3)
            << 100.0 * largestDiff.differentPixels / pixelCount << "%), "
            << largestDiff.hitChangedPixels << " gained/lost a hit\n"
            << "Results appended to " << BENCHMARK_CSV_PATH << std::endl;

  // Leave the fastest traversal grid for the first view loaded, for interactive use
  const Result* best = nullptr;
  for (const Result& result : results) {
    if (result.view == VIEWS[0].name && result.method == METHODS[1].name &&
        (!best || result.stats.medianGpuMs < best->stats.medianGpuMs)) {
      best = &result;
    }
  }
  if (best) {
    CubeScene firstScene = createCubeScene(NUM_CUBES, SCENE_CENTER, VIEWS[0].sceneHalfExtent);
    BenchmarkInfo unused;
    setView(VIEWS[0]);
    loadGrid(firstScene, METHODS[1], best->resolution, unused);
  }
  window_.setTitle("Vulkan - benchmark done (WASD to move)");
}

void Engine::setView(const BenchmarkView& view) {
  camera_.pos = view.cameraPos;
  yaw_ = view.yaw;
  pitch_ = view.pitch;
  updateCameraFront();
}

void Engine::loadGrid(const CubeScene& scene, const GridMethodConfig& method, uint32_t resolution,
                      BenchmarkInfo& info) {
  // The old grid may still be in use by frames in flight
  volkan_.device().waitIdle();
  discardPendingTimings();
  gridMaterial_.reset();

  auto start = Clock::now();
  GridData grid = buildGrid(scene, method.layout, resolution);
  info.gridBuildMs = msSince(start);
  info.gridResolution = resolution;
  info.gridEntries = grid.cubeIds.size();

  // Upload once through staging buffers, so the shader reads from VRAM
  start = Clock::now();
  auto upload = [&](const auto& values) {
    return volkan_.createDeviceLocalBuffer(values.data(), sizeof(values[0]) * values.size(),
                                           vk::BufferUsageFlagBits::eStorageBuffer);
  };
  cubeBuffer_ = upload(grid.invTransforms);
  sphereBuffer_ = upload(grid.spheres);
  cellBuffer_ = upload(grid.cellStart);
  cubeIdBuffer_ = upload(grid.cubeIds);
  info.uploadMs = msSince(start);

  start = Clock::now();
  gridMaterial_ = std::make_unique<GridMaterial>(volkan_, method.shader, cubeBuffer_, sphereBuffer_,
                                                 cellBuffer_, cubeIdBuffer_,
                                                 volkan_.storageImageView());
  info.pipelineMs = msSince(start);

  gridMin_ = grid.gridMin;
  gridPad_ = grid.pad;
  gridResolution_ = resolution;
}

bool Engine::measureFrames(std::vector<FrameTiming>& timings) {
  collected_.clear();
  framesSeen_ = 0;
  collecting_ = true;
  lastSubmitTime_ = Clock::now();

  while (true) {
    glfwPollEvents();
    if (window_.shouldClose()) {
      collecting_ = false;
      return false;
    }
    drawFrame();

    double totalGpuMs = 0.0;
    for (const FrameTiming& timing : collected_) {
      totalGpuMs += timing.gpuMs;
    }
    if (collected_.size() >= MAX_TIMED_FRAMES ||
        (collected_.size() >= MIN_TIMED_FRAMES && totalGpuMs >= MAX_TIMED_GPU_MS)) {
      break;
    }
  }

  collecting_ = false;
  volkan_.device().waitIdle();
  discardPendingTimings();
  timings = collected_;
  return true;
}

void Engine::discardPendingTimings() {
  for (bool& pending : timestampsPending_) {
    pending = false;
  }
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

void Engine::run() {
  if (RENDER_MODE == RenderMode::GridCompute) {
    runBenchmarkSuite();
  }
  benchmarkDone_ = true;
  lastSubmitTime_ = Clock::now();

  while (!window_.shouldClose()) {
    glfwPollEvents();
    processInput();
    drawFrame();
  }
}

void Engine::drawFrame() {
  if (!volkan_.nextFrameReady()) {
    if (volkan_.consumeSwapchainRecreated()) {
      if (rtMaterial_) {
        rtMaterial_->updateStorageImage(volkan_, volkan_.storageImageView());
      }
      if (gridMaterial_) {
        gridMaterial_->updateStorageImage(volkan_, volkan_.storageImageView());
      }
    }
    return;
  }

  // nextFrameReady() waited on this frame slot's fence, so the GPU has finished
  // the last frame that used it and its timestamps can be read without stalling.
  uint32_t frameIndex = volkan_.currentFrameIndex();
  if (timestampsPending_[frameIndex]) {
    collectFrameTiming(frameIndex);
  }

  volkan_.beginCommandBuffer();
  recordFrame(volkan_.currentCommandBuffer(), frameIndex);
  volkan_.copyStorageImageToSwapchain();
  volkan_.submit();

  auto now = Clock::now();
  frameIntervalMs_[frameIndex] = msBetween(lastSubmitTime_, now);
  lastSubmitTime_ = now;
  timestampsPending_[frameIndex] = true;
}

void Engine::recordFrame(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex) {
  // Timestamps around the render work measure pure GPU time for this frame.
  // The start uses eAllCommands, not eTopOfPipe: with two frames in flight the
  // GPU can reach this command while the previous frame is still rendering, and a
  // top-of-pipe timestamp would then include part of that frame's work.
  uint32_t firstQuery = 2 * frameIndex;
  commandBuffer.resetQueryPool(*timestampQueryPool_, firstQuery, 2);
  commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *timestampQueryPool_,
                               firstQuery);

  vk::Extent2D extent = volkan_.extent();
  if (rtMaterial_) {
    rtMaterial_->traceRays(commandBuffer, frameIndex, extent, volkan_, camera_);
  } else if (gridMaterial_) {
    gridMaterial_->dispatch(commandBuffer, frameIndex, makeGridPushConstants(), extent);
  }

  commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestampQueryPool_,
                               firstQuery + 1);
}

GridPushConstants Engine::makeGridPushConstants() const {
  vk::Extent2D extent = volkan_.extent();
  glm::mat4 viewProj = camera_.getProjectionMatrix(extent.width / static_cast<float>(extent.height)) *
                       camera_.getViewMatrix();

  GridPushConstants pushConstants;
  pushConstants.invViewProj = glm::inverse(viewProj);
  pushConstants.cameraPos = glm::vec4(camera_.pos, 1.0f);
  pushConstants.params = glm::uvec4(NUM_CUBES, 0, 0, gridResolution_);
  pushConstants.gridMin = gridMin_;
  pushConstants.gridPad = glm::vec4(gridPad_, 0.0f, 0.0f, 0.0f);
  return pushConstants;
}

void Engine::collectFrameTiming(uint32_t frameIndex) {
  timestampsPending_[frameIndex] = false;
  if (!collecting_) {
    return;
  }

  auto [result, timestamps] = timestampQueryPool_.getResults<uint64_t>(
      2 * frameIndex, 2, 2 * sizeof(uint64_t), sizeof(uint64_t),
      vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

  framesSeen_++;
  if (framesSeen_ <= WARMUP_FRAMES) {
    return;
  }
  FrameTiming timing;
  timing.gpuMs = (timestamps[1] - timestamps[0]) * volkan_.getTimestampPeriod() / 1e6;
  timing.frameIntervalMs = frameIntervalMs_[frameIndex];
  collected_.push_back(timing);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

void Engine::updateCameraFront() {
  glm::vec3 front;
  front.x = std::cos(glm::radians(yaw_)) * std::cos(glm::radians(pitch_));
  front.y = std::sin(glm::radians(pitch_));
  front.z = std::sin(glm::radians(yaw_)) * std::cos(glm::radians(pitch_));
  camera_.front = glm::normalize(front);
}

void Engine::processInput() {
  GLFWwindow* window = window_.handle();
  updateCameraFront();
  float cameraSpeed = 0.05f;
  glm::vec3 right = glm::normalize(glm::cross(camera_.front, camera_.up));
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera_.pos += cameraSpeed * camera_.front;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera_.pos -= cameraSpeed * camera_.front;
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera_.pos -= right * cameraSpeed;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera_.pos += right * cameraSpeed;
}

void Engine::handleFramebufferResized() {
  volkan_.notifyFramebufferResized();
}
