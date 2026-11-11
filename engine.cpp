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

Engine::Engine(const StartupConfig& startup) : startup_(startup), window_(WIDTH, HEIGHT, "Vulkan") {
  glfwSetWindowUserPointer(window_.handle(), this);
  glfwSetFramebufferSizeCallback(window_.handle(), framebufferResizeCallback);

  volkan_.init(window_.handle());
  setView(SPARSE_SCENE.camera);
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
  CubeScene scene = createCubeScene(SPARSE_SCENE.regions);

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
// Benchmark suite: every shading mode at every grid resolution
// ---------------------------------------------------------------------------

namespace {

const char* shadingName(ShadingMode mode) {
  switch (mode) {
    case ShadingMode::Flat: return "flat";
    case ShadingMode::Diffuse: return "diffuse";
    case ShadingMode::Shadows: return "shadows";
  }
  return "?";
}

}  // namespace

void Engine::runBenchmarkSuite() {
  struct Result {
    ShadingMode shading;
    uint32_t resolution;
    FrameStats stats;
  };
  std::vector<Result> results;
  ImageDiff largestDiff;

  auto start = Clock::now();
  const SceneConfig& sceneConfig = QUADRANT_SCENE;
  CubeScene scene = createCubeScene(sceneConfig.regions);
  double sceneGenMs = msSince(start);
  setView(sceneConfig.camera);

  std::cout << "Benchmark suite: " << scene.spheres.size() << " cubes in " << sceneConfig.regions.size()
            << " regions, " << volkan_.extent().width << "x" << volkan_.extent().height << "\n"
            << std::left << std::setw(10) << "shading" << std::right << std::setw(5) << "N"
            << std::setw(12) << "entries" << std::setw(11) << "median ms" << std::setw(10)
            << "stddev" << std::setw(8) << "frames" << "  image vs first N of the same shading"
            << std::endl;

  // One reference image per shading mode (each mode renders a different picture)
  std::vector<std::vector<uint8_t>> referenceImages(std::size(SHADING_MODES));

  for (uint32_t resolution : GRID_RESOLUTIONS) {
    uint64_t entries = countGridEntries(scene, GridLayout::ByOverlap, resolution);
    if (entries * BYTES_PER_ENTRY > MAX_GRID_BYTES) {
      std::cout << std::setw(15) << resolution << std::setw(12) << entries
                << "  skipped: grid would need " << entries * BYTES_PER_ENTRY / (1 << 20)
                << " MiB" << std::endl;
      continue;
    }

    BenchmarkInfo info;
    info.label = BENCHMARK_LABEL;
    info.view = sceneConfig.camera.name;
    info.buildConfig = BUILD_CONFIG;
    info.deviceName = volkan_.getDeviceName();
    info.width = volkan_.extent().width;
    info.height = volkan_.extent().height;
    info.cubeCount = static_cast<uint32_t>(scene.spheres.size());
    info.bytesPerEntry = BYTES_PER_ENTRY;
    info.sceneGenMs = sceneGenMs;
    loadGrid(scene, resolution, info);

    for (size_t m = 0; m < std::size(SHADING_MODES); m++) {
      const ShadingConfig& shading = SHADING_MODES[m];
      setShading(shading.mode);
      info.renderer = std::string("grid-traversal/") + shading.name;
      window_.setTitle(std::string("Vulkan - ") + shading.name + ", N=" + std::to_string(resolution));

      std::vector<FrameTiming> timings;
      if (!measureFrames(timings)) {
        std::cout << "Window closed, benchmark aborted." << std::endl;
        return;
      }

      // Every grid resolution should render the same image for a given shading
      // mode. Count the pixels that differ from that mode's first resolution.
      std::vector<uint8_t> image = volkan_.readStorageImage();
      info.imageHash = hashBytes(image);
      if (referenceImages[m].empty()) {
        referenceImages[m] = image;
      }
      ImageDiff diff = compareImages(referenceImages[m], image);
      if (diff.differentPixels > largestDiff.differentPixels) {
        largestDiff = diff;
      }

      FrameStats stats = computeStats(timings);
      appendBenchmarkCsv(BENCHMARK_CSV_PATH, info, stats);
      results.push_back({shading.mode, resolution, stats});

      std::cout << std::left << std::setw(10) << shading.name << std::right << std::setw(5)
                << resolution << std::setw(12) << entries << std::fixed << std::setprecision(2)
                << std::setw(11) << stats.medianGpuMs << std::setw(10) << stats.stddevGpuMs
                << std::setw(8) << stats.frames << "  ";
      if (diff.differentPixels == 0) {
        std::cout << "identical" << std::endl;
      } else {
        std::cout << diff.differentPixels << " px differ (" << diff.hitChangedPixels
                  << " gained/lost a hit)" << std::endl;
      }
    }
  }

  // Fastest resolution per shading mode
  std::cout << "\nFastest grid resolution (median GPU ms per frame):\n";
  uint32_t bestShadowResolution = sceneConfig.defaultGridResolution;
  for (const ShadingConfig& shading : SHADING_MODES) {
    const Result* best = nullptr;
    for (const Result& result : results) {
      if (result.shading == shading.mode &&
          (!best || result.stats.medianGpuMs < best->stats.medianGpuMs)) {
        best = &result;
      }
    }
    if (best) {
      std::cout << "  " << std::left << std::setw(10) << shading.name << std::right << "N="
                << std::setw(3) << best->resolution << std::fixed << std::setprecision(2)
                << std::setw(10) << best->stats.medianGpuMs << " ms\n";
      if (shading.mode == DEFAULT_SHADING) {
        bestShadowResolution = best->resolution;
      }
    }
  }
  uint64_t pixelCount = uint64_t(volkan_.extent().width) * volkan_.extent().height;
  std::cout << "Largest image difference across resolutions: " << largestDiff.differentPixels
            << " of " << pixelCount << " px (" << std::setprecision(3)
            << 100.0 * largestDiff.differentPixels / pixelCount << "%), "
            << largestDiff.hitChangedPixels << " gained/lost a hit\n"
            << "Results appended to " << BENCHMARK_CSV_PATH << std::endl;

  // Leave the default shading at its fastest resolution, for interactive use
  if (gridResolution_ != bestShadowResolution) {
    BenchmarkInfo unused;
    loadGrid(scene, bestShadowResolution, unused);
  }
  setShading(DEFAULT_SHADING);
  window_.setTitle("Vulkan - benchmark done (WASD move, QE up/down, arrows turn, Shift fast)");
}

void Engine::setView(const CameraView& view) {
  camera_.pos = view.position;
  yaw_ = view.yaw;
  pitch_ = view.pitch;
  updateCameraFront();
}

void Engine::setShading(ShadingMode mode) {
  shading_ = mode;
  gridLabel_ = "grid-traversal N=" + std::to_string(gridResolution_) + " " + shadingName(mode);
}

void Engine::loadGrid(const CubeScene& scene, uint32_t resolution, BenchmarkInfo& info) {
  // The old grid may still be in use by frames in flight
  volkan_.device().waitIdle();
  discardPendingTimings();
  gridMaterial_.reset();

  auto start = Clock::now();
  GridData grid = buildGrid(scene, GridLayout::ByOverlap, resolution);
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
  gridMaterial_ = std::make_unique<GridMaterial>(volkan_, GRID_SHADER, cubeBuffer_, sphereBuffer_,
                                                 cellBuffer_, cubeIdBuffer_,
                                                 volkan_.storageImageView());
  info.pipelineMs = msSince(start);

  gridMin_ = grid.gridMin;
  gridResolution_ = resolution;
  setShading(shading_);  // refreshes the perf marker name
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

void Engine::loadViewer() {
  const SceneConfig* sceneConfig = &SPARSE_SCENE;
  if (startup_.scene == QUADRANT_SCENE.name) {
    sceneConfig = &QUADRANT_SCENE;
  } else if (!startup_.scene.empty() && startup_.scene != SPARSE_SCENE.name) {
    throw std::runtime_error("unknown scene " + startup_.scene + " (try --help)");
  }

  uint32_t resolution = startup_.gridResolution != 0 ? startup_.gridResolution
                                                     : sceneConfig->defaultGridResolution;
  ShadingMode shading = DEFAULT_SHADING;
  if (!startup_.shading.empty()) {
    bool found = false;
    for (const ShadingConfig& candidate : SHADING_MODES) {
      if (startup_.shading == candidate.name) {
        shading = candidate.mode;
        found = true;
      }
    }
    if (!found) throw std::runtime_error("unknown shading " + startup_.shading + " (try --help)");
  }

  CubeScene scene = createCubeScene(sceneConfig->regions);
  setView(sceneConfig->camera);
  BenchmarkInfo info;
  loadGrid(scene, resolution, info);
  setShading(shading);
  window_.setTitle(std::string("Vulkan - ") + sceneConfig->name + ", " + shadingName(shading) +
                   ", N=" + std::to_string(resolution) + "  (WASD move, QE up/down, arrows turn, Shift fast)");
  std::cout << "Rendering " << sceneConfig->name << ", " << shadingName(shading) << ", N="
            << resolution << " (" << info.gridEntries << " grid entries)" << std::endl;
}

void Engine::run() {
  if (RENDER_MODE == RenderMode::GridCompute) {
    if (startup_.benchmark) {
      runBenchmarkSuite();
    } else {
      loadViewer();
    }
  }
  lastSubmitTime_ = Clock::now();
  lastInputTime_ = Clock::now();

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
  vk::raii::CommandBuffer& commandBuffer = volkan_.currentCommandBuffer();
  recordFrame(commandBuffer, frameIndex);

  // Perf markers: named regions that profilers show on their timeline, and that
  // Nsight's multi-pass metrics use to match the same work across frames
  commandBuffer.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT("copy to swapchain"));
  volkan_.copyStorageImageToSwapchain();
  commandBuffer.endDebugUtilsLabelEXT();
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
    commandBuffer.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT("hardware-rt"));
    rtMaterial_->traceRays(commandBuffer, frameIndex, extent, volkan_, camera_);
    commandBuffer.endDebugUtilsLabelEXT();
  } else if (gridMaterial_) {
    // Same name every frame for a given configuration, e.g. "grid-traversal N=64"
    commandBuffer.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT(gridLabel_.c_str()));
    gridMaterial_->dispatch(commandBuffer, frameIndex, makeGridPushConstants(), extent);
    commandBuffer.endDebugUtilsLabelEXT();
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
  pushConstants.params = glm::uvec4(0, static_cast<uint32_t>(shading_), 0, gridResolution_);
  pushConstants.gridMin = gridMin_;
  pushConstants.gridPad = glm::vec4(0.0f);
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
  // Move by time, not by frame, so speed doesn't depend on the frame rate
  auto now = Clock::now();
  float dt = static_cast<float>(msBetween(lastInputTime_, now) / 1000.0);
  lastInputTime_ = now;
  dt = std::min(dt, 0.1f);  // don't jump after a stall (e.g. a slow frame)

  GLFWwindow* window = window_.handle();
  auto pressed = [&](int key) { return glfwGetKey(window, key) == GLFW_PRESS; };

  // Arrow keys turn the camera
  float turn = TURN_SPEED * dt;
  if (pressed(GLFW_KEY_LEFT)) yaw_ += turn;
  if (pressed(GLFW_KEY_RIGHT)) yaw_ -= turn;
  if (pressed(GLFW_KEY_UP)) pitch_ -= turn;    // -y is up, so looking up lowers the pitch
  if (pressed(GLFW_KEY_DOWN)) pitch_ += turn;
  pitch_ = std::clamp(pitch_, -89.0f, 89.0f);
  updateCameraFront();

  float speed = MOVE_SPEED * dt * (pressed(GLFW_KEY_LEFT_SHIFT) ? 5.0f : 1.0f);
  glm::vec3 right = glm::normalize(glm::cross(camera_.front, camera_.up));
  if (pressed(GLFW_KEY_W)) camera_.pos += speed * camera_.front;
  if (pressed(GLFW_KEY_S)) camera_.pos -= speed * camera_.front;
  if (pressed(GLFW_KEY_A)) camera_.pos -= speed * right;
  if (pressed(GLFW_KEY_D)) camera_.pos += speed * right;
  if (pressed(GLFW_KEY_E)) camera_.pos += speed * camera_.up;
  if (pressed(GLFW_KEY_Q)) camera_.pos -= speed * camera_.up;
}

void Engine::handleFramebufferResized() {
  volkan_.notifyFramebufferResized();
}
