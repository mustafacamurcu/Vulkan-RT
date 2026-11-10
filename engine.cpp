#include "engine.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

#include <glm/gtc/constants.hpp>

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
  updateCameraFront();

  if (RENDER_MODE == RenderMode::HardwareRayTracing) {
    createRayTracingResources();
  } else {
    createBruteForceResources();
  }

  timestampQueryPool_ = volkan_.createTimestampQueryPool(2 * MAX_FRAMES_IN_FLIGHT);
  benchmarkInfo_.label = BENCHMARK_LABEL;
  benchmarkInfo_.buildConfig = BUILD_CONFIG;
  benchmarkInfo_.deviceName = volkan_.getDeviceName();
  benchmarkInfo_.cubeCount = NUM_CUBES;
  benchmarkInfo_.projectedCubeCount = PROJECTED_CUBE_COUNT;
}

Engine::~Engine() {
  // The GPU may still be using resources from the last frames. Everything
  // else is released by member destructors, in reverse declaration order.
  volkan_.device().waitIdle();
}

void Engine::createRayTracingResources() {
  auto start = Clock::now();
  std::vector<glm::mat4> transforms = createRandomCubeTransforms(NUM_CUBES);
  benchmarkInfo_.sceneGenMs = msSince(start);

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

  start = Clock::now();
  vk::BufferUsageFlags geometryUsage =
    vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
  vk::MemoryPropertyFlags hostMemory =
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

  vk::DeviceSize vertexBufferSize = sizeof(Vertex) * vertices.size();
  cubeVertexBuffer_ = volkan_.createBuffer(
    vertexBufferSize, geometryUsage | vk::BufferUsageFlagBits::eVertexBuffer, hostMemory);

  void* data = cubeVertexBuffer_.memory.mapMemory(0, vertexBufferSize);
  std::memcpy(data, vertices.data(), vertexBufferSize);
  cubeVertexBuffer_.memory.unmapMemory();

  vk::DeviceSize indexBufferSize = sizeof(uint32_t) * indices.size();
  cubeIndexBuffer_ = volkan_.createBuffer(
    indexBufferSize, geometryUsage | vk::BufferUsageFlagBits::eIndexBuffer, hostMemory);

  data = cubeIndexBuffer_.memory.mapMemory(0, indexBufferSize);
  std::memcpy(data, indices.data(), indexBufferSize);
  cubeIndexBuffer_.memory.unmapMemory();
  benchmarkInfo_.uploadMs = msSince(start);

  // Build the acceleration structures: one cube BLAS shared by every TLAS instance
  start = Clock::now();
  blas_ = volkan_.createBLAS(blasBuffer_,
                             volkan_.getBufferAddress(cubeVertexBuffer_),
                             static_cast<uint32_t>(vertices.size()),
                             volkan_.getBufferAddress(cubeIndexBuffer_),
                             static_cast<uint32_t>(indices.size()));
  tlas_ = volkan_.createTLAS(tlasBuffer_, blas_, transforms);
  benchmarkInfo_.cubePrepMs = msSince(start);

  start = Clock::now();
  rtMaterial_ = std::make_unique<RayTracingMaterial>(volkan_, tlas_, volkan_.storageImageView());
  benchmarkInfo_.pipelineMs = msSince(start);

  benchmarkInfo_.renderer = "hardware-rt";
  benchmarkInfo_.linearInCubeCount = false;  // BVH traversal is not linear in cube count
}

void Engine::createBruteForceResources() {
  auto start = Clock::now();
  std::vector<glm::mat4> transforms = createRandomCubeTransforms(NUM_CUBES);
  benchmarkInfo_.sceneGenMs = msSince(start);

  // The shader needs world -> cube space, so store the inverse of every transform
  start = Clock::now();
  std::vector<glm::mat4> invTransforms(transforms.size());
  for (size_t i = 0; i < transforms.size(); i++) {
    invTransforms[i] = glm::inverse(transforms[i]);
  }

  // Bounding sphere of each cube, for the cheap rejection test. The unit cube's
  // corners are sqrt(3)/2 from its center, scaled by the cube's (uniform) scale.
  // The corners sit exactly on that sphere, so a 0.1% larger radius is a safety
  // margin against float rounding rejecting a ray that grazes a corner.
  constexpr float RADIUS_PADDING = 1.001f;
  std::vector<glm::vec4> spheres(transforms.size());
  for (size_t i = 0; i < transforms.size(); i++) {
    glm::vec3 center = glm::vec3(transforms[i][3]);
    float scale = glm::length(glm::vec3(transforms[i][0]));
    spheres[i] = glm::vec4(center, scale * 0.5f * std::sqrt(3.0f) * RADIUS_PADDING);
  }
  // Uniform grid for 3D-DDA traversal. The ray only visits cells it actually
  // crosses, so every cube must be listed in EVERY cell its bounding box
  // overlaps (a cube centered in a cell the ray skips can still stick into one
  // it visits). Cells are cubic and span the bounding boxes of all cubes.
  glm::vec3 gridLo(FLT_MAX);
  glm::vec3 gridHi(-FLT_MAX);
  for (const glm::vec4& sphere : spheres) {
    gridLo = glm::min(gridLo, glm::vec3(sphere) - sphere.w);
    gridHi = glm::max(gridHi, glm::vec3(sphere) + sphere.w);
  }
  glm::vec3 gridExtent = gridHi - gridLo;
  // Tiny margin so boxes touching the max corner still land inside the last cell
  float cellSize = std::max({gridExtent.x, gridExtent.y, gridExtent.z}) / GRID_RESOLUTION * 1.0001f;
  gridMin_ = glm::vec4(gridLo, cellSize);

  const uint32_t N = GRID_RESOLUTION;
  auto cellCoord = [&](const glm::vec3& point) {
    return glm::min(glm::uvec3((point - gridLo) / cellSize), glm::uvec3(N - 1));
  };
  // Calls visit(cellIndex) for every cell the sphere's bounding box overlaps
  auto forEachOverlappedCell = [&](const glm::vec4& sphere, auto&& visit) {
    glm::uvec3 lo = cellCoord(glm::vec3(sphere) - sphere.w);
    glm::uvec3 hi = cellCoord(glm::vec3(sphere) + sphere.w);
    for (uint32_t z = lo.z; z <= hi.z; z++) {
      for (uint32_t y = lo.y; y <= hi.y; y++) {
        for (uint32_t x = lo.x; x <= hi.x; x++) {
          visit(x + N * (y + N * z));
        }
      }
    }
  };

  // Counting sort by cell: count entries per cell, prefix-sum into start offsets,
  // then copy each cube into every cell it overlaps.
  std::vector<uint32_t> cellStart(N * N * N + 1, 0);
  for (const glm::vec4& sphere : spheres) {
    forEachOverlappedCell(sphere, [&](uint32_t cell) { cellStart[cell + 1]++; });
  }
  for (size_t c = 1; c < cellStart.size(); c++) {
    cellStart[c] += cellStart[c - 1];
  }
  uint32_t entryCount = cellStart.back();
  std::vector<uint32_t> nextSlot(cellStart.begin(), cellStart.end() - 1);
  std::vector<glm::mat4> sortedInvTransforms(entryCount);
  std::vector<glm::vec4> sortedSpheres(entryCount);
  std::vector<uint32_t> cubeIds(entryCount);
  for (uint32_t i = 0; i < spheres.size(); i++) {
    forEachOverlappedCell(spheres[i], [&](uint32_t cell) {
      uint32_t slot = nextSlot[cell]++;
      sortedInvTransforms[slot] = invTransforms[i];
      sortedSpheres[slot] = spheres[i];
      cubeIds[slot] = i;
    });
  }
  std::cout << "Grid: " << N << "^3 cells, " << entryCount << " entries for "
            << spheres.size() << " cubes (" << 100.0 * (entryCount - spheres.size()) / spheres.size()
            << "% duplicates)" << std::endl;
  benchmarkInfo_.cubePrepMs = msSince(start);

  // Upload once through staging buffers, so the shader reads from VRAM
  start = Clock::now();
  auto upload = [&](const auto& values) {
    return volkan_.createDeviceLocalBuffer(values.data(), sizeof(values[0]) * values.size(),
                                           vk::BufferUsageFlagBits::eStorageBuffer);
  };
  cubeBuffer_ = upload(sortedInvTransforms);
  sphereBuffer_ = upload(sortedSpheres);
  cellBuffer_ = upload(cellStart);
  cubeIdBuffer_ = upload(cubeIds);
  benchmarkInfo_.uploadMs = msSince(start);

  start = Clock::now();
  bruteForceMaterial_ = std::make_unique<BruteForceMaterial>(
      volkan_, cubeBuffer_, sphereBuffer_, cellBuffer_, cubeIdBuffer_,
      volkan_.storageImageView());
  benchmarkInfo_.pipelineMs = msSince(start);

  benchmarkInfo_.renderer = BRUTE_FORCE_TILED ? "grid-dda-compute-tiled" : "grid-dda-compute";
  benchmarkInfo_.tileSize = BRUTE_FORCE_TILED ? TILE_SIZE : 0;
  benchmarkInfo_.linearInCubeCount = false;  // the grid skips most cubes
  benchmarkInfo_.bytesPerCube = sizeof(glm::mat4) + sizeof(glm::vec4) + sizeof(uint32_t);
  benchmarkInfo_.cubeMemory =
      volkan_.describeBufferMemory(cubeBuffer_, vk::MemoryPropertyFlagBits::eDeviceLocal);
  benchmarkInfo_.workgroupSize = BruteForceMaterial::WORKGROUP_SIZE;
}

float Engine::sceneHalfExtent() {
  // Volume scales with the cube count, so the side length scales with its cube root
  return FULL_SCENE_HALF_EXTENT *
         std::cbrt(static_cast<float>(NUM_CUBES) / static_cast<float>(PROJECTED_CUBE_COUNT));
}

std::vector<glm::mat4> Engine::createRandomCubeTransforms(uint32_t count) const {
  // Fixed seed so the scene is the same every run
  std::mt19937 rng(1337);
  float halfExtent = sceneHalfExtent();
  std::uniform_real_distribution<float> positionDist(-halfExtent, halfExtent);
  std::uniform_real_distribution<float> angleDist(0.0f, glm::two_pi<float>());
  std::uniform_real_distribution<float> axisDist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> scaleDist(0.5f, 3.0f);

  std::vector<glm::mat4> transforms;
  transforms.reserve(count);
  for (uint32_t i = 0; i < count; i++) {
    glm::vec3 position = SCENE_CENTER +
        glm::vec3(positionDist(rng), positionDist(rng), positionDist(rng));

    glm::vec3 axis(axisDist(rng), axisDist(rng), axisDist(rng));
    if (glm::length(axis) < 0.001f) {
      axis = glm::vec3(0.0f, 1.0f, 0.0f);
    }

    glm::mat4 transform = glm::translate(glm::mat4(1.0f), position);
    transform = glm::rotate(transform, angleDist(rng), glm::normalize(axis));
    transform = glm::scale(transform, glm::vec3(scaleDist(rng)));
    transforms.push_back(transform);
  }
  return transforms;
}

void Engine::run() {
  std::cout << "Benchmarking " << BENCHMARK_WARMUP_FRAMES << " warm-up + "
            << BENCHMARK_TIMED_FRAMES << " timed frames..." << std::endl;
  if (BRUTE_FORCE_TILED) {
    // Renders the frames up front, reports, then keeps showing the last image
    runTiledBenchmark();
  }
  lastSubmitTime_ = Clock::now();

  while (!window_.shouldClose()) {
    glfwPollEvents();
    // Keep the camera still while benchmarking so every frame renders the same thing.
    // Tiled frames take seconds each, so the camera stays put in that mode.
    if (benchmarkDone_ && !BRUTE_FORCE_TILED) {
      processInput();
    }
    drawFrame();
  }
}

void Engine::drawFrame() {
  if (!volkan_.nextFrameReady()) {
    if (volkan_.consumeSwapchainRecreated()) {
      if (rtMaterial_) {
        rtMaterial_->updateStorageImage(volkan_, volkan_.storageImageView());
      }
      if (bruteForceMaterial_) {
        bruteForceMaterial_->updateStorageImage(volkan_, volkan_.storageImageView());
      }
      if (BRUTE_FORCE_TILED && benchmarkDone_) {
        // The storage image was recreated empty, so render it again (slow!)
        FrameTiming timing;
        renderTiledFrame(timing);
      }
    }
    return;
  }

  if (BRUTE_FORCE_TILED) {
    // The image was rendered tile by tile outside the frame loop; only present it
    volkan_.beginCommandBuffer();
    volkan_.copyStorageImageToSwapchain();
    volkan_.submit();
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
  if (RENDER_MODE == RenderMode::HardwareRayTracing) {
    rtMaterial_->traceRays(commandBuffer, frameIndex, extent, volkan_, camera_);
  } else {
    bruteForceMaterial_->dispatch(commandBuffer, frameIndex, makeBruteForcePushConstants(),
                                  extent);
  }

  commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestampQueryPool_,
                               firstQuery + 1);
}

BruteForcePushConstants Engine::makeBruteForcePushConstants() const {
  vk::Extent2D extent = volkan_.extent();
  glm::mat4 viewProj = camera_.getProjectionMatrix(extent.width / static_cast<float>(extent.height)) *
                       camera_.getViewMatrix();

  BruteForcePushConstants pushConstants;
  pushConstants.invViewProj = glm::inverse(viewProj);
  pushConstants.cameraPos = glm::vec4(camera_.pos, 1.0f);
  pushConstants.params = glm::uvec4(NUM_CUBES, 0, 0, GRID_RESOLUTION);  // tile offset starts at 0
  pushConstants.gridMin = gridMin_;
  return pushConstants;
}

bool Engine::renderTiledFrame(FrameTiming& timing) {
  vk::Extent2D extent = volkan_.extent();
  BruteForcePushConstants pushConstants = makeBruteForcePushConstants();
  float timestampPeriod = volkan_.getTimestampPeriod();
  uint32_t tilesX = (extent.width + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t tilesY = (extent.height + TILE_SIZE - 1) / TILE_SIZE;

  timing = FrameTiming();
  double slowestTileMs = 0.0;
  auto frameStart = Clock::now();

  for (uint32_t tileY = 0; tileY < tilesY; tileY++) {
    for (uint32_t tileX = 0; tileX < tilesX; tileX++) {
      uint32_t offsetX = tileX * TILE_SIZE;
      uint32_t offsetY = tileY * TILE_SIZE;
      pushConstants.params.y = offsetX;
      pushConstants.params.z = offsetY;
      vk::Extent2D region{std::min(TILE_SIZE, extent.width - offsetX),
                          std::min(TILE_SIZE, extent.height - offsetY)};

      // Each tile is its own submission, timed with its own pair of timestamps
      vk::raii::CommandBuffer cmdBuffer = volkan_.beginSingleTimeCommands();
      cmdBuffer.resetQueryPool(*timestampQueryPool_, 0, 2);
      cmdBuffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, *timestampQueryPool_, 0);
      bruteForceMaterial_->dispatch(cmdBuffer, 0, pushConstants, region);
      cmdBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestampQueryPool_, 1);
      volkan_.endSingleTimeCommands(cmdBuffer);

      auto [result, timestamps] = timestampQueryPool_.getResults<uint64_t>(
          0, 2, 2 * sizeof(uint64_t), sizeof(uint64_t),
          vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
      double tileMs = (timestamps[1] - timestamps[0]) * timestampPeriod / 1e6;
      timing.gpuMs += tileMs;
      slowestTileMs = std::max(slowestTileMs, tileMs);

      // Keep the window responsive during a multi-second frame
      glfwPollEvents();
      if (window_.shouldClose()) {
        return false;
      }
      uint32_t tileIndex = tileY * tilesX + tileX + 1;
      window_.setTitle("Vulkan - tile " + std::to_string(tileIndex) + "/" +
                       std::to_string(tilesX * tilesY));
    }
  }

  timing.frameIntervalMs = msSince(frameStart);
  std::cout << "  GPU " << timing.gpuMs << " ms, slowest tile " << slowestTileMs
            << " ms (Windows GPU timeout ~2000 ms)" << std::endl;
  return true;
}

void Engine::runTiledBenchmark() {
  uint32_t totalFrames = BENCHMARK_WARMUP_FRAMES + BENCHMARK_TIMED_FRAMES;
  for (uint32_t i = 0; i < totalFrames; i++) {
    bool warmup = i < BENCHMARK_WARMUP_FRAMES;
    std::cout << (warmup ? "Warm-up frame " : "Timed frame ") << i + 1 << "/" << totalFrames
              << ":" << std::endl;

    FrameTiming timing;
    if (!renderTiledFrame(timing)) {
      std::cout << "Window closed, benchmark aborted." << std::endl;
      return;
    }
    drawFrame();  // show the result
    if (!warmup) {
      frameTimings_.push_back(timing);
    }
  }
  finishBenchmark();
}

void Engine::collectFrameTiming(uint32_t frameIndex) {
  timestampsPending_[frameIndex] = false;
  if (benchmarkDone_) {
    return;
  }

  auto [result, timestamps] = timestampQueryPool_.getResults<uint64_t>(
      2 * frameIndex, 2, 2 * sizeof(uint64_t), sizeof(uint64_t),
      vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

  FrameTiming timing;
  timing.gpuMs = (timestamps[1] - timestamps[0]) * volkan_.getTimestampPeriod() / 1e6;
  timing.frameIntervalMs = frameIntervalMs_[frameIndex];

  framesMeasured_++;
  bool warmup = framesMeasured_ <= BENCHMARK_WARMUP_FRAMES;
  std::cout << (warmup ? "  warm-up " : "  frame   ") << framesMeasured_ << ": GPU "
            << timing.gpuMs << " ms" << std::endl;
  if (!warmup) {
    frameTimings_.push_back(timing);
  }

  if (frameTimings_.size() == BENCHMARK_TIMED_FRAMES) {
    finishBenchmark();
  }
}

void Engine::finishBenchmark() {
  benchmarkDone_ = true;
  benchmarkInfo_.width = volkan_.extent().width;
  benchmarkInfo_.height = volkan_.extent().height;

  printBenchmarkReport(benchmarkInfo_, frameTimings_);
  appendBenchmarkCsv(BENCHMARK_CSV_PATH, benchmarkInfo_, frameTimings_);
  std::cout << "Results appended to " << BENCHMARK_CSV_PATH << "\n"
            << "Benchmark done: WASD now moves the camera." << std::endl;
  window_.setTitle("Vulkan - benchmark done");
}

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
