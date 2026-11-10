#include "engine.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <random>
#include <chrono>
#include <string>

#include "vk_utils.h"

#include "rt_material.h"

static double msSince(std::chrono::high_resolution_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - start)
      .count();
}

static void framebufferResizeCallback(GLFWwindow* window, int width,
                                      int height) {
  auto engine = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
  engine->handleFramebufferResized();
}

Engine::Engine() {
  createWindow(WIDTH, HEIGHT, "Vulkan Scene");
  volkan_.init(window_);
  updateCameraFront();
  if (RENDER_MODE == RenderMode::HardwareRayTracing) {
    createRayTracingResources();
  } else {
    createBruteForceResources();
  }
}

void Engine::createRayTracingResources() {
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
  vk::MemoryPropertyFlags hostMemory =
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

  vk::DeviceSize vertexBufferSize = sizeof(Vertex) * vertices.size();
  cubeVertexBuffer_ = volkan_.createBuffer(
    vertexBufferSize, geometryUsage | vk::BufferUsageFlagBits::eVertexBuffer, hostMemory);

  void* data = cubeVertexBuffer_.memory.mapMemory(0, vertexBufferSize);
  memcpy(data, vertices.data(), vertexBufferSize);
  cubeVertexBuffer_.memory.unmapMemory();

  vk::DeviceSize indexBufferSize = sizeof(uint32_t) * indices.size();
  cubeIndexBuffer_ = volkan_.createBuffer(
    indexBufferSize, geometryUsage | vk::BufferUsageFlagBits::eIndexBuffer, hostMemory);

  data = cubeIndexBuffer_.memory.mapMemory(0, indexBufferSize);
  memcpy(data, indices.data(), indexBufferSize);
  cubeIndexBuffer_.memory.unmapMemory();

  // Create BLAS (one cube, shared by every instance)
  blas_ = volkan_.createBLAS(blasBuffer_,
                             volkan_.getBufferAddress(cubeVertexBuffer_),
                             static_cast<uint32_t>(vertices.size()),
                             volkan_.getBufferAddress(cubeIndexBuffer_),
                             static_cast<uint32_t>(indices.size()));

  // Create TLAS (one instance per cube)
  tlas_ = volkan_.createTLAS(tlasBuffer_, blas_, createRandomCubeTransforms(NUM_CUBES));

  // Create ray tracing material
  rtMaterial_ = std::make_unique<RayTracingMaterial>(volkan_, tlas_, volkan_.storageImageView_);
}

void Engine::createBruteForceResources() {
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<glm::mat4> transforms = createRandomCubeTransforms(NUM_CUBES);
  benchmarkInfo_.sceneGenMs = msSince(start);

  // The shader needs world -> cube space, so store the inverse of every transform
  start = std::chrono::high_resolution_clock::now();
  std::vector<glm::mat4> invTransforms(transforms.size());
  for (size_t i = 0; i < transforms.size(); i++) {
    invTransforms[i] = glm::inverse(transforms[i]);
  }
  benchmarkInfo_.cubePrepMs = msSince(start);

  // Naive upload: one host-visible buffer the GPU reads directly, no staging
  start = std::chrono::high_resolution_clock::now();
  vk::DeviceSize bufferSize = sizeof(glm::mat4) * invTransforms.size();
  vk::MemoryPropertyFlags hostMemory =
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
  cubeBuffer_ = volkan_.createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer, hostMemory);

  void* data = cubeBuffer_.memory.mapMemory(0, bufferSize);
  memcpy(data, invTransforms.data(), bufferSize);
  cubeBuffer_.memory.unmapMemory();
  benchmarkInfo_.uploadMs = msSince(start);

  start = std::chrono::high_resolution_clock::now();
  bruteForceMaterial_ = std::make_unique<BruteForceMaterial>(
      volkan_, cubeBuffer_, volkan_.storageImageView_);
  benchmarkInfo_.pipelineMs = msSince(start);

  timestampQueryPool_ = volkan_.createTimestampQueryPool(2);

  benchmarkInfo_.label = BENCHMARK_LABEL;
  benchmarkInfo_.deviceName = volkan_.getDeviceName();
  benchmarkInfo_.cubeCount = NUM_CUBES;
  benchmarkInfo_.bytesPerCube = sizeof(glm::mat4);
  benchmarkInfo_.cubeMemory = volkan_.describeBufferMemory(cubeBuffer_, hostMemory);
  benchmarkInfo_.tileSize = TILE_SIZE;
  benchmarkInfo_.workgroupSize = BruteForceMaterial::WORKGROUP_SIZE;
}

bool Engine::renderBruteForceFrame(FrameTiming& timing) {
  uint32_t width = volkan_.swapChainExtent_.width;
  uint32_t height = volkan_.swapChainExtent_.height;

  glm::mat4 viewProj = camera_.getProjectionMatrix(width / (float)height) * camera_.getViewMatrix();

  BruteForcePushConstants pushConstants;
  pushConstants.invViewProj = glm::inverse(viewProj);
  pushConstants.cameraPos = glm::vec4(camera_.pos, 1.0f);
  pushConstants.tileInfo = glm::uvec4(0, 0, NUM_CUBES, 0);

  float timestampPeriod = volkan_.getTimestampPeriod();
  uint32_t tilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t tilesY = (height + TILE_SIZE - 1) / TILE_SIZE;

  timing = FrameTiming();
  auto frameStart = std::chrono::high_resolution_clock::now();

  for (uint32_t tileY = 0; tileY < tilesY; tileY++) {
    for (uint32_t tileX = 0; tileX < tilesX; tileX++) {
      uint32_t offsetX = tileX * TILE_SIZE;
      uint32_t offsetY = tileY * TILE_SIZE;
      pushConstants.tileInfo.x = offsetX;
      pushConstants.tileInfo.y = offsetY;

      // Timestamps around the dispatch measure pure GPU time for this tile
      vk::raii::CommandBuffer cmdBuffer = volkan_.beginSingleTimeCommands();
      cmdBuffer.resetQueryPool(*timestampQueryPool_, 0, 2);
      cmdBuffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *timestampQueryPool_, 0);
      bruteForceMaterial_->dispatchTile(cmdBuffer, pushConstants,
                                        std::min(TILE_SIZE, width - offsetX),
                                        std::min(TILE_SIZE, height - offsetY));
      cmdBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestampQueryPool_, 1);
      volkan_.endSingleTimeCommands(cmdBuffer);

      auto [result, timestamps] = timestampQueryPool_.getResults<uint64_t>(
          0, 2, 2 * sizeof(uint64_t), sizeof(uint64_t),
          vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
      double tileGpuMs = (timestamps[1] - timestamps[0]) * timestampPeriod / 1e6;

      timing.gpuMs += tileGpuMs;
      timing.maxTileGpuMs = std::max(timing.maxTileGpuMs, tileGpuMs);
      timing.tileCount++;

      // Keep the window responsive during a multi-second frame
      glfwPollEvents();
      if (glfwWindowShouldClose(window_)) {
        return false;
      }
      std::string title = "Vulkan - tile " + std::to_string(timing.tileCount) + "/" +
                          std::to_string(tilesX * tilesY);
      glfwSetWindowTitle(window_, title.c_str());
    }
  }

  timing.wallMs = msSince(frameStart);
  return true;
}

void Engine::runBenchmark() {
  std::vector<FrameTiming> frames;
  uint32_t totalFrames = BENCHMARK_WARMUP_FRAMES + BENCHMARK_TIMED_FRAMES;

  for (uint32_t i = 0; i < totalFrames; i++) {
    bool warmup = i < BENCHMARK_WARMUP_FRAMES;
    std::cout << (warmup ? "Warm-up frame " : "Timed frame ") << i + 1 << "/" << totalFrames
              << "..." << std::flush;

    FrameTiming timing;
    if (!renderBruteForceFrame(timing)) {
      std::cout << " window closed, benchmark aborted." << std::endl;
      return;
    }
    std::cout << " GPU " << timing.gpuMs << " ms" << std::endl;

    // Show the result
    drawFrame();

    if (!warmup) {
      frames.push_back(timing);
    }
  }

  benchmarkInfo_.width = volkan_.swapChainExtent_.width;
  benchmarkInfo_.height = volkan_.swapChainExtent_.height;
  printBenchmarkReport(benchmarkInfo_, frames);
  appendBenchmarkCsv(BENCHMARK_CSV_PATH, benchmarkInfo_, frames);
  std::cout << "Results appended to " << BENCHMARK_CSV_PATH << std::endl;

  glfwSetWindowTitle(window_, "Vulkan - benchmark done");
}

std::vector<glm::mat4> Engine::createRandomCubeTransforms(uint32_t count) {
  // Fixed seed so the scene is the same every run
  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> positionDist(-SCENE_HALF_EXTENT, SCENE_HALF_EXTENT);
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

Engine::~Engine() {
  // Wait for all GPU operations to complete before cleanup
  volkan_.device_.waitIdle();

  // Clean up materials first
  rtMaterial_.reset();
  bruteForceMaterial_.reset();

  // Clean up brute force resources
  timestampQueryPool_ = nullptr;
  cubeBuffer_ = BufferData();

  // Clean up ray tracing resources
  tlas_ = nullptr;
  tlasBuffer_ = BufferData();
  blas_ = nullptr;
  blasBuffer_ = BufferData();
  cubeIndexBuffer_ = BufferData();
  cubeVertexBuffer_ = BufferData();

  // Clean up Volkan resources
  volkan_.cleanupRenderResources();

  glfwDestroyWindow(window_);
  glfwTerminate();
}

void Engine::run() {
  if (RENDER_MODE == RenderMode::BruteForceCompute) {
    // Renders a few frames, reports, then keeps showing the last image
    runBenchmark();
  }

  while (!glfwWindowShouldClose(window_)) {

    glfwPollEvents();
    processInput(window_);
    
    drawFrame();
  }
}

void Engine::drawFrame() {
  if (!volkan_.nextFrameReady()) {
    if (volkan_.swapchainRecreated_) {
      volkan_.swapchainRecreated_ = false;
      if (RENDER_MODE == RenderMode::HardwareRayTracing) {
        rtMaterial_->updateStorageImage(volkan_, volkan_.storageImageView_);
      } else {
        // The storage image was recreated empty, so render it again (slow!)
        bruteForceMaterial_->updateStorageImage(volkan_, volkan_.storageImageView_);
        FrameTiming timing;
        renderBruteForceFrame(timing);
      }
    }
    return;
  }

  volkan_.beginCommandBuffer();

  if (RENDER_MODE == RenderMode::HardwareRayTracing) {
    auto& frameData = volkan_.frameDatas_[volkan_.currentFrame_];
    rtMaterial_->traceRays(frameData.commandBuffer, volkan_.swapChainExtent_.width,
                           volkan_.swapChainExtent_.height, volkan_, volkan_.currentFrame_,
                           camera_);
  }
  // Brute force mode renders in tiles outside the frame loop; here we only present it
  volkan_.copyStorageImageToSwapchain();
  volkan_.submit();
}

void Engine::updateCameraFront() {
  glm::vec3 front;
  front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
  front.y = sin(glm::radians(pitch));
  front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
  camera_.front = glm::normalize(front);
}

void Engine::processInput(GLFWwindow* window) {
  updateCameraFront();
  float cameraSpeed = 0.05f;
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera_.pos += cameraSpeed * camera_.front;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera_.pos -= cameraSpeed * camera_.front;
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera_.pos -=
        glm::normalize(glm::cross(camera_.front, camera_.up)) * cameraSpeed;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera_.pos +=
        glm::normalize(glm::cross(camera_.front, camera_.up)) * cameraSpeed;
}

void Engine::createWindow(int width, int height, const char* title) {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  window_ = glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
}

void Engine::handleFramebufferResized() {
  volkan_.framebufferResized_ = true;
}


