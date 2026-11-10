#pragma once
#include <GLFW/glfw3.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

#include "structs.h"

// Owns the Vulkan instance, device, swapchain and per-frame resources.
class Volkan {
public:
  void init(GLFWwindow* window);

  // Object creation. These don't change Volkan's own state, so they're const.
  vk::raii::ShaderModule createShaderModule(const std::string& filename) const;
  vk::raii::DescriptorSetLayout createDescriptorSetLayout(const vk::DescriptorSetLayoutCreateInfo& layoutInfo) const;
  vk::raii::DescriptorSets createDescriptorSets(const vk::raii::DescriptorSetLayout& layout) const;
  vk::raii::PipelineLayout createPipelineLayout(const vk::PipelineLayoutCreateInfo& layoutInfo) const;
  vk::raii::Pipeline createRayTracingPipeline(const vk::RayTracingPipelineCreateInfoKHR& pipelineInfo) const;
  vk::raii::Pipeline createComputePipeline(const vk::ComputePipelineCreateInfo& pipelineInfo) const;

  // Buffers and acceleration structures
  BufferData createBuffer(vk::DeviceSize bufferSize,
                          vk::BufferUsageFlags usage,
                          vk::MemoryPropertyFlags properties) const;
  // Fast-for-the-GPU buffer in VRAM, filled through a temporary staging buffer
  BufferData createDeviceLocalBuffer(const void* data, vk::DeviceSize size,
                                     vk::BufferUsageFlags usage);
  vk::DeviceAddress getBufferAddress(const BufferData& bufferData) const;
  vk::raii::AccelerationStructureKHR createBLAS(BufferData& blasBuffer,
                                                 vk::DeviceAddress vertexBufferAddress,
                                                 uint32_t vertexCount,
                                                 vk::DeviceAddress indexBufferAddress,
                                                 uint32_t indexCount);
  vk::raii::AccelerationStructureKHR createTLAS(BufferData& tlasBuffer,
                                                 const vk::raii::AccelerationStructureKHR& blas,
                                                 const std::vector<glm::mat4>& transforms);
  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR getRayTracingProperties() const;

  // Frame loop: nextFrameReady -> beginCommandBuffer -> record -> copyStorageImageToSwapchain -> submit
  bool nextFrameReady();
  void beginCommandBuffer();
  void copyStorageImageToSwapchain();
  void submit();

  // One-off GPU work (uploads, AS builds). Blocks until done.
  vk::raii::CommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(vk::raii::CommandBuffer& cmdBuffer);

  // Profiling helpers
  vk::raii::QueryPool createTimestampQueryPool(uint32_t queryCount) const;
  float getTimestampPeriod() const;  // nanoseconds per timestamp tick
  std::string getDeviceName() const;
  std::string describeBufferMemory(const BufferData& bufferData, vk::MemoryPropertyFlags properties) const;
  // Copies the storage image to the CPU (RGBA8, row by row), for correctness checks
  std::vector<uint8_t> readStorageImage();

  // Accessors: callers can use these resources but not replace them
  const vk::raii::Device& device() const { return device_; }
  vk::Extent2D extent() const { return swapChainExtent_; }
  const vk::raii::ImageView& storageImageView() const { return storageImageView_; }
  vk::raii::CommandBuffer& currentCommandBuffer() { return frameDatas_[currentFrame_].commandBuffer; }
  uint32_t currentFrameIndex() const { return currentFrame_; }

  void notifyFramebufferResized() { framebufferResized_ = true; }
  // True once after the swapchain (and storage image) was recreated
  bool consumeSwapchainRecreated() {
    bool recreated = swapchainRecreated_;
    swapchainRecreated_ = false;
    return recreated;
  }

private:
  struct FrameData {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailableSemaphore;
    vk::raii::Semaphore renderFinishedSemaphore;
    vk::raii::Fence inFlightFence;
    uint32_t imageIndex;
  };

  inline static const std::vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};

  void createInstance();
  void createSurface();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSwapChain();
  void createCommandPool();
  void createDescriptorPool();
  void createFrameData();
  void createStorageImage();

  void recreateSwapChain();

  // Members are destroyed in REVERSE declaration order, so each object is
  // declared after the objects it depends on. That makes destruction order
  // correct automatically: no manual cleanup function needed.
  GLFWwindow* window_ = nullptr;
  vk::raii::Context context_;
  vk::raii::Instance instance_ = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger_ = nullptr;
  vk::raii::SurfaceKHR surface_ = nullptr;
  vk::raii::PhysicalDevice physicalDevice_ = nullptr;
  vk::raii::Device device_ = nullptr;
  vk::raii::Queue graphicsQueue_ = nullptr;
  vk::raii::Queue presentQueue_ = nullptr;
  vk::raii::CommandPool commandPool_ = nullptr;
  vk::raii::DescriptorPool descriptorPool_ = nullptr;
  std::vector<FrameData> frameDatas_;  // command buffers come from commandPool_

  vk::raii::SwapchainKHR swapChain_ = nullptr;
  std::vector<vk::Image> swapChainImages_;  // owned by the swapchain
  vk::Format swapChainImageFormat_ = vk::Format::eUndefined;
  vk::Extent2D swapChainExtent_;

  // Storage image both ray tracers write into, then copied to the swapchain
  vk::raii::DeviceMemory storageImageMemory_ = nullptr;
  vk::raii::Image storageImage_ = nullptr;
  vk::raii::ImageView storageImageView_ = nullptr;

  uint32_t currentFrame_ = 0;
  bool framebufferResized_ = false;
  bool swapchainRecreated_ = false;
};
