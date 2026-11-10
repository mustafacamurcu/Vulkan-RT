#pragma once
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

#include "structs.h"
#include "material.h"

class Volkan {
public:
  void init(GLFWwindow* window);

  vk::raii::ShaderModule createShaderModule(const std::string& filename);
  vk::raii::DescriptorSetLayout createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo layoutInfo);
  vk::raii::DescriptorSets createDescriptorSets(vk::raii::DescriptorSetLayout& layout);
  vk::raii::PipelineLayout createPipelineLayout(vk::PipelineLayoutCreateInfo plci);
  vk::raii::Pipeline createRayTracingPipeline(vk::RayTracingPipelineCreateInfoKHR pipelineInfo);
  vk::raii::Pipeline createComputePipeline(vk::ComputePipelineCreateInfo pipelineInfo);

  // Buffers and acceleration structures
  BufferData createBuffer(vk::DeviceSize bufferSize,
                          vk::BufferUsageFlags usage,
                          vk::MemoryPropertyFlags properties);
  vk::DeviceAddress getBufferAddress(BufferData& bufferData);
  vk::raii::AccelerationStructureKHR createBLAS(BufferData& blasBuffer,
                                                 vk::DeviceAddress vertexBufferAddress,
                                                 uint32_t vertexCount,
                                                 vk::DeviceAddress indexBufferAddress,
                                                 uint32_t indexCount);
  vk::raii::AccelerationStructureKHR createTLAS(BufferData& tlasBuffer,
                                                 vk::raii::AccelerationStructureKHR& blas,
                                                 const std::vector<glm::mat4>& transforms);
  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR getRayTracingProperties();

  // Frame loop
  bool nextFrameReady();
  void beginCommandBuffer();
  void copyStorageImageToSwapchain();
  void submit();

  // One-off GPU work (uploads, AS builds, benchmark tiles). Blocks until done.
  vk::raii::CommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(vk::raii::CommandBuffer& cmdBuffer);

  // Profiling helpers
  vk::raii::QueryPool createTimestampQueryPool(uint32_t queryCount);
  float getTimestampPeriod();  // nanoseconds per timestamp tick
  std::string getDeviceName();
  std::string describeBufferMemory(BufferData& bufferData, vk::MemoryPropertyFlags properties);

  void cleanupRenderResources();

  struct FrameData {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailableSemaphore;
    vk::raii::Semaphore renderFinishedSemaphore;
    vk::raii::Fence inFlightFence;
    uint32_t imageIndex;
  };

  const std::vector<const char*> extensions_ = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};
  bool framebufferResized_ = false;
  bool swapchainRecreated_ = false;

private:
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

    GLFWwindow* window_;
    vk::raii::Instance instance_ = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger_ = nullptr;
    vk::raii::SurfaceKHR surface_ = nullptr;
    vk::raii::PhysicalDevice physicalDevice_ = nullptr;
    vk::raii::Queue graphicsQueue_ = nullptr;
    vk::raii::Queue presentQueue_ = nullptr;
    vk::raii::SwapchainKHR swapChain_ = nullptr;
    std::vector<vk::Image> swapChainImages_;
    vk::Format swapChainImageFormat_;
    vk::raii::CommandPool commandPool_ = nullptr;
    vk::raii::DescriptorPool descriptorPool_ = nullptr;

public:
    // Public device handle for materials to access
    vk::raii::Device device_ = nullptr;

    // Frame data and current frame (for Engine to access)
    std::vector<FrameData> frameDatas_;
    size_t currentFrame_ = 0;
    vk::Extent2D swapChainExtent_;

    // Storage image both ray tracers write into, then copied to the swapchain
    vk::raii::Image storageImage_ = nullptr;
    vk::raii::ImageView storageImageView_ = nullptr;
    vk::raii::DeviceMemory storageImageMemory_ = nullptr;
};
