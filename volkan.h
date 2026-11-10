#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_raii.hpp>
#include <random>

#include "structs.h"
#include "material.h"

class Volkan {
public:
  void init(GLFWwindow* window);

  vk::Viewport getViewport();
  vk::Rect2D getScissor();
  vk::raii::ShaderModule createShaderModule(const std::string& filename);
  vk::raii::DescriptorSetLayout createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo layoutInfo);
  vk::raii::DescriptorSets createDescriptorSets(vk::raii::DescriptorSetLayout& layout);
  vk::raii::Pipeline createPipeline(vk::GraphicsPipelineCreateInfo gpci);
  vk::raii::PipelineLayout createPipelineLayout(vk::PipelineLayoutCreateInfo plci);
  BufferData allocateUniformBuffer(vk::raii::DescriptorSet& dset, uint64_t size);

  // Ray tracing helpers
  BufferData createBuffer(vk::DeviceSize bufferSize,
                          vk::BufferUsageFlags usage,
                          vk::MemoryPropertyFlags properties);
  vk::raii::AccelerationStructureKHR createBLAS(BufferData& blasBuffer,
                                                 vk::DeviceAddress vertexBufferAddress,
                                                 uint32_t vertexCount,
                                                 vk::DeviceAddress indexBufferAddress,
                                                 uint32_t indexCount);
  vk::raii::AccelerationStructureKHR createTLAS(BufferData& tlasBuffer,
                                                 vk::raii::AccelerationStructureKHR& blas,
                                                 const std::vector<glm::mat4>& transforms);
  vk::DeviceAddress getBufferAddress(BufferData& bufferData);
  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR getRayTracingProperties();
  vk::raii::Image createImage(vk::ImageCreateInfo imageInfo);
  vk::raii::DeviceMemory allocateImageMemory(vk::Image image);
  void transitionImageLayout(vk::raii::CommandBuffer& cmdBuffer,
                             vk::Image image,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout,
                             vk::PipelineStageFlags srcStage,
                             vk::PipelineStageFlags dstStage);

  bool nextFrameReady();

  void beginCommandBuffer();
  void beginRenderPass();
  void bindPipeline(Camera camera, glm::float32 time, Material& material);
  void drawMesh();
  void submit();

  // Ray tracing rendering helpers
  void copyStorageImageToSwapchain();
  vk::raii::CommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(vk::raii::CommandBuffer& cmdBuffer);

  vk::raii::Pipeline createMeshPipeline();
  vk::raii::Pipeline createRayTracingPipeline(vk::RayTracingPipelineCreateInfoKHR pipelineInfo);
  vk::raii::Pipeline createComputePipeline(vk::ComputePipelineCreateInfo pipelineInfo);

  // Profiling helpers
  vk::raii::QueryPool createTimestampQueryPool(uint32_t queryCount);
  float getTimestampPeriod();  // nanoseconds per timestamp tick
  std::string getDeviceName();
  std::string describeBufferMemory(BufferData& bufferData, vk::MemoryPropertyFlags properties);

  
  struct FrameData {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailableSemaphore;
    vk::raii::Semaphore renderFinishedSemaphore;
    vk::raii::Fence inFlightFence;
    uint32_t imageIndex;
  };
  
  const std::vector<const char*> extensions_ = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    // VK_EXT_MESH_SHADER_EXTENSION_NAME, // Not used for ray tracing
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};
  const uint32_t WIDTH = 1920;
  const uint32_t HEIGHT = 1080;
  bool framebufferResized_ = false;
  bool swapchainRecreated_ = false;

  vk::raii::RenderPass renderPass_ = nullptr;

  void cleanupRenderResources();

private:
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createSpecialization();
    void createCommandPool();
    void createDepthResources();
    void createFramebuffers();
    void createDescriptorPool();
    void createFrameData();
    void createWaves();
    void createStorageImage();

    BufferData createUniformBuffer(uint64_t size);

    void recreateSwapChain();

    void updatePushConstants(FrameData& fd);

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
    std::vector<vk::raii::ImageView> swapChainImageViews_;
    std::vector<vk::raii::Framebuffer> swapChainFramebuffers_;
    vk::raii::CommandPool commandPool_ = nullptr;
    vk::raii::DescriptorPool descriptorPool_ = nullptr;
    vk::raii::Image depthImage_ = nullptr;
    vk::raii::ImageView depthImageView_ = nullptr;
    vk::raii::DeviceMemory depthImageMemory_ = nullptr;

public:
    // Public device handle for materials to access
    vk::raii::Device device_ = nullptr;

    // Frame data and current frame (for Engine to access)
    std::vector<FrameData> frameDatas_;
    size_t currentFrame_ = 0;
    vk::Extent2D swapChainExtent_;

    // Ray tracing storage image (shared rendering resource)
    vk::raii::Image storageImage_ = nullptr;
    vk::raii::ImageView storageImageView_ = nullptr;
    vk::raii::DeviceMemory storageImageMemory_ = nullptr;
};
