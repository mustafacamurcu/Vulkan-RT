#include "volkan.h"

#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

#include "vk_utils.h"

#define KHRONOS_VALIDATION_LAYER "VK_LAYER_KHRONOS_validation"

#ifdef NDEBUG
const bool DEBUG = false;
#else
const bool DEBUG = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              vk::DebugUtilsMessageTypeFlagsEXT messageType,
              const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

vk::DebugUtilsMessengerCreateInfoEXT debugUtils() {
  vk::DebugUtilsMessengerCreateInfoEXT createInfo;
  createInfo.messageSeverity =
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
  createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
  createInfo.pfnUserCallback = debugCallback;
  return createInfo;
}

std::vector<const char*> getRequiredExtensions() {
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char*> extensions(glfwExtensions,
                                      glfwExtensions + glfwExtensionCount);

  if (DEBUG) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

void Volkan::init(GLFWwindow* window) {
  window_ = window;
  createInstance();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  createCommandPool();
  createDescriptorPool();
  createFrameData();
  createStorageImage();
}

void Volkan::createInstance() {
  vk::ApplicationInfo appInfo("vkEngine", VK_MAKE_API_VERSION(0, 1, 3, 0),
                              "No Engine", VK_MAKE_API_VERSION(0, 1, 0, 0),
                              VK_API_VERSION_1_3);
  auto extensions = getRequiredExtensions();

#ifdef NDEBUG
  vk::StructureChain<vk::InstanceCreateInfo> instanceCreateInfoChain(
      {{}, &appInfo, {}, extensions});
#else
  if (!checkValidationLayerSupport(KHRONOS_VALIDATION_LAYER)) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  const std::vector<const char*> layer_names = {KHRONOS_VALIDATION_LAYER};
  vk::InstanceCreateInfo instanceCI;
  instanceCI.setPApplicationInfo(&appInfo);
  instanceCI.setPEnabledLayerNames(layer_names);
  instanceCI.setPEnabledExtensionNames(extensions);

  vk::StructureChain<vk::InstanceCreateInfo,
                     vk::DebugUtilsMessengerCreateInfoEXT>
      instanceCreateInfoChain(instanceCI, debugUtils());
#endif

  instance_ = context_.createInstance(
      instanceCreateInfoChain.get<vk::InstanceCreateInfo>());
  // The debug utils extension is only enabled in Debug builds
  if (DEBUG) {
    debugMessenger_ = instance_.createDebugUtilsMessengerEXT(debugUtils());
  }
}

void Volkan::createSurface() {
  VkSurfaceKHR _surface;
  glfwCreateWindowSurface(*instance_, window_, nullptr, &_surface);
  surface_ = vk::raii::SurfaceKHR(instance_, _surface);
}

void Volkan::pickPhysicalDevice() {
  vk::raii::PhysicalDevices physicalDevices(instance_);
  for (vk::raii::PhysicalDevice device : physicalDevices) {
    if (isDeviceSuitable(device, surface_, DEVICE_EXTENSIONS)) {
      physicalDevice_ = device;
      return;
    }
  }

  throw std::runtime_error("failed to find a suitable GPU!");
}

void Volkan::createLogicalDevice() {
  QueueFamilyIndices indices = findQueueFamilies(physicalDevice_, surface_);
  std::set<uint32_t> uniqueQueueFamilies = {
      indices.graphicsFamily.value(), indices.presentFamily.value()};

  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

  float queuePriority = 1.0f;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCI;
    queueCI.setQueueFamilyIndex(queueFamily);
    queueCI.setQueuePriorities(queuePriority);
    queueCreateInfos.push_back(queueCI);
  }

  vk::PhysicalDeviceFeatures deviceFeatures{};

  // Ray tracing features
  vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures;
  bufferDeviceAddressFeatures.setBufferDeviceAddress(true);

  vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures;
  accelerationStructureFeatures.setAccelerationStructure(true);
  accelerationStructureFeatures.setPNext(&bufferDeviceAddressFeatures);

  vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures;
  rayTracingPipelineFeatures.setRayTracingPipeline(true);
  rayTracingPipelineFeatures.setPNext(&accelerationStructureFeatures);

  vk::DeviceCreateInfo deviceCI;
  deviceCI.setQueueCreateInfos(queueCreateInfos);
  deviceCI.setPEnabledFeatures(&deviceFeatures);
  deviceCI.setPNext(&rayTracingPipelineFeatures);
  deviceCI.setPEnabledExtensionNames(DEVICE_EXTENSIONS);
#ifndef NDEBUG
  const std::vector<const char*>layer_names = {KHRONOS_VALIDATION_LAYER};
  deviceCI.setPEnabledLayerNames(layer_names);
#endif

  device_ = physicalDevice_.createDevice(deviceCI);

  graphicsQueue_ = device_.getQueue(indices.graphicsFamily.value(), 0);
  presentQueue_ = device_.getQueue(indices.presentFamily.value(), 0);
}

void Volkan::createSwapChain() {
  SwapChainSupportDetails swapChainSupport =
      querySwapChainSupport(physicalDevice_, surface_);

  vk::SurfaceFormatKHR surfaceFormat =
      chooseSwapSurfaceFormat(swapChainSupport.formats);
  vk::PresentModeKHR presentMode =
      chooseSwapPresentMode(swapChainSupport.presentModes);
  vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities, window_);

  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  QueueFamilyIndices indices = findQueueFamilies(physicalDevice_, surface_);
  uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                    indices.presentFamily.value()};

  vk::SharingMode imageSharingMode;
  if (indices.graphicsFamily != indices.presentFamily) {
    imageSharingMode = vk::SharingMode::eConcurrent;
  } else {
    imageSharingMode = vk::SharingMode::eExclusive;
  }

  vk::SwapchainCreateInfoKHR swapchainCI;
  swapchainCI.setSurface(*surface_);
  swapchainCI.setMinImageCount(imageCount);
  swapchainCI.setImageFormat(surfaceFormat.format);
  swapchainCI.setImageColorSpace(surfaceFormat.colorSpace);
  swapchainCI.setImageExtent(extent);
  swapchainCI.setImageArrayLayers(1);
  swapchainCI.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst);
  swapchainCI.setImageSharingMode(imageSharingMode);
  swapchainCI.setPQueueFamilyIndices(queueFamilyIndices);
  swapchainCI.setQueueFamilyIndexCount(2);
  swapchainCI.setPreTransform(swapChainSupport.capabilities.currentTransform);
  swapchainCI.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
  swapchainCI.setPresentMode(presentMode);
  swapchainCI.setClipped(true);

  swapChain_ = device_.createSwapchainKHR(swapchainCI);
  swapChainImages_ = swapChain_.getImages();
  swapChainImageFormat_ = surfaceFormat.format;
  swapChainExtent_ = extent;
}

vk::raii::DescriptorSetLayout Volkan::createDescriptorSetLayout(
    const vk::DescriptorSetLayoutCreateInfo& layoutInfo) const {
  return device_.createDescriptorSetLayout(layoutInfo);
}

vk::raii::DescriptorSets Volkan::createDescriptorSets(
    const vk::raii::DescriptorSetLayout& layout) const {
  const std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *layout);
  vk::DescriptorSetAllocateInfo allocInfo;
  allocInfo.setDescriptorPool(*descriptorPool_);
  allocInfo.setSetLayouts(layouts);
  vk::raii::DescriptorSets dsets(device_, allocInfo);
  return dsets;
}

void Volkan::createCommandPool() {
  QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice_, surface_);
  commandPool_ = device_.createCommandPool(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       queueFamilyIndices.graphicsFamily.value()});
}

void Volkan::createDescriptorPool() {
  // One descriptor set per frame in flight for each material
  vk::DescriptorPoolSize poolSizes[] = {
      {vk::DescriptorType::eStorageImage, 2 * MAX_FRAMES_IN_FLIGHT},
      {vk::DescriptorType::eStorageBuffer, 4 * MAX_FRAMES_IN_FLIGHT},
      {vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT},
  };
  descriptorPool_ = device_.createDescriptorPool(
      {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 2 * MAX_FRAMES_IN_FLIGHT, poolSizes});
}

BufferData Volkan::createBuffer(vk::DeviceSize bufferSize, vk::BufferUsageFlags usage,
                                vk::MemoryPropertyFlags properties) const {
  vk::raii::Buffer buffer(device_,
                          {{}, bufferSize, usage, vk::SharingMode::eExclusive});

  vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

  uint32_t memoryType =
      findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice_);

  // Check if buffer needs device address
  vk::MemoryAllocateInfo allocInfo(memRequirements.size, memoryType);
  vk::MemoryAllocateFlagsInfo allocFlagsInfo;
  if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
    allocFlagsInfo.setFlags(vk::MemoryAllocateFlagBits::eDeviceAddress);
    allocInfo.setPNext(&allocFlagsInfo);
  }

  vk::raii::DeviceMemory memory(device_, allocInfo);

  buffer.bindMemory(memory, 0);

  return BufferData{std::move(buffer), std::move(memory)};
}

BufferData Volkan::createDeviceLocalBuffer(const void* data, vk::DeviceSize size,
                                           vk::BufferUsageFlags usage) {
  // 1. Staging buffer in host-visible memory (system RAM): the CPU can write it
  BufferData staging = createBuffer(
      size, vk::BufferUsageFlagBits::eTransferSrc,
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  void* mapped = staging.memory.mapMemory(0, size);
  std::memcpy(mapped, data, size);
  staging.memory.unmapMemory();

  // 2. Final buffer in device-local memory (VRAM): fast for the GPU, but the CPU can't write it
  BufferData buffer = createBuffer(size, usage | vk::BufferUsageFlagBits::eTransferDst,
                                   vk::MemoryPropertyFlagBits::eDeviceLocal);

  // 3. The GPU copies staging -> VRAM across PCIe, once
  vk::raii::CommandBuffer cmdBuffer = beginSingleTimeCommands();
  cmdBuffer.copyBuffer(*staging.buffer, *buffer.buffer, vk::BufferCopy(0, 0, size));

  // Make the copied data visible to whatever reads the buffer later (shaders, AS builds)
  vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead);
  cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eAllCommands, {}, barrier, {}, {});

  endSingleTimeCommands(cmdBuffer);  // waits for the copy, so staging can be freed now
  return buffer;
}

void Volkan::createFrameData() {
  vk::raii::CommandBuffers commandBuffers(
      device_,
      {commandPool_, vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT});

  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::raii::Semaphore imageAvailableSemaphore(device_, {{}, nullptr});
    vk::raii::Semaphore renderFinishedSemaphore(device_, {{}, nullptr});
    vk::raii::Fence inFlightFence(device_, {vk::FenceCreateFlagBits::eSignaled});

    FrameData frameData({std::move(commandBuffers[i]),
                         std::move(imageAvailableSemaphore),
                         std::move(renderFinishedSemaphore),
                         std::move(inFlightFence),
                         0});
    frameDatas_.push_back(std::move(frameData));
  }
}

void Volkan::recreateSwapChain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwWaitEvents();
    glfwGetFramebufferSize(window_, &width, &height);
  }

  device_.waitIdle();
  swapChain_ = nullptr;
  storageImageView_ = nullptr;
  storageImage_ = nullptr;
  storageImageMemory_ = nullptr;
  createSwapChain();
  createStorageImage();
  swapchainRecreated_ = true;
}

bool Volkan::nextFrameReady() {
  FrameData& fd = frameDatas_[currentFrame_];
  auto _ = device_.waitForFences({fd.inFlightFence}, VK_TRUE, UINT64_MAX);

  if (framebufferResized_) {
    framebufferResized_ = false;
    recreateSwapChain();
    return false;
  }

  uint32_t image_index;
  vk::Result result =
      static_cast<vk::Result>(swapChain_.getDispatcher()->vkAcquireNextImageKHR(
          static_cast<VkDevice>(swapChain_.getDevice()),
          static_cast<VkSwapchainKHR>(*swapChain_), UINT64_MAX,
          static_cast<VkSemaphore>(*fd.imageAvailableSemaphore),
          static_cast<VkFence>(nullptr),
          &image_index));

  if (result == vk::Result::eErrorOutOfDateKHR ||
      result == vk::Result::eSuboptimalKHR) {
    recreateSwapChain();
    return false;
  } else if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }
  fd.imageIndex = image_index;
  device_.resetFences({fd.inFlightFence});
  fd.commandBuffer.reset();
  return true;
}

void Volkan::beginCommandBuffer() {
  FrameData& fd = frameDatas_[currentFrame_];
  auto& commandBuffer = fd.commandBuffer;

  commandBuffer.begin({});
}

void Volkan::submit() {
  FrameData& fd = frameDatas_[currentFrame_];
  auto& commandBuffer = fd.commandBuffer;
  // The swapchain image is written by a copy (transfer stage), so that is the
  // stage that must wait until the presentation engine has released the image
  vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eTransfer);

  commandBuffer.end();

  vk::SubmitInfo submitInfo(*fd.imageAvailableSemaphore,
                            waitDestinationStageMask, *fd.commandBuffer,
                            *fd.renderFinishedSemaphore);

  graphicsQueue_.submit(submitInfo, fd.inFlightFence);


  vk::PresentInfoKHR present_info 
    {*fd.renderFinishedSemaphore, *swapChain_, fd.imageIndex};

  vk::Result result = static_cast<vk::Result>(
    presentQueue_.getDispatcher()->vkQueuePresentKHR(
        static_cast<VkQueue>(*presentQueue_),
    reinterpret_cast<const VkPresentInfoKHR*>(&present_info)));

  if (result == vk::Result::eErrorOutOfDateKHR ||
      result == vk::Result::eSuboptimalKHR) {
    framebufferResized_ = true;
  }

  currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

vk::raii::ShaderModule Volkan::createShaderModule(const std::string& filename) const {
  std::vector<uint32_t> code = readSpirvFile(filename);
  return device_.createShaderModule({{}, code.size() * sizeof(uint32_t), code.data()});
}

vk::raii::PipelineLayout Volkan::createPipelineLayout(
    const vk::PipelineLayoutCreateInfo& layoutInfo) const {
  return device_.createPipelineLayout(layoutInfo);
}

void Volkan::copyStorageImageToSwapchain() {
  FrameData& fd = frameDatas_[currentFrame_];
  auto& commandBuffer = fd.commandBuffer;

  // Transition storage image to transfer source
  vk::ImageMemoryBarrier storageBarrier;
  storageBarrier.setOldLayout(vk::ImageLayout::eGeneral);
  storageBarrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
  storageBarrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  storageBarrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  storageBarrier.setImage(*storageImage_);
  storageBarrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  storageBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
  storageBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferRead);

  commandBuffer.pipelineBarrier(
    vk::PipelineStageFlagBits::eRayTracingShaderKHR | vk::PipelineStageFlagBits::eComputeShader,
    vk::PipelineStageFlagBits::eTransfer,
    {}, {}, {}, storageBarrier
  );

  // Transition swapchain image to transfer destination
  vk::ImageMemoryBarrier swapchainBarrier;
  swapchainBarrier.setOldLayout(vk::ImageLayout::eUndefined);
  swapchainBarrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
  swapchainBarrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  swapchainBarrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  swapchainBarrier.setImage(swapChainImages_[fd.imageIndex]);
  swapchainBarrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  swapchainBarrier.setSrcAccessMask(vk::AccessFlagBits::eNone);
  swapchainBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

  commandBuffer.pipelineBarrier(
    vk::PipelineStageFlagBits::eTopOfPipe,
    vk::PipelineStageFlagBits::eTransfer,
    {}, {}, {}, swapchainBarrier
  );

  // Copy storage image to swapchain
  vk::ImageCopy copyRegion;
  copyRegion.setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
  copyRegion.setSrcOffset({0, 0, 0});
  copyRegion.setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
  copyRegion.setDstOffset({0, 0, 0});
  copyRegion.setExtent({swapChainExtent_.width, swapChainExtent_.height, 1});

  commandBuffer.copyImage(*storageImage_, vk::ImageLayout::eTransferSrcOptimal,
                          swapChainImages_[fd.imageIndex], vk::ImageLayout::eTransferDstOptimal,
                          copyRegion);

  // Transition swapchain image to present
  vk::ImageMemoryBarrier presentBarrier;
  presentBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
  presentBarrier.setNewLayout(vk::ImageLayout::ePresentSrcKHR);
  presentBarrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  presentBarrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  presentBarrier.setImage(swapChainImages_[fd.imageIndex]);
  presentBarrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  presentBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
  presentBarrier.setDstAccessMask(vk::AccessFlagBits::eNone);

  commandBuffer.pipelineBarrier(
    vk::PipelineStageFlagBits::eTransfer,
    vk::PipelineStageFlagBits::eBottomOfPipe,
    {}, {}, {}, presentBarrier
  );

  // Transition storage image back to general
  vk::ImageMemoryBarrier storageBackBarrier;
  storageBackBarrier.setOldLayout(vk::ImageLayout::eTransferSrcOptimal);
  storageBackBarrier.setNewLayout(vk::ImageLayout::eGeneral);
  storageBackBarrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  storageBackBarrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  storageBackBarrier.setImage(*storageImage_);
  storageBackBarrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  storageBackBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferRead);
  storageBackBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderWrite);

  commandBuffer.pipelineBarrier(
    vk::PipelineStageFlagBits::eTransfer,
    vk::PipelineStageFlagBits::eRayTracingShaderKHR | vk::PipelineStageFlagBits::eComputeShader,
    {}, {}, {}, storageBackBarrier
  );
}

// Ray tracing helper implementations
vk::raii::CommandBuffer Volkan::beginSingleTimeCommands() {
  vk::raii::CommandBuffer cmdBuffer = std::move(
    vk::raii::CommandBuffers(device_, {commandPool_, vk::CommandBufferLevel::ePrimary, 1})[0]
  );
  cmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  return cmdBuffer;
}

void Volkan::endSingleTimeCommands(vk::raii::CommandBuffer& cmdBuffer) {
  cmdBuffer.end();
  vk::SubmitInfo submitInfo;
  submitInfo.setCommandBuffers(*cmdBuffer);
  graphicsQueue_.submit(submitInfo, nullptr);
  graphicsQueue_.waitIdle();
}

vk::DeviceAddress Volkan::getBufferAddress(const BufferData& bufferData) const {
  vk::BufferDeviceAddressInfo addressInfo;
  addressInfo.setBuffer(*bufferData.buffer);
  return device_.getBufferAddress(addressInfo);
}

vk::raii::AccelerationStructureKHR Volkan::createBLAS(BufferData& blasBuffer,
                                                       vk::DeviceAddress vertexBufferAddress,
                                                       uint32_t vertexCount,
                                                       vk::DeviceAddress indexBufferAddress,
                                                       uint32_t indexCount) {

  // Setup geometry
  vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  triangles.setVertexData(vertexBufferAddress);
  triangles.setVertexStride(sizeof(float) * 3);
  triangles.setMaxVertex(vertexCount - 1);
  triangles.setIndexType(vk::IndexType::eUint32);
  triangles.setIndexData(indexBufferAddress);

  vk::AccelerationStructureGeometryKHR geometry;
  geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
  geometry.setGeometry({triangles});
  geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo;
  buildInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
  buildInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
  buildInfo.setGeometries(geometry);

  uint32_t primitiveCount = indexCount / 3;

  vk::AccelerationStructureBuildSizesInfoKHR sizeInfo =
    device_.getAccelerationStructureBuildSizesKHR(
      vk::AccelerationStructureBuildTypeKHR::eDevice,
      buildInfo,
      primitiveCount
    );

  // Create BLAS buffer (caller provides)
  blasBuffer = createBuffer(
    sizeInfo.accelerationStructureSize,
    vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eDeviceLocal
  );

  // Create BLAS
  vk::AccelerationStructureCreateInfoKHR createInfo;
  createInfo.setBuffer(*blasBuffer.buffer);
  createInfo.setSize(sizeInfo.accelerationStructureSize);
  createInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
  vk::raii::AccelerationStructureKHR blas = device_.createAccelerationStructureKHR(createInfo);

  // Create scratch buffer
  BufferData scratchBuffer = createBuffer(
    sizeInfo.buildScratchSize,
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eDeviceLocal
  );

  vk::BufferDeviceAddressInfo scratchAddressInfo;
  scratchAddressInfo.setBuffer(*scratchBuffer.buffer);
  vk::DeviceAddress scratchAddress = device_.getBufferAddress(scratchAddressInfo);

  // Build BLAS
  buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
  buildInfo.setDstAccelerationStructure(*blas);
  buildInfo.setScratchData(scratchAddress);

  vk::AccelerationStructureBuildRangeInfoKHR rangeInfo;
  rangeInfo.setPrimitiveCount(primitiveCount);
  rangeInfo.setPrimitiveOffset(0);
  rangeInfo.setFirstVertex(0);
  rangeInfo.setTransformOffset(0);

  // Build on GPU
  vk::raii::CommandBuffer cmdBuffer = beginSingleTimeCommands();
  cmdBuffer.buildAccelerationStructuresKHR(buildInfo, &rangeInfo);
  endSingleTimeCommands(cmdBuffer);

  return blas;
}

vk::raii::AccelerationStructureKHR Volkan::createTLAS(BufferData& tlasBuffer,
                                                       const vk::raii::AccelerationStructureKHR& blas,
                                                       const std::vector<glm::mat4>& transforms) {
  // Get BLAS device address
  vk::AccelerationStructureDeviceAddressInfoKHR blasAddressInfo;
  blasAddressInfo.setAccelerationStructure(*blas);
  vk::DeviceAddress blasAddress = device_.getAccelerationStructureAddressKHR(blasAddressInfo);

  // One instance per transform, all referencing the same BLAS
  std::vector<vk::AccelerationStructureInstanceKHR> instances;
  instances.reserve(transforms.size());
  for (uint32_t i = 0; i < transforms.size(); i++) {
    // glm is column-major, VkTransformMatrixKHR is a row-major 3x4
    const glm::mat4& m = transforms[i];
    vk::TransformMatrixKHR transform;
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 4; col++) {
        transform.matrix[row][col] = m[col][row];
      }
    }

    vk::AccelerationStructureInstanceKHR instance;
    instance.setTransform(transform);
    instance.setInstanceCustomIndex(i);
    instance.setMask(0xFF);
    instance.setInstanceShaderBindingTableRecordOffset(0);
    instance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
    instance.setAccelerationStructureReference(blasAddress);
    instances.push_back(instance);
  }

  vk::DeviceSize instancesSize = sizeof(vk::AccelerationStructureInstanceKHR) * instances.size();

  BufferData instanceBuffer = createBuffer(
    instancesSize,
    vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
  );

  void* data = instanceBuffer.memory.mapMemory(0, instancesSize);
  memcpy(data, instances.data(), instancesSize);
  instanceBuffer.memory.unmapMemory();

  vk::DeviceAddress instanceAddress = getBufferAddress(instanceBuffer);

  // Setup TLAS geometry
  vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
  instancesData.setArrayOfPointers(false);
  instancesData.setData(instanceAddress);

  vk::AccelerationStructureGeometryKHR geometry;
  geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
  geometry.setGeometry({instancesData});

  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo;
  buildInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
  buildInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
  buildInfo.setGeometries(geometry);

  uint32_t primitiveCount = static_cast<uint32_t>(instances.size());

  vk::AccelerationStructureBuildSizesInfoKHR sizeInfo =
    device_.getAccelerationStructureBuildSizesKHR(
      vk::AccelerationStructureBuildTypeKHR::eDevice,
      buildInfo,
      primitiveCount
    );

  // Create TLAS buffer (caller provides)
  tlasBuffer = createBuffer(
    sizeInfo.accelerationStructureSize,
    vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eDeviceLocal
  );

  // Create TLAS
  vk::AccelerationStructureCreateInfoKHR createInfo;
  createInfo.setBuffer(*tlasBuffer.buffer);
  createInfo.setSize(sizeInfo.accelerationStructureSize);
  createInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
  vk::raii::AccelerationStructureKHR tlas = device_.createAccelerationStructureKHR(createInfo);

  // Create scratch buffer
  BufferData scratchBuffer = createBuffer(
    sizeInfo.buildScratchSize,
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eDeviceLocal
  );

  vk::BufferDeviceAddressInfo scratchAddressInfo;
  scratchAddressInfo.setBuffer(*scratchBuffer.buffer);
  vk::DeviceAddress scratchAddress = device_.getBufferAddress(scratchAddressInfo);

  // Build TLAS
  buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
  buildInfo.setDstAccelerationStructure(*tlas);
  buildInfo.setScratchData(scratchAddress);

  vk::AccelerationStructureBuildRangeInfoKHR rangeInfo;
  rangeInfo.setPrimitiveCount(primitiveCount);
  rangeInfo.setPrimitiveOffset(0);
  rangeInfo.setFirstVertex(0);
  rangeInfo.setTransformOffset(0);

  // Build on GPU
  vk::raii::CommandBuffer cmdBuffer = beginSingleTimeCommands();
  cmdBuffer.buildAccelerationStructuresKHR(buildInfo, &rangeInfo);
  endSingleTimeCommands(cmdBuffer);

  return tlas;
}

void Volkan::createStorageImage() {
  vk::ImageCreateInfo imageInfo;
  imageInfo.setImageType(vk::ImageType::e2D);
  imageInfo.setFormat(vk::Format::eR8G8B8A8Unorm);
  imageInfo.setExtent({swapChainExtent_.width, swapChainExtent_.height, 1});
  imageInfo.setMipLevels(1);
  imageInfo.setArrayLayers(1);
  imageInfo.setSamples(vk::SampleCountFlagBits::e1);
  imageInfo.setTiling(vk::ImageTiling::eOptimal);
  imageInfo.setUsage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc);
  imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);

  storageImage_ = device_.createImage(imageInfo);

  vk::MemoryRequirements memReqs = storageImage_.getMemoryRequirements();
  storageImageMemory_ = device_.allocateMemory({
    memReqs.size,
    findMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, physicalDevice_)
  });

  storageImage_.bindMemory(*storageImageMemory_, 0);

  vk::ImageViewCreateInfo viewInfo;
  viewInfo.setImage(*storageImage_);
  viewInfo.setViewType(vk::ImageViewType::e2D);
  viewInfo.setFormat(vk::Format::eR8G8B8A8Unorm);
  viewInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

  storageImageView_ = device_.createImageView(viewInfo);

  // Transition to general layout
  vk::raii::CommandBuffer cmdBuffer = std::move(
    vk::raii::CommandBuffers(device_, {commandPool_, vk::CommandBufferLevel::ePrimary, 1})[0]
  );

  cmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  vk::ImageMemoryBarrier barrier;
  barrier.setOldLayout(vk::ImageLayout::eUndefined);
  barrier.setNewLayout(vk::ImageLayout::eGeneral);
  barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  barrier.setImage(*storageImage_);
  barrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  barrier.setSrcAccessMask(vk::AccessFlagBits::eNone);
  barrier.setDstAccessMask(vk::AccessFlagBits::eShaderWrite);

  cmdBuffer.pipelineBarrier(
    vk::PipelineStageFlagBits::eTopOfPipe,
    vk::PipelineStageFlagBits::eRayTracingShaderKHR | vk::PipelineStageFlagBits::eComputeShader,
    {}, {}, {}, barrier
  );

  cmdBuffer.end();

  vk::SubmitInfo submitInfo;
  submitInfo.setCommandBuffers(*cmdBuffer);
  graphicsQueue_.submit(submitInfo, nullptr);
  graphicsQueue_.waitIdle();
}

vk::raii::Pipeline Volkan::createRayTracingPipeline(
    const vk::RayTracingPipelineCreateInfoKHR& pipelineInfo) const {
  return device_.createRayTracingPipelineKHR(nullptr, nullptr, pipelineInfo);
}

vk::raii::Pipeline Volkan::createComputePipeline(
    const vk::ComputePipelineCreateInfo& pipelineInfo) const {
  return device_.createComputePipeline(nullptr, pipelineInfo);
}

vk::raii::QueryPool Volkan::createTimestampQueryPool(uint32_t queryCount) const {
  vk::QueryPoolCreateInfo createInfo;
  createInfo.setQueryType(vk::QueryType::eTimestamp);
  createInfo.setQueryCount(queryCount);
  return device_.createQueryPool(createInfo);
}

float Volkan::getTimestampPeriod() const {
  return physicalDevice_.getProperties().limits.timestampPeriod;
}

std::string Volkan::getDeviceName() const {
  return std::string(physicalDevice_.getProperties().deviceName.data());
}

std::string Volkan::describeBufferMemory(const BufferData& bufferData,
                                         vk::MemoryPropertyFlags properties) const {
  // Same lookup createBuffer() does, so this reports where the buffer actually lives
  vk::MemoryRequirements memRequirements = bufferData.buffer.getMemoryRequirements();
  uint32_t typeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice_);

  vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice_.getMemoryProperties();
  const vk::MemoryType& type = memProperties.memoryTypes[typeIndex];
  const vk::MemoryHeap& heap = memProperties.memoryHeaps[type.heapIndex];

  return "type " + std::to_string(typeIndex) + " " + vk::to_string(type.propertyFlags) +
         ", heap " + std::to_string(type.heapIndex) + " (" +
         std::to_string(heap.size / (1024 * 1024)) + " MiB" +
         ((heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) ? ", VRAM)" : ", system RAM)");
}

std::vector<uint8_t> Volkan::readStorageImage() {
  vk::DeviceSize size = vk::DeviceSize(swapChainExtent_.width) * swapChainExtent_.height * 4;
  BufferData readback = createBuffer(
      size, vk::BufferUsageFlagBits::eTransferDst,
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

  vk::raii::CommandBuffer cmdBuffer = beginSingleTimeCommands();

  // Finish all writes to the image before copying it (it stays in GENERAL layout)
  vk::ImageMemoryBarrier toCopy;
  toCopy.setOldLayout(vk::ImageLayout::eGeneral);
  toCopy.setNewLayout(vk::ImageLayout::eGeneral);
  toCopy.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  toCopy.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
  toCopy.setImage(*storageImage_);
  toCopy.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  toCopy.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite);
  toCopy.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
  cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, toCopy);

  vk::BufferImageCopy region;
  region.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
  region.setImageExtent({swapChainExtent_.width, swapChainExtent_.height, 1});
  cmdBuffer.copyImageToBuffer(*storageImage_, vk::ImageLayout::eGeneral, *readback.buffer, region);

  // Make the copied bytes visible to the CPU
  vk::MemoryBarrier toHost(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead);
  cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                            {}, toHost, {}, {});
  endSingleTimeCommands(cmdBuffer);

  const uint8_t* bytes = static_cast<const uint8_t*>(readback.memory.mapMemory(0, size));
  std::vector<uint8_t> pixels(bytes, bytes + size);
  readback.memory.unmapMemory();
  return pixels;
}

vk::PhysicalDeviceRayTracingPipelinePropertiesKHR Volkan::getRayTracingProperties() const {
  vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR> props;
  props = physicalDevice_.getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  return props.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
}
