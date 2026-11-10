#include "rt_material.h"
#include "volkan.h"

#include <vulkan/vulkan_raii.hpp>

RayTracingMaterial::RayTracingMaterial(Volkan& volkan,
                                       vk::raii::AccelerationStructureKHR& tlas,
                                       vk::raii::ImageView& storageImageView) : Material() {
  // Create descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings = {
    {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},
    {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR}
  };

  vk::DescriptorSetLayoutCreateInfo layoutInfo;
  layoutInfo.setBindings(bindings);
  descriptorSetLayout = volkan.createDescriptorSetLayout(layoutInfo);

  // Allocate descriptor sets
  descriptorSets = volkan.createDescriptorSets(descriptorSetLayout);

  // Update descriptor sets (for both frames)
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::WriteDescriptorSetAccelerationStructureKHR asInfo;
    asInfo.setAccelerationStructures(*tlas);

    vk::WriteDescriptorSet asWrite;
    asWrite.setDstSet(*descriptorSets[i]);
    asWrite.setDstBinding(0);
    asWrite.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
    asWrite.setDescriptorCount(1);
    asWrite.setPNext(&asInfo);

    vk::DescriptorImageInfo imageInfo;
    imageInfo.setImageView(*storageImageView);
    imageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet imageWrite;
    imageWrite.setDstSet(*descriptorSets[i]);
    imageWrite.setDstBinding(1);
    imageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
    imageWrite.setImageInfo(imageInfo);

    std::vector<vk::WriteDescriptorSet> writes = {asWrite, imageWrite};
    volkan.device_.updateDescriptorSets(writes, {});
  }

  // Create pipeline layout
  vk::PushConstantRange pushConstantRange(vk::ShaderStageFlagBits::eRaygenKHR, 0,
                                          sizeof(RayTracingPushConstants));

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
  pipelineLayoutInfo.setSetLayouts(*descriptorSetLayout);
  pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);
  pipelineLayout = volkan.createPipelineLayout(pipelineLayoutInfo);

  // Load shaders
  vk::raii::ShaderModule raygenShader = volkan.createShaderModule("shaders/raygen.spv");
  vk::raii::ShaderModule missShader = volkan.createShaderModule("shaders/miss.spv");
  vk::raii::ShaderModule chitShader = volkan.createShaderModule("shaders/closesthit.spv");

  // Shader stages
  std::vector<vk::PipelineShaderStageCreateInfo> stages = {
    {{}, vk::ShaderStageFlagBits::eRaygenKHR, *raygenShader, "main"},
    {{}, vk::ShaderStageFlagBits::eMissKHR, *missShader, "main"},
    {{}, vk::ShaderStageFlagBits::eClosestHitKHR, *chitShader, "main"}
  };

  // Shader groups
  std::vector<vk::RayTracingShaderGroupCreateInfoKHR> groups = {
    {vk::RayTracingShaderGroupTypeKHR::eGeneral, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
    {vk::RayTracingShaderGroupTypeKHR::eGeneral, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
    {vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR}
  };

  // Create ray tracing pipeline
  vk::RayTracingPipelineCreateInfoKHR pipelineInfo;
  pipelineInfo.setStages(stages);
  pipelineInfo.setGroups(groups);
  pipelineInfo.setMaxPipelineRayRecursionDepth(1);
  pipelineInfo.setLayout(*pipelineLayout);

  pipeline = volkan.createRayTracingPipeline(pipelineInfo);

  // Create SBT
  auto rtProps = volkan.getRayTracingProperties();
  uint32_t handleSize = rtProps.shaderGroupHandleSize;
  uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
  uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;

  auto alignUp = [](uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
  };

  uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

  // Get shader group handles
  uint32_t groupCount = 3;
  size_t sbtSize = groupCount * handleSizeAligned;
  std::vector<uint8_t> handles = pipeline.getRayTracingShaderGroupHandlesKHR<uint8_t>(0, groupCount, sbtSize);

  // Create SBT buffers
  raygenSBT_ = volkan.createBuffer(
    alignUp(handleSizeAligned, baseAlignment),
    vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
  );

  missSBT_ = volkan.createBuffer(
    alignUp(handleSizeAligned, baseAlignment),
    vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
  );

  hitSBT_ = volkan.createBuffer(
    alignUp(handleSizeAligned, baseAlignment),
    vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
  );

  // Copy handles to SBT buffers
  void* raygenData = raygenSBT_.memory.mapMemory(0, handleSizeAligned);
  memcpy(raygenData, handles.data(), handleSize);
  raygenSBT_.memory.unmapMemory();

  void* missData = missSBT_.memory.mapMemory(0, handleSizeAligned);
  memcpy(missData, handles.data() + handleSizeAligned, handleSize);
  missSBT_.memory.unmapMemory();

  void* hitData = hitSBT_.memory.mapMemory(0, handleSizeAligned);
  memcpy(hitData, handles.data() + 2 * handleSizeAligned, handleSize);
  hitSBT_.memory.unmapMemory();
}

void RayTracingMaterial::traceRays(vk::raii::CommandBuffer& commandBuffer,
                                   uint32_t width, uint32_t height, Volkan& volkan,
                                   size_t currentFrameIndex, const Camera& camera) {
  auto rtProps = volkan.getRayTracingProperties();
  uint32_t handleSize = rtProps.shaderGroupHandleSize;
  uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
  uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;

  auto alignUp = [](uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
  };

  uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);
  uint32_t alignedSize = alignUp(handleSizeAligned, baseAlignment);

  // Get buffer addresses
  vk::BufferDeviceAddressInfo raygenAddrInfo;
  raygenAddrInfo.setBuffer(*raygenSBT_.buffer);
  vk::DeviceAddress raygenAddr = volkan.device_.getBufferAddress(raygenAddrInfo);

  vk::BufferDeviceAddressInfo missAddrInfo;
  missAddrInfo.setBuffer(*missSBT_.buffer);
  vk::DeviceAddress missAddr = volkan.device_.getBufferAddress(missAddrInfo);

  vk::BufferDeviceAddressInfo hitAddrInfo;
  hitAddrInfo.setBuffer(*hitSBT_.buffer);
  vk::DeviceAddress hitAddr = volkan.device_.getBufferAddress(hitAddrInfo);

  // Setup SBT regions
  vk::StridedDeviceAddressRegionKHR raygenRegion;
  raygenRegion.setDeviceAddress(raygenAddr);
  raygenRegion.setStride(alignedSize);
  raygenRegion.setSize(alignedSize);

  vk::StridedDeviceAddressRegionKHR missRegion;
  missRegion.setDeviceAddress(missAddr);
  missRegion.setStride(alignedSize);
  missRegion.setSize(alignedSize);

  vk::StridedDeviceAddressRegionKHR hitRegion;
  hitRegion.setDeviceAddress(hitAddr);
  hitRegion.setStride(alignedSize);
  hitRegion.setSize(alignedSize);

  vk::StridedDeviceAddressRegionKHR callableRegion{};

  // Bind pipeline and descriptor sets
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0,
                                   *descriptorSets[currentFrameIndex], {});

  RayTracingPushConstants pushConstants;
  pushConstants.invView = glm::inverse(camera.getViewMatrix());
  pushConstants.invProj = glm::inverse(camera.getProjectionMatrix(width / (float)height));
  commandBuffer.pushConstants<RayTracingPushConstants>(
      *pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, pushConstants);

  // Trace rays
  commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, callableRegion,
                             width, height, 1);
}

void RayTracingMaterial::updateStorageImage(Volkan& volkan, vk::raii::ImageView& storageImageView) {
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DescriptorImageInfo imageInfo;
    imageInfo.setImageView(*storageImageView);
    imageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet imageWrite;
    imageWrite.setDstSet(*descriptorSets[i]);
    imageWrite.setDstBinding(1);
    imageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
    imageWrite.setImageInfo(imageInfo);

    volkan.device_.updateDescriptorSets({imageWrite}, {});
  }
}
