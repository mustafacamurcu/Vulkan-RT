#include "brute_force_material.h"
#include "volkan.h"

#include <vulkan/vulkan_raii.hpp>

BruteForceMaterial::BruteForceMaterial(Volkan& volkan, BufferData& cubeBuffer,
                                       vk::raii::ImageView& storageImageView) : Material() {
  // Create descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings = {
    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}
  };

  vk::DescriptorSetLayoutCreateInfo layoutInfo;
  layoutInfo.setBindings(bindings);
  descriptorSetLayout = volkan.createDescriptorSetLayout(layoutInfo);

  // Allocate descriptor sets
  descriptorSets = volkan.createDescriptorSets(descriptorSetLayout);

  // Update descriptor sets (for both frames)
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DescriptorBufferInfo bufferInfo;
    bufferInfo.setBuffer(*cubeBuffer.buffer);
    bufferInfo.setOffset(0);
    bufferInfo.setRange(VK_WHOLE_SIZE);

    vk::WriteDescriptorSet bufferWrite;
    bufferWrite.setDstSet(*descriptorSets[i]);
    bufferWrite.setDstBinding(0);
    bufferWrite.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    bufferWrite.setBufferInfo(bufferInfo);

    vk::DescriptorImageInfo imageInfo;
    imageInfo.setImageView(*storageImageView);
    imageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet imageWrite;
    imageWrite.setDstSet(*descriptorSets[i]);
    imageWrite.setDstBinding(1);
    imageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
    imageWrite.setImageInfo(imageInfo);

    std::vector<vk::WriteDescriptorSet> writes = {bufferWrite, imageWrite};
    volkan.device_.updateDescriptorSets(writes, {});
  }

  // Create pipeline layout
  vk::PushConstantRange pushConstantRange(vk::ShaderStageFlagBits::eCompute, 0,
                                          sizeof(BruteForcePushConstants));

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
  pipelineLayoutInfo.setSetLayouts(*descriptorSetLayout);
  pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);
  pipelineLayout = volkan.createPipelineLayout(pipelineLayoutInfo);

  // Create compute pipeline
  vk::raii::ShaderModule computeShader = volkan.createShaderModule("shaders/brute_force.spv");

  vk::ComputePipelineCreateInfo pipelineInfo;
  pipelineInfo.setStage({{}, vk::ShaderStageFlagBits::eCompute, *computeShader, "main"});
  pipelineInfo.setLayout(*pipelineLayout);

  pipeline = volkan.createComputePipeline(pipelineInfo);
}

void BruteForceMaterial::dispatchTile(vk::raii::CommandBuffer& commandBuffer,
                                      const BruteForcePushConstants& pushConstants,
                                      uint32_t tileWidth, uint32_t tileHeight) {
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0,
                                   *descriptorSets[0], {});
  commandBuffer.pushConstants<BruteForcePushConstants>(
      *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushConstants);

  uint32_t groupsX = (tileWidth + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  uint32_t groupsY = (tileHeight + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  commandBuffer.dispatch(groupsX, groupsY, 1);
}

void BruteForceMaterial::updateStorageImage(Volkan& volkan, vk::raii::ImageView& storageImageView) {
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
