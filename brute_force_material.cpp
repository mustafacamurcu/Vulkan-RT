#include "brute_force_material.h"
#include "volkan.h"

BruteForceMaterial::BruteForceMaterial(const Volkan& volkan, const BufferData& cubeBuffer,
                                       const BufferData& sphereBuffer,
                                       const BufferData& cellBuffer,
                                       const BufferData& cubeIdBuffer,
                                       const vk::raii::ImageView& storageImageView) {
  // Create descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings = {
    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
  };

  vk::DescriptorSetLayoutCreateInfo layoutInfo;
  layoutInfo.setBindings(bindings);
  descriptorSetLayout = volkan.createDescriptorSetLayout(layoutInfo);

  // Allocate descriptor sets
  descriptorSets = volkan.createDescriptorSets(descriptorSetLayout);

  // Point every frame's set at the scene buffers (binding -> buffer) and the storage image
  const std::pair<uint32_t, const BufferData*> storageBuffers[] = {
    {0, &cubeBuffer}, {2, &sphereBuffer}, {3, &cellBuffer}, {4, &cubeIdBuffer}};
  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    for (const auto& [binding, bufferData] : storageBuffers) {
      vk::DescriptorBufferInfo bufferInfo(*bufferData->buffer, 0, VK_WHOLE_SIZE);

      vk::WriteDescriptorSet bufferWrite;
      bufferWrite.setDstSet(*descriptorSets[i]);
      bufferWrite.setDstBinding(binding);
      bufferWrite.setDescriptorType(vk::DescriptorType::eStorageBuffer);
      bufferWrite.setBufferInfo(bufferInfo);

      volkan.device().updateDescriptorSets(bufferWrite, {});
    }
  }
  updateStorageImage(volkan, storageImageView);

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

void BruteForceMaterial::dispatch(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex,
                                  const BruteForcePushConstants& pushConstants,
                                  vk::Extent2D region) const {
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0,
                                   *descriptorSets[frameIndex], {});
  commandBuffer.pushConstants<BruteForcePushConstants>(
      *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushConstants);

  // Round up so partial workgroups cover the image edges (the shader bounds-checks)
  uint32_t groupsX = (region.width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  uint32_t groupsY = (region.height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  commandBuffer.dispatch(groupsX, groupsY, 1);
}

void BruteForceMaterial::updateStorageImage(const Volkan& volkan,
                                            const vk::raii::ImageView& storageImageView) {
  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DescriptorImageInfo imageInfo;
    imageInfo.setImageView(*storageImageView);
    imageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet imageWrite;
    imageWrite.setDstSet(*descriptorSets[i]);
    imageWrite.setDstBinding(1);
    imageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
    imageWrite.setImageInfo(imageInfo);

    volkan.device().updateDescriptorSets(imageWrite, {});
  }
}
