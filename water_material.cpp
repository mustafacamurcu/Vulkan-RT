#include "water_material.h"
#include "volkan.h"

#include <vulkan/vulkan_raii.hpp>

float rf() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  return dis(gen);
}


Waves createWaves() {
  float amp_factor = 0.78f;
  float freq_factor = 1.18;
  float sharpness_factor = 1.05;
  float amp = .3f;
  float freq = 0.1f;
  float sharpness = 1.2f;

  Waves waves;
  for (int i = 0; i < 10; i++) {
    Wave& wave = waves.waves[i];

    glm::vec2 dir(rf() * 1.5 - 1, rf() * 1.5 - 1);
    dir = glm::normalize(dir);
    wave.dir_x = dir.x;
    wave.dir_z = dir.y;
    wave.freq = freq;
    wave.amp = amp;
    wave.phase = rf() * 2 + 0.5;
    wave.sharpness = sharpness;

    amp *= amp_factor;
    freq *= freq_factor;
    sharpness *= sharpness_factor;
  }
  return waves;
}

WaterMaterial::WaterMaterial(Volkan& volkan, SpecData specData, vk::raii::RenderPass& renderPass) : Material(){
  // Shaders
  vk::raii::ShaderModule meshShaderModule =
      volkan.createShaderModule("shaders/mesh.spv");
  vk::raii::ShaderModule taskShaderModule =
      volkan.createShaderModule("shaders/task.spv");
  vk::raii::ShaderModule fragShaderModule =
      volkan.createShaderModule("shaders/frag.spv");

  // Specialization constants
  const std::vector<vk::SpecializationMapEntry> specMapEntries = SpecData::specMapEntries();
  vk::SpecializationInfo specInfo;
  specInfo.setMapEntries(specMapEntries);
  specInfo.setData<SpecData>(specData);

  vk::PipelineShaderStageCreateInfo taskShaderStageInfo;
  taskShaderStageInfo.setStage(vk::ShaderStageFlagBits::eTaskEXT);
  taskShaderStageInfo.setModule(*taskShaderModule);
  taskShaderStageInfo.setPName("main");
  taskShaderStageInfo.setPSpecializationInfo(&specInfo);

  vk::PipelineShaderStageCreateInfo meshShaderStageInfo;
  meshShaderStageInfo.setStage(vk::ShaderStageFlagBits::eMeshEXT);
  meshShaderStageInfo.setModule(*meshShaderModule);
  meshShaderStageInfo.setPName("main");
  meshShaderStageInfo.setPSpecializationInfo(&specInfo);

  vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
  fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment);
  fragShaderStageInfo.setModule(*fragShaderModule);
  fragShaderStageInfo.setPName("main");
  fragShaderStageInfo.setPSpecializationInfo(&specInfo);

  std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
      taskShaderStageInfo, meshShaderStageInfo, fragShaderStageInfo};

  vk::Viewport viewport = volkan.getViewport();
  vk::Rect2D scissor = volkan.getScissor();

  std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                 vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);

  vk::PipelineViewportStateCreateInfo viewportState({}, viewport, scissor);

  vk::PipelineRasterizationStateCreateInfo rasterizationState(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
      vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);

  vk::PipelineMultisampleStateCreateInfo multisampleState(
      {}, vk::SampleCountFlagBits::e1, false);

  vk::PipelineColorBlendAttachmentState colorBlendAttachment;
  colorBlendAttachment.setColorWriteMask(
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  colorBlendAttachment.setBlendEnable(false);

  vk::PipelineColorBlendStateCreateInfo colorBlendState(
      {}, false, vk::LogicOp::eCopy, colorBlendAttachment);

  vk::PipelineDepthStencilStateCreateInfo depthStencilState;
  depthStencilState.setDepthTestEnable(true);
  depthStencilState.setDepthWriteEnable(true);
  depthStencilState.setDepthCompareOp(vk::CompareOp::eLess);
  depthStencilState.setDepthBoundsTestEnable(false);
  depthStencilState.setStencilTestEnable(false);

  vk::PushConstantRange push_constant;
  push_constant.offset = 0;
  push_constant.size = sizeof(PerFrameVariables);
  push_constant.stageFlags =
      vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment;

  vk::DescriptorSetLayoutBinding layoutBinding;
  layoutBinding.setBinding(0);
  layoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
  layoutBinding.setDescriptorCount(1);
  layoutBinding.setStageFlags(vk::ShaderStageFlagBits::eMeshEXT |
                              vk::ShaderStageFlagBits::eFragment);

  vk::DescriptorSetLayoutCreateInfo layoutInfo;
  layoutInfo.setBindings(layoutBinding);
  descriptorSetLayout = volkan.createDescriptorSetLayout(layoutInfo);

  descriptorSets = volkan.createDescriptorSets(descriptorSetLayout);


  Waves waves = createWaves();
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    wavesBuffers.push_back(volkan.allocateUniformBuffer(descriptorSets[i], sizeof(Waves)));
    memcpy(wavesBuffers[i].mapped, &waves, sizeof(Waves));
  }

  vk::PipelineLayoutCreateInfo plci;
  plci.setSetLayouts(*descriptorSetLayout);
  plci.setPushConstantRanges(push_constant);
  pipelineLayout = volkan.createPipelineLayout(plci);

  vk::GraphicsPipelineCreateInfo pipelineCI;
  pipelineCI.setStages(shaderStages);
  pipelineCI.setPViewportState(&viewportState);
  pipelineCI.setPRasterizationState(&rasterizationState);
  pipelineCI.setPMultisampleState(&multisampleState);
  pipelineCI.setPDepthStencilState(&depthStencilState);
  pipelineCI.setPColorBlendState(&colorBlendState);
  pipelineCI.setPDynamicState(&dynamicState);
  pipelineCI.setLayout(*pipelineLayout);
  pipelineCI.setRenderPass(*renderPass);
  pipelineCI.setSubpass(0);

  pipeline = volkan.createPipeline(pipelineCI);
}
