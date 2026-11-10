#pragma once
#include <vulkan/vulkan_raii.hpp>

#include "vk_utils.h"
#include "structs.h"

class Material {
 public:
  Material() = default;
  virtual ~Material() = default;

  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline pipeline = nullptr;
};