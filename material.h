#pragma once
#include <vector>
#include <vulkan/vulkan_raii.hpp>

#include "structs.h"

// Pipeline + descriptor state shared by every renderer
class Material {
 public:
  Material() = default;
  virtual ~Material() = default;

  // Owns Vulkan handles, so it can be moved but not copied
  Material(const Material&) = delete;
  Material& operator=(const Material&) = delete;

  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;  // one per frame in flight
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline pipeline = nullptr;
};
