#pragma once
#include "material.h"

#include <string>

class Volkan;

// Compute-shader ray tracer over a uniform grid of cubes. The shader decides
// how the grid is used (grid_rejection.spv or grid_traversal.spv).
class GridMaterial : public Material {
 public:
  static constexpr uint32_t WORKGROUP_SIZE = 8;

  GridMaterial(const Volkan& volkan, const std::string& shaderPath, const BufferData& cubeBuffer,
                     const BufferData& sphereBuffer, const BufferData& cellBuffer,
                     const BufferData& cubeIdBuffer,
                     const vk::raii::ImageView& storageImageView);

  // One dispatch covering `region` pixels, starting at the offset in pushConstants.params.yz
  void dispatch(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex,
                const GridPushConstants& pushConstants, vk::Extent2D region) const;

  void updateStorageImage(const Volkan& volkan, const vk::raii::ImageView& storageImageView);
};
