#pragma once
#include "material.h"

class Volkan;

// Most primitive GPU ray tracer: a compute shader where every pixel
// tests every cube. No acceleration structure.
class BruteForceMaterial : public Material {
 public:
  static constexpr uint32_t WORKGROUP_SIZE = 8;

  BruteForceMaterial(const Volkan& volkan, const BufferData& cubeBuffer,
                     const vk::raii::ImageView& storageImageView);

  // One dispatch covering `region` pixels, starting at the offset in pushConstants.params.yz
  void dispatch(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex,
                const BruteForcePushConstants& pushConstants, vk::Extent2D region) const;

  void updateStorageImage(const Volkan& volkan, const vk::raii::ImageView& storageImageView);
};
