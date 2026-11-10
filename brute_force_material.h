#pragma once
#include "material.h"

class Volkan;

// Most primitive GPU ray tracer: a compute shader where every pixel
// tests every cube. No acceleration structure.
class BruteForceMaterial : public Material {
 public:
  static constexpr uint32_t WORKGROUP_SIZE = 8;

  BruteForceMaterial(Volkan& volkan, BufferData& cubeBuffer,
                     vk::raii::ImageView& storageImageView);

  void dispatchTile(vk::raii::CommandBuffer& commandBuffer,
                    const BruteForcePushConstants& pushConstants,
                    uint32_t tileWidth, uint32_t tileHeight);

  void updateStorageImage(Volkan& volkan, vk::raii::ImageView& storageImageView);
};
