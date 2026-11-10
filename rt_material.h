#pragma once
#include "material.h"

class Volkan;

class RayTracingMaterial : public Material {
 public:
  RayTracingMaterial(Volkan& volkan,
                     vk::raii::AccelerationStructureKHR& tlas,
                     vk::raii::ImageView& storageImageView);

  void traceRays(vk::raii::CommandBuffer& commandBuffer,
                 uint32_t width, uint32_t height, Volkan& volkan,
                 size_t currentFrameIndex, const Camera& camera);

  void updateStorageImage(Volkan& volkan, vk::raii::ImageView& storageImageView);

 private:
  BufferData raygenSBT_;
  BufferData missSBT_;
  BufferData hitSBT_;
};
