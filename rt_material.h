#pragma once
#include "material.h"

class Volkan;

// Hardware ray tracing: the driver builds a BVH (BLAS/TLAS) and the RT cores traverse it
class RayTracingMaterial : public Material {
 public:
  RayTracingMaterial(const Volkan& volkan,
                     const vk::raii::AccelerationStructureKHR& tlas,
                     const vk::raii::ImageView& storageImageView);

  void traceRays(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex,
                 vk::Extent2D extent, const Volkan& volkan, const Camera& camera) const;

  void updateStorageImage(const Volkan& volkan, const vk::raii::ImageView& storageImageView);

 private:
  BufferData raygenSBT_;
  BufferData missSBT_;
  BufferData hitSBT_;
};
