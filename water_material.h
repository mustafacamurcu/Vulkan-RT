#pragma once
#include "material.h"

class Volkan;

class WaterMaterial : public Material {
 public:
  struct SpecData {
    int num_waves = 10;
    int ocean_width = 200;
    float ocean_cell_width = 1.0f;

    static std::vector<vk::SpecializationMapEntry> specMapEntries() {
      return {
          {0, offsetof(SpecData, num_waves), sizeof(int)},
          {1, offsetof(SpecData, ocean_width), sizeof(int)},
          {2, offsetof(SpecData, ocean_cell_width), sizeof(float)},
      };
    }
  };

  WaterMaterial(Volkan& volkan, SpecData specData,
                vk::raii::RenderPass& renderPass);
 private:
};
