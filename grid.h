#pragma once
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

// CPU-side scene and grid construction (no Vulkan here).

// A random cloud of cubes, ready for the GPU
struct CubeScene {
  std::vector<glm::mat4> transforms;     // world transform of each cube
  std::vector<glm::mat4> invTransforms;  // world -> cube space, for the box test
  std::vector<glm::vec4> spheres;        // bounding sphere: xyz = center, w = radius
};

// `count` cubes with random rotation and size (0.5-3) in a box of `halfExtent`
// around `center`. With clusterCount == 0 they are spread uniformly; otherwise
// they are split evenly into clusters placed at random in the box, each cube
// normally distributed (std dev clusterSigma) around its cluster's center.
// Fixed seeds, so the scene is the same every run.
CubeScene createCubeScene(uint32_t count, glm::vec3 center, float halfExtent,
                          uint32_t clusterCount = 0, float clusterSigma = 0.0f);

enum class GridLayout {
  ByCenter,   // each cube in the one cell holding its center; cells padded on test
  ByOverlap,  // each cube in every cell its bounding box overlaps (needed for DDA)
};

// A uniform grid with the cubes sorted by cell. Entry i of every array belongs
// to the same (cube, cell) pair; cell c owns entries cellStart[c] .. cellStart[c+1]-1.
struct GridData {
  uint32_t resolution = 0;
  glm::vec4 gridMin{0.0f};  // xyz = min corner, w = cell size
  float pad = 0.0f;         // ByCenter: largest sphere radius (cells must be grown by it)
  std::vector<uint32_t> cellStart;
  std::vector<glm::mat4> invTransforms;
  std::vector<glm::vec4> spheres;
  std::vector<uint32_t> cubeIds;  // original cube index, for its color
};

// How many (cube, cell) entries a grid would have, without building it
uint64_t countGridEntries(const CubeScene& scene, GridLayout layout, uint32_t resolution);

GridData buildGrid(const CubeScene& scene, GridLayout layout, uint32_t resolution);
