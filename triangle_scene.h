#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

// A triangle mesh scene (e.g. loaded from FBX), flattened into world space.
// CPU only: no Vulkan here.
struct TriangleScene {
  // 3 vec4 per triangle: xyz = vertex position; w of the first vertex = material index
  std::vector<glm::vec4> triangles;
  std::vector<glm::vec4> materialColors;  // rgb = color shown for the material
  glm::vec3 boundsMin{0.0f};
  glm::vec3 boundsMax{0.0f};

  size_t triangleCount() const { return triangles.size() / 3; }
};

// Loads every mesh instance of an FBX file into world space, in meters, with -y up
// (this project's convention). Throws std::runtime_error if the file can't be read.
TriangleScene loadFbxScene(const std::string& path);

// Uniform grid over a triangle scene. Cells are cubes; `resolution` cells span
// the longest side of the scene, and the other sides get as many as they need.
// Cell c lists the triangles overlapping it: entries[cellStart[c] .. cellStart[c+1]-1].
struct TriangleGrid {
  glm::uvec3 dims{0};       // cells per axis
  glm::vec4 gridMin{0.0f};  // xyz = min corner, w = cell size
  std::vector<uint32_t> cellStart;
  std::vector<uint32_t> entries;  // triangle indices
};

TriangleGrid buildTriangleGrid(const TriangleScene& scene, uint32_t resolution);
