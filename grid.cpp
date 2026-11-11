#include "grid.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <random>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

CubeScene createCubeScene(const std::vector<CubeRegion>& regions) {
  // The unit cube's corners are sqrt(3)/2 from its center, scaled by the cube's
  // (uniform) scale. They sit exactly on that sphere, so a 0.1% larger radius is
  // a safety margin against float rounding rejecting a ray that grazes a corner.
  constexpr float RADIUS_PADDING = 1.001f;
  const float cornerDistance = 0.5f * std::sqrt(3.0f) * RADIUS_PADDING;

  CubeScene scene;
  for (size_t r = 0; r < regions.size(); r++) {
    const CubeRegion& region = regions[r];
    // Each region has its own fixed seed, so changing one region leaves the others alone
    std::mt19937 rng(1337u + static_cast<uint32_t>(r));
    std::uniform_real_distribution<float> xDist(region.min.x, region.max.x);
    std::uniform_real_distribution<float> yDist(region.min.y, region.max.y);
    std::uniform_real_distribution<float> zDist(region.min.z, region.max.z);
    std::normal_distribution<float> clusterOffsetDist(
        0.0f, region.clusterSigma > 0.0f ? region.clusterSigma : 1.0f);
    std::uniform_real_distribution<float> angleDist(0.0f, glm::two_pi<float>());
    std::uniform_real_distribution<float> axisDist(-1.0f, 1.0f);
    // Log-uniform: every factor-of-2 range of sizes gets the same number of cubes,
    // so most cubes are small and a few are very large
    std::uniform_real_distribution<float> scaleDist(
        region.logScale ? std::log(region.minScale) : region.minScale,
        region.logScale ? std::log(region.maxScale) : region.maxScale);

    // Cluster centers, kept 3 sigma inside the region's walls
    std::vector<glm::vec3> clusterCenters;
    for (uint32_t c = 0; c < region.clusterCount; c++) {
      glm::vec3 margin(3.0f * region.clusterSigma);
      glm::vec3 lo = glm::min(region.min + margin, (region.min + region.max) * 0.5f);
      glm::vec3 hi = glm::max(region.max - margin, (region.min + region.max) * 0.5f);
      clusterCenters.push_back(glm::vec3(std::uniform_real_distribution<float>(lo.x, hi.x)(rng),
                                         std::uniform_real_distribution<float>(lo.y, hi.y)(rng),
                                         std::uniform_real_distribution<float>(lo.z, hi.z)(rng)));
    }

    for (uint32_t i = 0; i < region.count; i++) {
      glm::vec3 position;
      if (region.clusterCount == 0) {
        position = glm::vec3(xDist(rng), yDist(rng), zDist(rng));
      } else {
        position = clusterCenters[i % region.clusterCount] +
                   glm::vec3(clusterOffsetDist(rng), clusterOffsetDist(rng), clusterOffsetDist(rng));
      }

      glm::vec3 axis(axisDist(rng), axisDist(rng), axisDist(rng));
      if (glm::length(axis) < 0.001f) {
        axis = glm::vec3(0.0f, 1.0f, 0.0f);
      }
      float angle = angleDist(rng);
      float scale = region.logScale ? std::exp(scaleDist(rng)) : scaleDist(rng);

      glm::mat4 transform = glm::translate(glm::mat4(1.0f), position);
      transform = glm::rotate(transform, angle, glm::normalize(axis));
      transform = glm::scale(transform, glm::vec3(scale));

      scene.transforms.push_back(transform);
      scene.invTransforms.push_back(glm::inverse(transform));
      scene.spheres.push_back(glm::vec4(position, scale * cornerDistance));
    }
  }
  return scene;
}

namespace {

// Grid bounds, cell size and the cell range each cube is stored in
struct GridMapping {
  uint32_t resolution;
  glm::vec3 lo;
  float cellSize;
  float pad;
  GridLayout layout;

  glm::uvec3 cellCoord(const glm::vec3& point) const {
    return glm::min(glm::uvec3((point - lo) / cellSize), glm::uvec3(resolution - 1));
  }

  // Calls visit(cellIndex) for every cell the cube is stored in
  template <typename Visit>
  void forEachCell(const glm::vec4& sphere, Visit&& visit) const {
    glm::vec3 center(sphere);
    glm::uvec3 first = cellCoord(layout == GridLayout::ByCenter ? center : center - sphere.w);
    glm::uvec3 last = cellCoord(layout == GridLayout::ByCenter ? center : center + sphere.w);
    for (uint32_t z = first.z; z <= last.z; z++) {
      for (uint32_t y = first.y; y <= last.y; y++) {
        for (uint32_t x = first.x; x <= last.x; x++) {
          visit(x + resolution * (y + resolution * z));
        }
      }
    }
  }
};

GridMapping makeMapping(const CubeScene& scene, GridLayout layout, uint32_t resolution) {
  // ByCenter spans the cube centers (cells are padded by the largest radius when
  // tested); ByOverlap spans the cubes' full bounding boxes.
  glm::vec3 lo(FLT_MAX);
  glm::vec3 hi(-FLT_MAX);
  float maxRadius = 0.0f;
  for (const glm::vec4& sphere : scene.spheres) {
    float extent = layout == GridLayout::ByCenter ? 0.0f : sphere.w;
    lo = glm::min(lo, glm::vec3(sphere) - extent);
    hi = glm::max(hi, glm::vec3(sphere) + extent);
    maxRadius = std::max(maxRadius, sphere.w);
  }
  glm::vec3 size = hi - lo;
  // Cubic cells; a tiny margin keeps points on the max corner inside the last cell
  float cellSize = std::max({size.x, size.y, size.z}) / resolution * 1.0001f;
  return {resolution, lo, cellSize, layout == GridLayout::ByCenter ? maxRadius : 0.0f, layout};
}

}  // namespace

uint64_t countGridEntries(const CubeScene& scene, GridLayout layout, uint32_t resolution) {
  GridMapping mapping = makeMapping(scene, layout, resolution);
  uint64_t entries = 0;
  for (const glm::vec4& sphere : scene.spheres) {
    mapping.forEachCell(sphere, [&](uint32_t) { entries++; });
  }
  return entries;
}

GridData buildGrid(const CubeScene& scene, GridLayout layout, uint32_t resolution) {
  GridMapping mapping = makeMapping(scene, layout, resolution);
  const uint32_t cellCount = resolution * resolution * resolution;

  GridData grid;
  grid.resolution = resolution;
  grid.gridMin = glm::vec4(mapping.lo, mapping.cellSize);
  grid.pad = mapping.pad;

  // Counting sort by cell: count entries per cell, prefix-sum into start
  // offsets, then copy each cube into its cell(s).
  grid.cellStart.assign(cellCount + 1, 0);
  for (const glm::vec4& sphere : scene.spheres) {
    mapping.forEachCell(sphere, [&](uint32_t cell) { grid.cellStart[cell + 1]++; });
  }
  for (size_t c = 1; c < grid.cellStart.size(); c++) {
    grid.cellStart[c] += grid.cellStart[c - 1];
  }

  uint32_t entryCount = grid.cellStart.back();
  grid.invTransforms.resize(entryCount);
  grid.spheres.resize(entryCount);
  grid.cubeIds.resize(entryCount);
  std::vector<uint32_t> nextSlot(grid.cellStart.begin(), grid.cellStart.end() - 1);
  for (uint32_t i = 0; i < scene.spheres.size(); i++) {
    mapping.forEachCell(scene.spheres[i], [&](uint32_t cell) {
      uint32_t slot = nextSlot[cell]++;
      grid.invTransforms[slot] = scene.invTransforms[i];
      grid.spheres[slot] = scene.spheres[i];
      grid.cubeIds[slot] = i;
    });
  }
  return grid;
}
