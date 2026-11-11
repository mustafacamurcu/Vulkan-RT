#include "triangle_scene.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>

#include "ufbx.h"

namespace {

// Distinct, saturated color for index i (golden-ratio hue spacing), like the cube colors
glm::vec3 generatedColor(uint32_t index) {
  float hue = std::fmod(static_cast<float>(index) * 0.61803398875f, 1.0f);
  glm::vec3 rgb = glm::clamp(
      glm::abs(glm::mod(hue * 6.0f + glm::vec3(0.0f, 4.0f, 2.0f), 6.0f) - 3.0f) - 1.0f, 0.0f, 1.0f);
  return glm::mix(glm::vec3(1.0f), rgb, 0.6f);
}

// The color to show for a material: its base color if it has a plain one, otherwise
// (it relies on a texture, which we don't load yet) a generated color per material
glm::vec3 materialColor(const ufbx_material* material, uint32_t index) {
  const ufbx_material_map& baseColor = material->pbr.base_color;
  bool usesTexture = baseColor.texture != nullptr;
  glm::vec3 value(baseColor.value_vec4.x, baseColor.value_vec4.y, baseColor.value_vec4.z);
  bool isPlainWhite = glm::all(glm::greaterThan(value, glm::vec3(0.99f)));
  if (!baseColor.has_value || usesTexture || isPlainWhite) {
    return generatedColor(index);
  }
  return value;
}

}  // namespace

TriangleScene loadFbxScene(const std::string& path) {
  ufbx_load_opts opts = {};
  opts.target_axes = ufbx_axes_right_handed_y_up;
  opts.target_unit_meters = 1.0f;

  ufbx_error error;
  ufbx_scene* fbx = ufbx_load_file(path.c_str(), &opts, &error);
  if (!fbx) {
    char message[1024];
    ufbx_format_error(message, sizeof(message), &error);
    throw std::runtime_error("failed to load " + path + ":\n" + message);
  }

  TriangleScene scene;
  for (size_t m = 0; m < fbx->materials.count; m++) {
    scene.materialColors.push_back(glm::vec4(materialColor(fbx->materials.data[m], uint32_t(m)), 1.0f));
  }
  // Faces without a material get one extra, neutral gray material
  const uint32_t noMaterial = static_cast<uint32_t>(scene.materialColors.size());
  scene.materialColors.push_back(glm::vec4(0.7f, 0.7f, 0.7f, 1.0f));

  scene.boundsMin = glm::vec3(FLT_MAX);
  scene.boundsMax = glm::vec3(-FLT_MAX);

  // The FBX is Y-up; this project uses -y up. A 180 degree turn around x (negate y
  // and z) converts it without mirroring the scene.
  auto toWorld = [](ufbx_vec3 p) { return glm::vec3(p.x, -p.y, -p.z); };

  std::vector<uint32_t> cornerIndices;
  for (size_t n = 0; n < fbx->nodes.count; n++) {
    const ufbx_node* node = fbx->nodes.data[n];
    const ufbx_mesh* mesh = node->mesh;
    if (!mesh) {
      continue;
    }
    cornerIndices.resize(mesh->max_face_triangles * 3);

    for (size_t f = 0; f < mesh->faces.count; f++) {
      // Per-instance materials live on the node (the same mesh can be reused with others)
      uint32_t material = noMaterial;
      if (mesh->face_material.count > 0 && node->materials.count > 0) {
        uint32_t slot = mesh->face_material.data[f];
        if (slot < node->materials.count && node->materials.data[slot]) {
          material = node->materials.data[slot]->typed_id;
        }
      }

      uint32_t triangleCount = ufbx_triangulate_face(cornerIndices.data(), cornerIndices.size(),
                                                     mesh, mesh->faces.data[f]);
      for (uint32_t t = 0; t < triangleCount; t++) {
        for (uint32_t corner = 0; corner < 3; corner++) {
          ufbx_vec3 local = ufbx_get_vertex_vec3(&mesh->vertex_position, cornerIndices[t * 3 + corner]);
          glm::vec3 position = toWorld(ufbx_transform_position(&node->geometry_to_world, local));
          scene.boundsMin = glm::min(scene.boundsMin, position);
          scene.boundsMax = glm::max(scene.boundsMax, position);
          float w = corner == 0 ? static_cast<float>(material) : 0.0f;
          scene.triangles.push_back(glm::vec4(position, w));
        }
      }
    }
  }

  ufbx_free_scene(fbx);
  if (scene.triangles.empty()) {
    throw std::runtime_error(path + " contains no triangles");
  }
  return scene;
}

TriangleGrid buildTriangleGrid(const TriangleScene& scene, uint32_t resolution) {
  TriangleGrid grid;
  glm::vec3 size = scene.boundsMax - scene.boundsMin;
  // Cubic cells; a tiny margin keeps points on the max corner inside the last cell
  float cellSize = std::max({size.x, size.y, size.z}) / resolution * 1.0001f;
  grid.dims = glm::max(glm::uvec3(glm::ceil(size / cellSize)), glm::uvec3(1));
  grid.gridMin = glm::vec4(scene.boundsMin, cellSize);

  auto cellCoord = [&](const glm::vec3& point) {
    return glm::min(glm::uvec3(glm::max((point - scene.boundsMin) / cellSize, glm::vec3(0.0f))),
                    grid.dims - 1u);
  };
  // Calls visit(cellIndex) for every cell the triangle's bounding box overlaps
  auto forEachCell = [&](size_t triangle, auto&& visit) {
    glm::vec3 a(scene.triangles[triangle * 3]);
    glm::vec3 b(scene.triangles[triangle * 3 + 1]);
    glm::vec3 c(scene.triangles[triangle * 3 + 2]);
    glm::uvec3 first = cellCoord(glm::min(a, glm::min(b, c)));
    glm::uvec3 last = cellCoord(glm::max(a, glm::max(b, c)));
    for (uint32_t z = first.z; z <= last.z; z++) {
      for (uint32_t y = first.y; y <= last.y; y++) {
        for (uint32_t x = first.x; x <= last.x; x++) {
          visit(x + grid.dims.x * (y + grid.dims.y * z));
        }
      }
    }
  };

  // Counting sort by cell, as for the cube grid
  size_t cellCount = size_t(grid.dims.x) * grid.dims.y * grid.dims.z;
  grid.cellStart.assign(cellCount + 1, 0);
  for (size_t t = 0; t < scene.triangleCount(); t++) {
    forEachCell(t, [&](uint32_t cell) { grid.cellStart[cell + 1]++; });
  }
  for (size_t c = 1; c < grid.cellStart.size(); c++) {
    grid.cellStart[c] += grid.cellStart[c - 1];
  }
  grid.entries.resize(grid.cellStart.back());
  std::vector<uint32_t> nextSlot(grid.cellStart.begin(), grid.cellStart.end() - 1);
  for (size_t t = 0; t < scene.triangleCount(); t++) {
    forEachCell(t, [&](uint32_t cell) { grid.entries[nextSlot[cell]++] = static_cast<uint32_t>(t); });
  }
  return grid;
}
