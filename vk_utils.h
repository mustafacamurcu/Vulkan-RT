#pragma once

#include <GLFW/glfw3.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

struct SwapChainSupportDetails {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;
};

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  bool isComplete() const {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

// Read-only inputs are taken by const reference: no copies, and the
// signature promises the function won't modify them.
QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& device,
                                     const vk::raii::SurfaceKHR& surface);

bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& device,
                                 const std::vector<const char*>& extensions);

SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& device,
                                              const vk::raii::SurfaceKHR& surface);

bool isDeviceSuitable(const vk::raii::PhysicalDevice& device,
                      const vk::raii::SurfaceKHR& surface,
                      const std::vector<const char*>& extensions);

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats);

vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& availablePresentModes);

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                              GLFWwindow* window);

bool checkValidationLayerSupport(const char* layerName);

// SPIR-V is a stream of 32-bit words, so read it as uint32_t (correctly aligned)
std::vector<uint32_t> readSpirvFile(const std::string& filename);

uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties,
                        const vk::raii::PhysicalDevice& physicalDevice);
