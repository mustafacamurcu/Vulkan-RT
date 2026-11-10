#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <optional>
#include <set>
#include <GLFW/glfw3.h>

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

QueueFamilyIndices findQueueFamilies(vk::raii::PhysicalDevice device,
                                     const vk::raii::SurfaceKHR& surface);

bool checkDeviceExtensionSupport(vk::raii::PhysicalDevice device,
                                 const std::vector<const char*> extensions);

SwapChainSupportDetails querySwapChainSupport(
    vk::raii::PhysicalDevice device, const vk::raii::SurfaceKHR& surface);

bool isDeviceSuitable(vk::raii::PhysicalDevice device,
                      const vk::raii::SurfaceKHR& surface,
                      const std::vector<const char*> extensions);

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats);

vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& availablePresentModes);

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                              GLFWwindow* window);

vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
                               vk::FormatFeatureFlags features,
                               vk::raii::PhysicalDevice physicalDevice);

vk::Format findDepthFormat(vk::raii::PhysicalDevice physicalDevice);

bool checkValidationLayerSupport(const char* layer_name);

std::vector<char> readFile(const std::string& filename);

uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties,
                        vk::raii::PhysicalDevice physicalDevice);
