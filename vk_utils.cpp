#include <vulkan/vulkan_raii.hpp>
#include <optional>
#include <set>
#include <GLFW/glfw3.h>
#include <fstream>

#include "vk_utils.h"

QueueFamilyIndices findQueueFamilies(vk::raii::PhysicalDevice device,
                                     const vk::raii::SurfaceKHR& surface) {
  QueueFamilyIndices indices;

  std::vector<vk::QueueFamilyProperties> queueFamilies =
      device.getQueueFamilyProperties();

  int i = 0;
  for (const vk::QueueFamilyProperties& queueFamily : queueFamilies) {
    if (device.getSurfaceSupportKHR(i, surface)) {
      indices.presentFamily = i;
    }
    if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.graphicsFamily = i;
    }

    if (indices.isComplete()) {
      break;
    }

    i++;
  }

  return indices;
}

bool checkDeviceExtensionSupport(vk::raii::PhysicalDevice device,
                                 const std::vector<const char*> extensions) {
  std::vector<vk::ExtensionProperties> availableExtensions =
      device.enumerateDeviceExtensionProperties();

  std::set<std::string> requiredExtensions(extensions.begin(),
                                           extensions.end());

  for (const auto& extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }

  return requiredExtensions.empty();
}

SwapChainSupportDetails querySwapChainSupport(
    vk::raii::PhysicalDevice device, const vk::raii::SurfaceKHR& surface) {
  vk::SurfaceCapabilitiesKHR capabilities =
      device.getSurfaceCapabilitiesKHR(surface);
  std::vector<vk::SurfaceFormatKHR> formats =
      device.getSurfaceFormatsKHR(surface);
  std::vector<vk::PresentModeKHR> presentModes =
      device.getSurfacePresentModesKHR(surface);

  return {capabilities, formats, presentModes};
}

bool isDeviceSuitable(vk::raii::PhysicalDevice device,
                      const vk::raii::SurfaceKHR& surface,
                      const std::vector<const char*> extensions) {
  QueueFamilyIndices indices = findQueueFamilies(device, surface);

  bool extensionsSupported = checkDeviceExtensionSupport(device, extensions);

  bool swapChainAdequate = false;
  if (extensionsSupported) {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(device, surface);
    swapChainAdequate = !swapChainSupport.formats.empty() &&
                        !swapChainSupport.presentModes.empty();
  }

  return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
  for (const auto& availableFormat : availableFormats) {
    if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
        availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
      return availableFormat;
    }
  }
  return availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& availablePresentModes) {
  for (const auto& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
      return availablePresentMode;
    }
  }

  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D chooseSwapExtent(
    const vk::SurfaceCapabilitiesKHR& capabilities, GLFWwindow* window) {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    vk::Extent2D actualExtent(width, height);

    actualExtent.width =
        std::clamp(actualExtent.width,
        capabilities.minImageExtent.width,
                    capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height,
                                      capabilities.minImageExtent.height,
                                      capabilities.maxImageExtent.height);

    return actualExtent;
  }
}

vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
                               vk::FormatFeatureFlags features,
                                vk::raii::PhysicalDevice physicalDevice) {
  for (vk::Format format : candidates) {
    vk::FormatProperties props = physicalDevice.getFormatProperties(format);
    if (props.optimalTilingFeatures & features) {
      return format;
    }
  }
  throw std::runtime_error("failed to find supported format!");
}

vk::Format findDepthFormat(vk::raii::PhysicalDevice physicalDevice) {
  return findSupportedFormat(
      {vk::Format::eD32Sfloat, vk::Format::eD32Sfloat,
       vk::Format::eD24UnormS8Uint},
      vk::FormatFeatureFlagBits::eDepthStencilAttachment, physicalDevice);
}

bool checkValidationLayerSupport(const char* layer_name) {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const auto& layerProperties : availableLayers) {
    if (strcmp(layer_name, layerProperties.layerName) == 0) {
      return true;
    }
  }

  return false;
}

std::vector<char> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }
  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

uint32_t findMemoryType(uint32_t typeFilter,
                        vk::MemoryPropertyFlags properties,
                        vk::raii::PhysicalDevice physicalDevice) {
  vk::PhysicalDeviceMemoryProperties memProperties =
      physicalDevice.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags &
                                  properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}
