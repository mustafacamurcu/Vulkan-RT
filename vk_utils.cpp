#include "vk_utils.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& device,
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

bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& device,
                                 const std::vector<const char*>& extensions) {
  std::vector<vk::ExtensionProperties> availableExtensions =
      device.enumerateDeviceExtensionProperties();

  std::set<std::string> requiredExtensions(extensions.begin(),
                                           extensions.end());

  for (const auto& extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }

  return requiredExtensions.empty();
}

SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& device,
                                              const vk::raii::SurfaceKHR& surface) {
  vk::SurfaceCapabilitiesKHR capabilities =
      device.getSurfaceCapabilitiesKHR(surface);
  std::vector<vk::SurfaceFormatKHR> formats =
      device.getSurfaceFormatsKHR(surface);
  std::vector<vk::PresentModeKHR> presentModes =
      device.getSurfacePresentModesKHR(surface);

  return {capabilities, formats, presentModes};
}

bool isDeviceSuitable(const vk::raii::PhysicalDevice& device,
                      const vk::raii::SurfaceKHR& surface,
                      const std::vector<const char*>& extensions) {
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

bool checkValidationLayerSupport(const char* layerName) {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const auto& layerProperties : availableLayers) {
    if (strcmp(layerName, layerProperties.layerName) == 0) {
      return true;
    }
  }

  return false;
}

std::vector<uint32_t> readSpirvFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    // Say which file: the usual cause is running from the wrong working directory
    throw std::runtime_error("failed to open " + filename);
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  if (fileSize % sizeof(uint32_t) != 0) {
    throw std::runtime_error(filename + " is not valid SPIR-V (size not a multiple of 4)");
  }

  std::vector<uint32_t> words(fileSize / sizeof(uint32_t));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(words.data()), fileSize);
  return words;
}

uint32_t findMemoryType(uint32_t typeFilter,
                        vk::MemoryPropertyFlags properties,
                        const vk::raii::PhysicalDevice& physicalDevice) {
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
