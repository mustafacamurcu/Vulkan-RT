#pragma once
#include <GLFW/glfw3.h>

#include <cstdint>
#include <string>

// Owns the GLFW window (and GLFW itself). Destroyed automatically, after
// everything declared later in the owning class, so the Vulkan surface is
// always gone before the window it was created from.
class Window {
 public:
  Window(uint32_t width, uint32_t height, const char* title);
  ~Window();

  // One window, one owner: copying would destroy it twice
  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;

  GLFWwindow* handle() const { return window_; }
  bool shouldClose() const { return glfwWindowShouldClose(window_); }
  void setTitle(const std::string& title) const { glfwSetWindowTitle(window_, title.c_str()); }

 private:
  GLFWwindow* window_ = nullptr;
};
