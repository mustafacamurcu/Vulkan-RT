#include "window.h"

#include <stdexcept>

Window::Window(uint32_t width, uint32_t height, const char* title) {
  if (!glfwInit()) {
    throw std::runtime_error("failed to initialize GLFW");
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  window_ = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title,
                             nullptr, nullptr);
  if (!window_) {
    glfwTerminate();
    throw std::runtime_error("failed to create GLFW window");
  }
}

Window::~Window() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}
