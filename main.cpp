#include "engine.h"
#include <iostream>

int main() {
  
  try {
    Engine engine;
    engine.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Exiting normally." << std::endl;

  return EXIT_SUCCESS;
}