#include "engine.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void printUsage() {
  std::cout << "Usage: VKRenderer [--scene NAME] [--grid N] [--shading MODE] | --benchmark\n"
               "  No options: open the viewer on the sparse scene.\n"
               "    --scene      sparse | quadrants\n"
               "    --grid       grid resolution, e.g. 64 (default: the scene's best)\n"
               "    --shading    flat | diffuse | shadows\n"
               "  --benchmark: run the benchmark suite on the quadrant scene, then open the viewer.\n"
               "  Controls: WASD move, Q/E down/up, arrow keys turn, hold Shift to move faster.\n";
}

StartupConfig parseArgs(int argc, char** argv) {
  StartupConfig config;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage();
      std::exit(EXIT_SUCCESS);
    }
    if (arg == "--benchmark") {
      config.benchmark = true;
      continue;
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value after " + arg);
    }
    std::string value = argv[++i];
    if (arg == "--scene") {
      config.scene = value;
    } else if (arg == "--shading") {
      config.shading = value;
    } else if (arg == "--grid") {
      config.gridResolution = static_cast<uint32_t>(std::stoul(value));
    } else {
      throw std::runtime_error("unknown option " + arg + " (try --help)");
    }
  }
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Engine engine(parseArgs(argc, argv));
    engine.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Exiting normally." << std::endl;
  return EXIT_SUCCESS;
}
