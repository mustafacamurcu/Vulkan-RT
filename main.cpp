#include "engine.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void printUsage() {
  std::cout << "Usage: VKRenderer [--view NAME --method NAME --grid N]\n"
               "  No options: run the full benchmark suite, then render interactively.\n"
               "  With options: skip the suite and render one configuration (e.g. for profiling).\n"
               "    --view    dense-inside | dense-outside | sparse-inside | clustered\n"
               "    --method  grid-rejection | grid-traversal\n"
               "    --grid    grid resolution, e.g. 64\n";
}

StartupConfig parseArgs(int argc, char** argv) {
  StartupConfig config;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage();
      std::exit(EXIT_SUCCESS);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value after " + arg);
    }
    std::string value = argv[++i];
    if (arg == "--view") {
      config.view = value;
    } else if (arg == "--method") {
      config.method = value;
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
