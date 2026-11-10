# Optimizing RT
Hardware: RTX 3060 at 1920x1080.

## Brute Force
A compute shader runs one thread per pixel, and every thread tests its ray against every cube. There is no acceleration structure, and the cube data sits in system RAM.


| Run | Cubes | GPU time per frame |
| --- | --- | --- |
| Estimated | 1,000,000 | 64.6 s |

![Brute force render of 10,000 cubes](docs/brute_force.png)

## Staging Buffer
The cube data is copied into VRAM once, through a staging buffer. Before, the GPU fetched it from system RAM over PCIe every time its 3 MB L2 cache missed, which happens constantly with 61 MiB of cubes. Now the 61 MiB crosses PCIe once (~3 ms) and every cache miss goes to VRAM instead.

| Run | Cubes | GPU time per frame |
| --- | --- | --- |
| Brute Force | 1,000,000 | 64.6 s |
| Staging Buffer | 1,000,000 | 39.7 s |
