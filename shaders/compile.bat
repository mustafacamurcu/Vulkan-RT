C:/VulkanSDK/1.4.309.0/Bin/glslc.exe --target-spv=spv1.5 shader.vert -o vert.spv
C:/VulkanSDK/1.4.309.0/Bin/glslc.exe --target-spv=spv1.5 shader.frag -o frag.spv
C:/VulkanSDK/1.4.309.0/Bin/glslc.exe --target-spv=spv1.5 shader.mesh -o mesh.spv
C:/VulkanSDK/1.4.309.0/Bin/glslc.exe --target-spv=spv1.5 shader.task -o task.spv
C:/VulkanSDK/1.4.309.0/Bin/glslc.exe --target-spv=spv1.5 --target-env=vulkan1.2 raygen.rgen -o raygen.spv
C:/VulkanSDK/1.4.309.0/Bin/glslc.exe --target-spv=spv1.5 --target-env=vulkan1.2 miss.rmiss -o miss.spv
C:/VulkanSDK/1.4.309.0/Bin/glslc.exe --target-spv=spv1.5 --target-env=vulkan1.2 closesthit.rchit -o closesthit.spv
pause