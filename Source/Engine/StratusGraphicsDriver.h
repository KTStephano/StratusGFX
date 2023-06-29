#pragma once

#include <string>
#include <cstdint>

namespace stratus {
    /**
     * This contains information about a lot of the
     * OpenGL configuration params after initialization
     * takes place.
     */
    struct GraphicsConfig {
        std::string renderer;
        std::string version;
        int32_t minorVersion;
        int32_t majorVersion;
        float maxAnisotropy;
        int32_t maxDrawBuffers;
        int32_t maxCombinedTextures;
        int32_t maxCubeMapTextureSize;
        int32_t maxFragmentUniformVectors;
        int32_t maxFragmentUniformComponents;
        int32_t maxVaryingFloats;
        int32_t maxRenderbufferSize;
        int32_t maxTextureImageUnits;
        int32_t maxTextureSize1D2D;
        int32_t maxTextureSize3D;
        int32_t maxTextureSizeCubeMap;
        int32_t maxVertexAttribs;
        int32_t maxVertexUniformVectors;
        int32_t maxVertexUniformComponents;
        int32_t maxViewportDims[2];
        bool supportsSparseTextures2D[3];
        // OpenGL may allow multiple page sizes at the same time which the application can select from
        // first element: RGBA8, second element: RGBA16, third element: RGBA32
        int32_t numPageSizes2D[3];
        // "Preferred" as in it was the first on the list of OpenGL's returned page sizes, which could
        // indicate that it is the most efficient page size for the implementation to work with
        int32_t preferredPageSizeX2D[3];
        int32_t preferredPageSizeY2D[3];
        bool supportsSparseTextures3D[3];
        int32_t numPageSizes3D[3];
        int32_t preferredPageSizeX3D[3];
        int32_t preferredPageSizeY3D[3];
        int32_t preferredPageSizeZ3D[3];
        int32_t maxComputeShaderStorageBlocks;
        int32_t maxComputeUniformBlocks;
        int32_t maxComputeTexImageUnits;
        int32_t maxComputeUniformComponents;
        int32_t maxComputeAtomicCounters;
        int32_t maxComputeAtomicCounterBuffers;
        int32_t maxComputeWorkGroupInvocations;
        int32_t maxComputeWorkGroupCount[3];
        int32_t maxComputeWorkGroupSize[3];
    };

    // Initializes both the underlying graphics context as well as any
    // global GPU memory that the system will need
    struct GraphicsDriver {
        static bool Initialize();
        static void Shutdown();
        static void MakeContextCurrent();
        static void SwapBuffers(const bool vsync);
        static const GraphicsConfig& GetConfig();
    };
}