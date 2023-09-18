#pragma once

#include <string>
#include <cstdint>
#include "StratusTypes.h"

namespace stratus {
#define NUM_SPARSE_INFO_LOOKUPS 5

    /**
     * This contains information about a lot of the
     * OpenGL configuration params after initialization
     * takes place.
     */
    struct GraphicsConfig {
        std::string renderer;
        std::string version;
        i32 minorVersion;
        i32 majorVersion;
        f32 maxAnisotropy;
        i32 maxDrawBuffers;
        i32 maxCombinedTextures;
        i32 maxCubeMapTextureSize;
        i32 maxFragmentUniformVectors;
        i32 maxFragmentUniformComponents;
        i32 maxVaryingFloats;
        i32 maxRenderbufferSize;
        i32 maxTextureImageUnits;
        i32 maxTextureSize1D2D;
        i32 maxTextureSize3D;
        i32 maxTextureSizeCubeMap;
        i32 maxVertexAttribs;
        i32 maxVertexUniformVectors;
        i32 maxVertexUniformComponents;
        i32 maxViewportDims[2];
        bool supportsSparseTextures2D[NUM_SPARSE_INFO_LOOKUPS];
        // OpenGL may allow multiple page sizes at the same time which the application can select from
        // first element: RGBA8, second element: RGBA16, third element: RGBA32
        i32 numPageSizes2D[NUM_SPARSE_INFO_LOOKUPS];
        // "Preferred" as in it was the first on the list of OpenGL's returned page sizes, which could
        // indicate that it is the most efficient page size for the implementation to work with
        i32 preferredPageSizeX2D[NUM_SPARSE_INFO_LOOKUPS];
        i32 preferredPageSizeY2D[NUM_SPARSE_INFO_LOOKUPS];
        bool supportsSparseTextures3D[NUM_SPARSE_INFO_LOOKUPS];
        i32 numPageSizes3D[NUM_SPARSE_INFO_LOOKUPS];
        i32 preferredPageSizeX3D[NUM_SPARSE_INFO_LOOKUPS];
        i32 preferredPageSizeY3D[NUM_SPARSE_INFO_LOOKUPS];
        i32 preferredPageSizeZ3D[NUM_SPARSE_INFO_LOOKUPS];
        i32 maxComputeShaderStorageBlocks;
        i32 maxComputeUniformBlocks;
        i32 maxComputeTexImageUnits;
        i32 maxComputeUniformComponents;
        i32 maxComputeAtomicCounters;
        i32 maxComputeAtomicCounterBuffers;
        i32 maxComputeWorkGroupInvocations;
        i32 maxComputeWorkGroupCount[3];
        i32 maxComputeWorkGroupSize[3];
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