#include "StratusGraphicsDriver.h"
#include "StratusGpuBuffer.h"
#include "StratusWindow.h"
#include "StratusCommon.h"
#include "StratusGpuCommon.h"
#include "StratusLog.h"

namespace stratus {
    struct GraphicsContext {
        GraphicsConfig config;
        SDL_GLContext context;
    };

    static GraphicsContext& GetContext() {
        static GraphicsContext context;
        return context;
    }

    static void PrintGLInfo() {
        const GraphicsConfig& config = GetContext().config;
        auto & log = STRATUS_LOG << std::endl;
        log << "==================== OpenGL Information ====================" << std::endl;
        log << "\tRenderer: "                               << config.renderer << std::endl;
        log << "\tVersion: "                                << config.version << std::endl;
        log << "\tMajor, minor Version: "                   << config.majorVersion << ", " << config.minorVersion << std::endl;
        log << "\tMax anisotropy: "                         << config.maxAnisotropy << std::endl;
        log << "\tMax draw buffers: "                       << config.maxDrawBuffers << std::endl;
        log << "\tMax combined textures: "                  << config.maxCombinedTextures << std::endl;
        log << "\tMax cube map texture size: "              << config.maxCubeMapTextureSize << std::endl;
        log << "\tMax fragment uniform vectors: "           << config.maxFragmentUniformVectors << std::endl;
        log << "\tMax fragment uniform components: "        << config.maxFragmentUniformComponents << std::endl;
        log << "\tMax varying floats: "                     << config.maxVaryingFloats << std::endl;
        log << "\tMax render buffer size: "                 << config.maxRenderbufferSize << std::endl;
        log << "\tMax texture image units: "                << config.maxTextureImageUnits << std::endl;
        log << "\tMax texture size 1D: "                    << config.maxTextureSize1D2D << std::endl;
        log << "\tMax texture size 2D: "                    << config.maxTextureSize1D2D << "x" << config.maxTextureSize1D2D << std::endl;
        log << "\tMax texture size 3D: "                    << config.maxTextureSize3D << "x" << config.maxTextureSize3D << "x" << config.maxTextureSize3D << std::endl;
        log << "\tMax vertex attribs: "                     << config.maxVertexAttribs << std::endl;
        log << "\tMax vertex uniform vectors: "             << config.maxVertexUniformVectors << std::endl;
        log << "\tMax vertex uniform components: "          << config.maxVertexUniformComponents << std::endl;
        log << "\tMax viewport dims: "                      << "(" << config.maxViewportDims[0] << ", " << config.maxViewportDims[1] << ")" << std::endl;

        log << std::endl << "\t==> Compute Information" << std::endl;
        log << "\tMax compute shader storage blocks: "      << config.maxComputeShaderStorageBlocks << std::endl;
        log << "\tMax compute uniform blocks: "             << config.maxComputeUniformBlocks << std::endl;
        log << "\tMax compute uniform texture image units: "<< config.maxComputeTexImageUnits << std::endl;
        log << "\tMax compute uniform components: "         << config.maxComputeUniformComponents << std::endl;
        log << "\tMax compute atomic counters: "            << config.maxComputeAtomicCounters << std::endl;
        log << "\tMax compute atomic counter buffers: "     << config.maxComputeAtomicCounterBuffers << std::endl;
        log << "\tMax compute work group invocations: "     << config.maxComputeWorkGroupInvocations << std::endl;
        log << "\tMax compute work group count: "           << config.maxComputeWorkGroupCount[0] << "x" 
                                                            << config.maxComputeWorkGroupCount[1] << "x" 
                                                            << config.maxComputeWorkGroupCount[2] << std::endl;
        log << "\tMax compute work group size: "            << config.maxComputeWorkGroupSize[0] << "x" 
                                                            << config.maxComputeWorkGroupSize[1] << "x" 
                                                            << config.maxComputeWorkGroupSize[2] << std::endl;

        log << std::boolalpha;
        log << std::endl << "\t==> Virtual/Sparse Texture Information" << std::endl;
        const std::vector<GLenum> internalFormats = std::vector<GLenum>{GL_RGBA8, GL_RGBA16, GL_RGBA32F};
        const std::vector<std::string> strInternalFormats = std::vector<std::string>{"GL_RGBA8", "GL_RGBA16", "GL_RGBA32F"};
        for (int i = 0; i < internalFormats.size(); ++i) {
            log << "\t" << strInternalFormats[i] << std::endl;
            log << "\t\tSupports sparse (virtual) textures 2D: "  << config.supportsSparseTextures2D[i] << std::endl;
            if (config.supportsSparseTextures2D[i]) {
                log << "\t\tNum sparse (virtual) page sizes 2D: " << config.numPageSizes2D[i] << std::endl;
                log << "\t\tPreferred page size X 2D: "           << config.preferredPageSizeX2D[i] << std::endl;
                log << "\t\tPreferred page size Y 2D: "           << config.preferredPageSizeY2D[i] << std::endl;
            }
            log << "\t\tSupports sparse (virtual) textures 3D: "  << config.supportsSparseTextures3D[i] << std::endl;
            if (config.supportsSparseTextures3D[i]) {
                log << "\t\tNum sparse (virtual) page sizes 3D: " << config.numPageSizes3D[i] << std::endl;
                log << "\t\tPreferred page size X 3D: "           << config.preferredPageSizeX3D[i] << std::endl;
                log << "\t\tPreferred page size Y 3D: "           << config.preferredPageSizeY3D[i] << std::endl;
                log << "\t\tPreferred page size Z 3D: "           << config.preferredPageSizeZ3D[i] << std::endl;
            }
        }
    }

    bool GraphicsDriver::Initialize() {
        SDL_Window * window = (SDL_Window *)Window::Instance()->GetWindowObject();

        // Set the profile to core as opposed to compatibility mode
        SDL_GL_SetAttribute(SDL_GLattr::SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        // Set max/min version
        //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
        // Enable double buffering
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

        // Create the gl context
        STRATUS_LOG << "Creating OpenGL context" << std::endl;
        GraphicsContext& context = GetContext();
        context.context = SDL_GL_CreateContext(window);
        if (context.context == nullptr) {
            STRATUS_ERROR << "Unable to create a valid OpenGL context" << std::endl;
            STRATUS_ERROR << SDL_GetError() << std::endl;
            return false;
        }

        // Init gl core profile using gl3w
        if (gl3wInit()) {
            STRATUS_ERROR << "Failed to initialize core OpenGL profile" << std::endl;
            return false;
        }

        //if (!gl3wIsSupported(maxGLVersion, minGLVersion)) {
        //    STRATUS_ERROR << "[error] OpenGL 3.2 not supported" << std::endl;
        //    _isValid = false;
        //    return;
        //}

        // Query OpenGL about various different hardware capabilities
        context.config.renderer = (const char *)glGetString(GL_RENDERER);
        context.config.version = (const char *)glGetString(GL_VERSION);
        glGetIntegerv(GL_MINOR_VERSION, &context.config.minorVersion);
        glGetIntegerv(GL_MAJOR_VERSION, &context.config.majorVersion);
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &context.config.maxAnisotropy);
        glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &context.config.maxCombinedTextures);
        glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &context.config.maxCubeMapTextureSize);
        glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS, &context.config.maxFragmentUniformVectors);
        glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &context.config.maxRenderbufferSize);
        glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &context.config.maxTextureImageUnits);
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &context.config.maxTextureSize1D2D);
        glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &context.config.maxTextureSize3D);
        glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &context.config.maxVertexAttribs);
        glGetIntegerv(GL_MAX_VERTEX_UNIFORM_VECTORS, &context.config.maxVertexUniformVectors);
        glGetIntegerv(GL_MAX_DRAW_BUFFERS, &context.config.maxDrawBuffers);
        glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &context.config.maxFragmentUniformComponents);
        glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &context.config.maxVertexUniformComponents);
        glGetIntegerv(GL_MAX_VARYING_FLOATS, &context.config.maxVaryingFloats);
        glGetIntegerv(GL_MAX_VIEWPORT_DIMS, context.config.maxViewportDims);
        glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &context.config.maxComputeShaderStorageBlocks);
        glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS, &context.config.maxComputeUniformBlocks);
        glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS, &context.config.maxComputeTexImageUnits);
        glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_COMPONENTS, &context.config.maxComputeUniformComponents);
        glGetIntegerv(GL_MAX_COMPUTE_ATOMIC_COUNTERS, &context.config.maxComputeAtomicCounters);
        glGetIntegerv(GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS, &context.config.maxComputeAtomicCounterBuffers);
        glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &context.config.maxComputeWorkGroupInvocations);
        // 0, 1, 2 count for x, y and z dims
        for (int i = 0; i < 3; ++i) {
            glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i, &context.config.maxComputeWorkGroupCount[i]);
            glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &context.config.maxComputeWorkGroupSize[i]);
        }

        if (context.config.majorVersion < 4 || context.config.minorVersion < 3) {
            STRATUS_ERROR << "OpenGL version LOWER than 4.3 - this is not supported" << std::endl;
            return false;
        }

        const std::vector<GLenum> internalFormats = std::vector<GLenum>{GL_RGBA8, GL_RGBA16, GL_RGBA32F};
        for (int i = 0; i < internalFormats.size(); ++i) {
            const GLenum internalFormat = internalFormats[i];
            // Query OpenGL about sparse textures (2D)
            glGetInternalformativ(GL_TEXTURE_2D, internalFormat, GL_NUM_VIRTUAL_PAGE_SIZES_ARB, sizeof(uint32_t), &context.config.numPageSizes2D[i]);
            context.config.supportsSparseTextures2D[i] = context.config.numPageSizes2D[i] > 0;
            if (context.config.supportsSparseTextures2D) {
                // 1 * sizeof(int32_t) since we only want the first valid page size rather than all context.config.numPageSizes of them
                glGetInternalformativ(GL_TEXTURE_2D, internalFormat, GL_VIRTUAL_PAGE_SIZE_X_ARB, 1 * sizeof(int32_t), &context.config.preferredPageSizeX2D[i]);
                glGetInternalformativ(GL_TEXTURE_2D, internalFormat, GL_VIRTUAL_PAGE_SIZE_Y_ARB, 1 * sizeof(int32_t), &context.config.preferredPageSizeY2D[i]);
            }

            // Query OpenGL about sparse textures (3D)
            glGetInternalformativ(GL_TEXTURE_3D, internalFormat, GL_NUM_VIRTUAL_PAGE_SIZES_ARB, sizeof(uint32_t), &context.config.numPageSizes3D[i]);
            context.config.supportsSparseTextures3D[i] = context.config.numPageSizes3D[i] > 0;
            if (context.config.supportsSparseTextures3D) {
                // 1 * sizeof(int32_t) since we only want the first valid page size rather than all context.config.numPageSizes of them
                glGetInternalformativ(GL_TEXTURE_3D, internalFormat, GL_VIRTUAL_PAGE_SIZE_X_ARB, 1 * sizeof(int32_t), &context.config.preferredPageSizeX3D[i]);
                glGetInternalformativ(GL_TEXTURE_3D, internalFormat, GL_VIRTUAL_PAGE_SIZE_Y_ARB, 1 * sizeof(int32_t), &context.config.preferredPageSizeY3D[i]);
                glGetInternalformativ(GL_TEXTURE_3D, internalFormat, GL_VIRTUAL_PAGE_SIZE_Z_ARB, 1 * sizeof(int32_t), &context.config.preferredPageSizeZ3D[i]);
            }
        }

        PrintGLInfo();

        // Initialize GpuBuffer memory
        GpuMeshAllocator::Initialize_();

        return true;
    }

    void GraphicsDriver::Shutdown() {
        GpuMeshAllocator::Shutdown_();

        if (GetContext().context) {
            SDL_GL_DeleteContext(GetContext().context);
            GetContext().context = nullptr;
        }
    }

    void GraphicsDriver::MakeContextCurrent() {
        SDL_Window* window = (SDL_Window*)Window::Instance()->GetWindowObject();
        SDL_GL_MakeCurrent(window, GetContext().context);
    }

    void GraphicsDriver::SwapBuffers(const bool vsync) {
        if (!vsync) {
            // 0 lets it run as fast as it can
            SDL_GL_SetSwapInterval(0);
        }
        else {
            // 1 synchronizes updates with the vertical retrace
            SDL_GL_SetSwapInterval(1);
        }

        // Swap front and back buffer
        SDL_Window* window = (SDL_Window *)Window::Instance()->GetWindowObject();
        SDL_GL_SwapWindow(window);
    }

    const GraphicsConfig& GraphicsDriver::GetConfig() {
        return GetContext().config;
    }
}