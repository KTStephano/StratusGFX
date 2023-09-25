
#include <iostream>
#include <StratusLight.h>
#include "StratusGpuCommon.h"
#include "StratusPipeline.h"
#include "StratusRendererBackend.h"
#include <math.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <ctime>
#include "StratusUtils.h"
#include "StratusMath.h"
#include "StratusLog.h"
#include "StratusResourceManager.h"
#include "StratusApplicationThread.h"
#include "StratusEngine.h"
#include "StratusWindow.h"
#include "StratusGraphicsDriver.h"
#include "StratusGpuBindings.h"

namespace stratus {
bool IsRenderable(const EntityPtr& p) {
    return p->Components().ContainsComponent<RenderComponent>();
}

bool IsLightInteracting(const EntityPtr& p) {
    auto component = p->Components().GetComponent<LightInteractionComponent>();
    return component.status == EntityComponentStatus::COMPONENT_ENABLED;
}

usize GetMeshCount(const EntityPtr& p) {
    return p->Components().GetComponent<RenderComponent>().component->GetMeshCount();
}

static MeshPtr GetMesh(const EntityPtr& p, const usize meshIndex) {
    return p->Components().GetComponent<RenderComponent>().component->GetMesh(meshIndex);
}

static MeshPtr GetMesh(const RenderMeshContainerPtr& p) {
    return p->render->GetMesh(p->meshIndex);
}

static MaterialPtr GetMeshMaterial(const RenderMeshContainerPtr& p) {
    return p->render->GetMaterialAt(p->meshIndex);
}

static const glm::mat4& GetMeshTransform(const RenderMeshContainerPtr& p) {
    return p->transform->transforms[p->meshIndex];
}

// See https://www.khronos.org/opengl/wiki/Debug_Output
void OpenGLDebugCallback(GLenum source, GLenum type, GLuint id,
                         GLenum severity, GLsizei length, const GLchar * message, const void * userParam) {
    //if (severity == GL_DEBUG_SEVERITY_MEDIUM || severity == GL_DEBUG_SEVERITY_HIGH) {
    if (severity == GL_DEBUG_SEVERITY_HIGH) {
       std::cout << "[OpenGL] " << message << std::endl;
    }
}

RendererBackend::RendererBackend(const u32 width, const u32 height, const std::string& appName) {
    static_assert(sizeof(GpuVec) == 16, "Memory alignment must match up with GLSL");

    isValid_ = true;

    // Set up OpenGL debug logging
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(OpenGLDebugCallback, nullptr);

    if (GraphicsDriver::GetConfig().majorVersion != 4 || GraphicsDriver::GetConfig().minorVersion != 6) {
        throw std::runtime_error("Unable to initialize renderer - driver does not support OpenGL 4.6");
    }

    const std::filesystem::path shaderRoot("../Source/Shaders");
    const ShaderApiVersion version{GraphicsDriver::GetConfig().majorVersion, GraphicsDriver::GetConfig().minorVersion};

    // Initialize the pipelines
    state_.depthPrepass = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"depth_prepass.vs", ShaderType::VERTEX}, 
        Shader{"depth_prepass.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.depthPrepass.get());

    state_.geometry = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"pbr_geometry_pass.vs", ShaderType::VERTEX}, 
        Shader{"pbr_geometry_pass.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.geometry.get());

    state_.forward = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"flat_forward_pass.vs", ShaderType::VERTEX}, 
        Shader{"flat_forward_pass.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.forward.get());

    using namespace std;

    state_.skybox = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"skybox.vs", ShaderType::VERTEX}, 
        Shader{"skybox.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.skybox.get());

    state_.skyboxLayered = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"skybox.vs", ShaderType::VERTEX},
        Shader{"skybox.fs", ShaderType::FRAGMENT} },
        // Defines
        { {"USE_LAYERED_RENDERING", "1"} }));
    state_.shaders.push_back(state_.skyboxLayered.get());

    // Set up the hdr/gamma postprocessing shader

    state_.gammaTonemap = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"gammaTonemap.vs", ShaderType::VERTEX},
        Shader{"gammaTonemap.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.gammaTonemap.get());

    // Set up the shadow preprocessing shaders
    state_.shadows = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"shadow.vs", ShaderType::VERTEX},
        //Shader{"shadow.gs", ShaderType::GEOMETRY},
        Shader{"shadow.fs", ShaderType::FRAGMENT}}
    ));
    state_.shaders.push_back(state_.shadows.get());

    state_.vplShadows = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"shadow.vs", ShaderType::VERTEX},
        //Shader{"shadow.gs", ShaderType::GEOMETRY},
        Shader{"shadowVpl.fs", ShaderType::FRAGMENT}}
    ));
    state_.shaders.push_back(state_.vplShadows.get());

    state_.lighting = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"pbr.vs", ShaderType::VERTEX},
        Shader{"pbr.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.lighting.get());

    state_.lightingWithInfiniteLight = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"pbr.vs", ShaderType::VERTEX},
        Shader{"pbr.fs", ShaderType::FRAGMENT} },
        {{"INFINITE_LIGHTING_ENABLED", "1"}}));
    state_.shaders.push_back(state_.lightingWithInfiniteLight.get());

    state_.bloom = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"bloom.vs", ShaderType::VERTEX},
        Shader{"bloom.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.bloom.get());

    for (i32 i = 0; i < 6; ++i) {
        state_.csmDepth.push_back(std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
            Shader{"csm.vs", ShaderType::VERTEX},
            //Shader{"csm.gs", ShaderType::GEOMETRY},
            Shader{"csm.fs", ShaderType::FRAGMENT}},
            // Defines
            {{"DEPTH_LAYER", std::to_string(i)}}))
        );
        state_.shaders.push_back(state_.csmDepth[i].get());

        state_.csmDepthRunAlphaTest.push_back(std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
            Shader{"csm.vs", ShaderType::VERTEX},
            //Shader{"csm.gs", ShaderType::GEOMETRY},
            Shader{"csm.fs", ShaderType::FRAGMENT}},
            // Defines
            {{"DEPTH_LAYER", std::to_string(i)},
             {"RUN_CSM_ALPHA_TEST", "1"}}))
        );
        state_.shaders.push_back(state_.csmDepthRunAlphaTest[i].get());
    }

    state_.ssaoOcclude = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"ssao.vs", ShaderType::VERTEX},
        Shader{"ssao.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.ssaoOcclude.get());

    state_.ssaoBlur = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        // Intentionally reuse ssao.vs since it works for both this and ssao.fs
        Shader{"ssao.vs", ShaderType::VERTEX},
        Shader{"ssao_blur.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.ssaoBlur.get());

    state_.atmospheric = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"atmospheric.vs", ShaderType::VERTEX},
        Shader{"atmospheric.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.atmospheric.get());

    state_.atmosphericPostFx = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"atmospheric_postfx.vs", ShaderType::VERTEX},
        Shader{"atmospheric_postfx.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.atmosphericPostFx.get());

    state_.vplCulling = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"viscull_vpls.cs", ShaderType::COMPUTE}}));
    state_.shaders.push_back(state_.vplCulling.get());

    state_.vsmCull = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"viscull_vsm.cs", ShaderType::COMPUTE}}));
    state_.shaders.push_back(state_.vsmCull.get());

    state_.vsmMarkScreen = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vsm_mark_screen.cs", ShaderType::COMPUTE}}));
    state_.shaders.push_back(state_.vsmMarkScreen.get());

    state_.vsmClear = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vsm_clear.cs", ShaderType::COMPUTE}}));
    state_.shaders.push_back(state_.vsmClear.get());

    state_.vplColoring = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vpl_light_color.cs", ShaderType::COMPUTE}}));
    state_.shaders.push_back(state_.vplColoring.get());

    state_.vplTileDeferredCullingStage1 = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vpl_tiled_deferred_culling_stage1.cs", ShaderType::COMPUTE}}));
    state_.shaders.push_back(state_.vplTileDeferredCullingStage1.get());

    state_.vplTileDeferredCullingStage2 = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vpl_tiled_deferred_culling_stage2.cs", ShaderType::COMPUTE}}));
    state_.shaders.push_back(state_.vplTileDeferredCullingStage2.get());

    state_.vplGlobalIllumination = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vpl_pbr_gi.vs", ShaderType::VERTEX},
        Shader{"vpl_pbr_gi.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.vplGlobalIllumination.get());

    state_.vplGlobalIlluminationDenoising = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vpl_pbr_gi.vs", ShaderType::VERTEX},
        Shader{"vpl_pbr_gi_denoise.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.vplGlobalIlluminationDenoising.get());

    state_.fxaaLuminance = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"fxaa.vs", ShaderType::VERTEX},
        Shader{"fxaa_luminance.fs", ShaderType::FRAGMENT}
    }));
    state_.shaders.push_back(state_.fxaaLuminance.get());

    state_.fxaaSmoothing = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"fxaa.vs", ShaderType::VERTEX},
        Shader{"fxaa_smoothing.fs", ShaderType::FRAGMENT}
    }));
    state_.shaders.push_back(state_.fxaaSmoothing.get());

    state_.taa = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"taa.vs", ShaderType::VERTEX},
        Shader{"taa.fs", ShaderType::FRAGMENT}
    }));
    state_.shaders.push_back(state_.taa.get());

    state_.aabbDraw = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"aabb_draw.vs", ShaderType::VERTEX},
        Shader{"aabb_draw.fs", ShaderType::FRAGMENT}
    }));
    state_.shaders.push_back(state_.aabbDraw.get());

    state_.fullscreen = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"fullscreen.vs", ShaderType::VERTEX},
        Shader{"fullscreen.fs", ShaderType::FRAGMENT}
        }));
    state_.shaders.push_back(state_.fullscreen.get());

    state_.fullscreenPages = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"fullscreen.vs", ShaderType::VERTEX},
        Shader{"fullscreen_pages.fs", ShaderType::FRAGMENT}
        }));
    state_.shaders.push_back(state_.fullscreenPages.get());

    state_.fullscreenPageGroups = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"fullscreen.vs", ShaderType::VERTEX},
        Shader{"fullscreen_page_groups.fs", ShaderType::FRAGMENT}
        }));
    state_.shaders.push_back(state_.fullscreenPageGroups.get());

    state_.viscullPointLights = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"viscull_point_lights.cs", ShaderType::COMPUTE} }));
    state_.shaders.push_back(state_.viscullPointLights.get());

    state_.vsmAnalyzeDepth = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vsm_analyze_depth.cs", ShaderType::COMPUTE} }));
    state_.shaders.push_back(state_.vsmAnalyzeDepth.get());

    state_.vsmMarkPages = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vsm_mark_pages.cs", ShaderType::COMPUTE} }));
    state_.shaders.push_back(state_.vsmMarkPages.get());

    state_.vsmFreePages = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vsm_free_pages.cs", ShaderType::COMPUTE} }));
    state_.shaders.push_back(state_.vsmFreePages.get());

    state_.depthPyramidConstruct = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"hzb_construct.cs", ShaderType::COMPUTE} }));
    state_.shaders.push_back(state_.depthPyramidConstruct.get());

    // Create skybox cube
    state_.skyboxCube = ResourceManager::Instance()->CreateCube();

    // Create the screen quad
    state_.screenQuad = ResourceManager::Instance()->CreateQuad();

    // Use the shader isValid() method to determine if everything succeeded
    ValidateAllShaders_();

    // Init constant SSAO data
    InitSSAO_();

    // Init constant atmospheric data
    InitAtmosphericShadowing_();

    // Create a pool of shadow maps for point lights to use
    InitPointShadowMaps_();

    // Virtual point lights
    InitializeVplData_();

    // Initialize Halton sequence
    if (haltonSequence.size() * sizeof(std::pair<f32, f32>) != haltonSequence.size() * sizeof(GpuHaltonEntry)) {
        throw std::runtime_error("Halton sequence size check failed");
    }
    haltonSequence_ = GpuBuffer((const void *)haltonSequence.data(), sizeof(GpuHaltonEntry) * haltonSequence.size(), GPU_DYNAMIC_DATA);

    // Initialize per light draw calls
    state_.dynamicPerPointLightDrawCalls.resize(6);
    state_.staticPerPointLightDrawCalls.resize(6);
    for (usize i = 0; i < state_.dynamicPerPointLightDrawCalls.size(); ++i) {
        state_.dynamicPerPointLightDrawCalls[i] = GpuCommandReceiveManager::Create();
        state_.staticPerPointLightDrawCalls[i]  = GpuCommandReceiveManager::Create();
    }
}

void RendererBackend::InitPointShadowMaps_() {
    // Create the normal point shadow map cache
    smapCache_ = CreateShadowMap3DCache_(state_.shadowCubeMapX, state_.shadowCubeMapY, state_.numRegularShadowMaps, false, TextureComponentSize::BITS_16);

    // Initialize the point light buffers including shadow map texture buffer
    const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
    state_.nonShadowCastingPointLights = GpuBuffer(nullptr, sizeof(GpuPointLight) * state_.maxTotalRegularLightsPerFrame, flags);
    state_.shadowIndices = GpuBuffer(nullptr, sizeof(GpuAtlasEntry) * state_.maxShadowCastingLightsPerFrame, flags);
    state_.shadowCastingPointLights = GpuBuffer(nullptr, sizeof(GpuPointLight) * state_.maxShadowCastingLightsPerFrame, flags);

    STRATUS_LOG << "Size: " << smapCache_.buffers.size() << std::endl;

    // Create the virtual point light shadow map cache
    vplSmapCache_ = CreateShadowMap3DCache_(state_.vpls.vplShadowCubeMapX, state_.vpls.vplShadowCubeMapY, MAX_TOTAL_VPL_SHADOW_MAPS, true, TextureComponentSize::BITS_16);
    state_.vpls.shadowDiffuseIndices = GpuBuffer(nullptr, sizeof(GpuAtlasEntry) * MAX_TOTAL_VPL_SHADOW_MAPS, flags);

    std::vector<GpuTextureHandle> diffuseHandles;
    std::vector<GpuTextureHandle> shadowHandles;
    diffuseHandles.reserve(vplSmapCache_.buffers.size());
    shadowHandles.reserve(vplSmapCache_.buffers.size());

    state_.vpls.vplDiffuseHandles.clear();
    state_.vpls.vplShadowHandles.clear();

    // Make resident
    for (auto& fbo : vplSmapCache_.buffers) {
        state_.vpls.vplDiffuseHandles.push_back(TextureMemResidencyGuard(fbo.GetColorAttachments()[0]));
        state_.vpls.vplShadowHandles.push_back(TextureMemResidencyGuard(*fbo.GetDepthStencilAttachment()));

        diffuseHandles.push_back(fbo.GetColorAttachments()[0].GpuHandle());
        shadowHandles.push_back(fbo.GetDepthStencilAttachment()->GpuHandle());
    }

    state_.vpls.vplDiffuseMaps = GpuBuffer((const void *)diffuseHandles.data(), sizeof(GpuTextureHandle) * diffuseHandles.size(), flags);
    state_.vpls.vplShadowMaps = GpuBuffer((const void *)shadowHandles.data(), sizeof(GpuTextureHandle) * shadowHandles.size(), flags);

    state_.vpls.vplDiffuseMaps.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_DIFFUSE_MAP_BINDING_POINT);
    state_.vpls.vplShadowMaps.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_SHADOW_MAP_BINDING_POINT);

    STRATUS_LOG << "Size: " << vplSmapCache_.buffers.size() << std::endl;
}

void RendererBackend::InitializeVplData_() {
    const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
    // +1 since we store the total size of the visibility array at the first index
    std::vector<i32> visibleIndicesData(MAX_TOTAL_VPLS_BEFORE_CULLING + 1, 0);
    state_.vpls.vplVisibleIndices = GpuBuffer((const void *)visibleIndicesData.data(), sizeof(i32) * visibleIndicesData.size(), flags);
    state_.vpls.vplData = GpuBuffer(nullptr, sizeof(GpuVplData) * MAX_TOTAL_VPLS_BEFORE_CULLING, flags);
    state_.vpls.vplUpdatedData = GpuBuffer(nullptr, sizeof(GpuVplData) * MAX_TOTAL_VPLS_PER_FRAME, flags);
    //state_.vpls.vplNumVisible = GpuBuffer(nullptr, sizeof(i32), flags);
}

void RendererBackend::ValidateAllShaders_() {
    isValid_ = ValidateAllPipelines(state_.shaders);
}

RendererBackend::~RendererBackend() {
    for (Pipeline * shader : shaders_) delete shader;
    shaders_.clear();

    // Delete the main frame buffer
    ClearGBuffer_();
}

void RendererBackend::RecompileShaders() {
    for (Pipeline* p : state_.shaders) {
        p->Recompile();
    }
    ValidateAllShaders_();
}

bool RendererBackend::Valid() const {
    return isValid_;
}

const Pipeline *RendererBackend::GetCurrentShader() const {
    return nullptr;
}

void RendererBackend::RecalculateCascadeData_() {
    const u32 cascadeResolutionXY = frame_->vsmc.cascadeResolutionXY;
    const u32 numCascades = frame_->vsmc.cascades.size();
    if (frame_->vsmc.regenerateFbo || !frame_->vsmc.fbo.Valid()) {
        STRATUS_LOG << "Regenerating Cascade Data" << std::endl;

        const auto numVsmMemoryPools = 2;

        // Create the depth buffer
        // @see https://stackoverflow.com/questions/22419682/glsl-sampler2dshadow-and-shadow2d-clarificationssss
        frame_->vsmc.vsm = Texture(
            TextureConfig{ 
                TextureType::TEXTURE_2D_ARRAY, 
                TextureComponentFormat::RED, 
                TextureComponentSize::BITS_32, 
                TextureComponentType::FLOAT, 
                frame_->vsmc.cascadeResolutionXY, 
                frame_->vsmc.cascadeResolutionXY, 
                numVsmMemoryPools, 
                false, 
                false // true for hardware sparse 
            }, 
                
            NoTextureData
        );

        if (frame_->vsmc.vsm.Depth() != numVsmMemoryPools) {
            throw std::runtime_error("Error: Requested number of VSM memory pools not matched");
        }

        frame_->vsmc.vsm.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        frame_->vsmc.vsm.SetCoordinateWrapping(TextureCoordinateWrapping::REPEAT);
        // We need to set this when using sampler2DShadow in the GLSL shader
        frame_->vsmc.vsm.SetTextureCompare(TextureCompareMode::COMPARE_REF_TO_TEXTURE, TextureCompareFunc::LEQUAL);

        // Create the frame buffer
        //frame_->vsmc.fbo = FrameBuffer({ tex }, frame_->vsmc.cascadeResolutionXY, frame_->vsmc.cascadeResolutionXY); 
        frame_->vsmc.fbo = FrameBuffer(std::vector<Texture>(), frame_->vsmc.cascadeResolutionXY, frame_->vsmc.cascadeResolutionXY);

        const u32 numPages = frame_->vsmc.cascadeResolutionXY / Texture::VirtualPageSizeXY();
        const u32 numPagesSquared = numCascades * numPages * numPages;

        frame_->vsmc.numPageGroupsX = numPages;
        frame_->vsmc.numPageGroupsY = numPages;

        std::vector<GpuPageResidencyEntry, StackBasedPoolAllocator<GpuPageResidencyEntry>> pageResidencyData(
            numPagesSquared, GpuPageResidencyEntry(), StackBasedPoolAllocator<GpuPageResidencyEntry>(frame_->perFrameScratchMemory)
        );

        const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;

        //frame_->vsmc.prevFramePageResidencyTable = GpuBuffer((const void *)pageResidencyData.data(), sizeof(GpuPageResidencyEntry) * numPagesSquared, flags);
        frame_->vsmc.pageResidencyTable = GpuBuffer((const void *)pageResidencyData.data(), sizeof(GpuPageResidencyEntry) * numPagesSquared, flags);

        //std::vector<u8, StackBasedPoolAllocator<u8>> pageStatusData(
        //    numPagesSquared, u8(0), StackBasedPoolAllocator<u8>(frame_->perFrameScratchMemory)
        //);

        //frame_->vsmc.pageStatusTable = GpuBuffer((const void *)pageStatusData.data(), sizeof(u8) * numPagesSquared, flags);

        i32 value = 0;
        // frame_->vsmc.numDrawCalls = GpuBuffer((const void *)&value, sizeof(int), flags);
        frame_->vsmc.numPagesToCommit = GpuBuffer((const void *)&value, sizeof(i32), flags);
        frame_->vsmc.pagesToCommitList = GpuBuffer(nullptr, 3 * sizeof(i32) * numPagesSquared, flags);

        frame_->vsmc.numPagesFree = GpuBuffer((const void *)&value, sizeof(i32), flags);

        std::vector<u32, StackBasedPoolAllocator<u32>> pageFreeList(
            StackBasedPoolAllocator<u32>(frame_->perFrameScratchMemory)
        );
        pageFreeList.reserve(numVsmMemoryPools * 3 * numPages * numPages);

        for (u32 cascade = 0; cascade < numVsmMemoryPools; ++cascade) {
            for (u32 y = 0; y < numPages; ++y) {
                for (u32 x = 0; x < numPages; ++x) {
                    pageFreeList.push_back(cascade);
                    pageFreeList.push_back(x);
                    pageFreeList.push_back(y);
                }
            }
        }

        frame_->vsmc.pagesFreeList = GpuBuffer((const void* )pageFreeList.data(), sizeof(u32) * pageFreeList.size(), flags);

        const auto numPageGroups = numCascades * frame_->vsmc.numPageGroupsX * frame_->vsmc.numPageGroupsY;
        std::vector<u32> pagesGroupsToRender(numPageGroups, 0);
        frame_->vsmc.pageGroupsToRender = GpuBuffer((const void *)pagesGroupsToRender.data(), sizeof(u32) * numPageGroups, flags);

        //frame_->vsmc.pageGroupUpdateQueue = MakeUnsafe<VirtualIndex2DUpdateQueue>(frame_->vsmc.numPageGroupsX, frame_->vsmc.numPageGroupsY);
        //frame_->vsmc.backPageGroupUpdateQueue = MakeUnsafe<VirtualIndex2DUpdateQueue>(frame_->vsmc.numPageGroupsX, frame_->vsmc.numPageGroupsY);
        frame_->vsmc.pageGroupUpdateQueue = std::vector<UnsafePtr<VirtualIndex2DUpdateQueue>>(numCascades);
        frame_->vsmc.backPageGroupUpdateQueue = std::vector<UnsafePtr<VirtualIndex2DUpdateQueue>>(numCascades);

        for (usize cascade = 0; cascade < numCascades; ++cascade) {
            frame_->vsmc.pageGroupUpdateQueue[cascade] = MakeUnsafe<VirtualIndex2DUpdateQueue>(frame_->vsmc.numPageGroupsX, frame_->vsmc.numPageGroupsY);
            frame_->vsmc.backPageGroupUpdateQueue[cascade] = MakeUnsafe<VirtualIndex2DUpdateQueue>(frame_->vsmc.numPageGroupsX, frame_->vsmc.numPageGroupsY);
        }

        frame_->vsmc.pageBoundingBox = GpuBuffer(nullptr, 4 * numCascades * sizeof(i32), flags);
    }

    frame_->vsmc.regenerateFbo = false;
}

void RendererBackend::ClearGBuffer_() {
    state_.currentFrame = GBuffer();
    state_.gaussianBuffers.clear();
    state_.postFxBuffers.clear();
}

void RendererBackend::InitGBuffer_() {
    // Regenerate the main frame buffer
    ClearGBuffer_();

    std::vector<GBuffer *> buffers = {
        &state_.currentFrame,
        &state_.previousFrame
    };

    // Reset depth pyramid texture
    state_.depthPyramid = Texture();

    for (GBuffer* gbptr : buffers) {
        GBuffer& buffer = *gbptr;

        // Position buffer
        //buffer.position = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, NoTextureData);
        //buffer.position.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

        // Normal buffer
        buffer.normals = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.normals.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        buffer.normals.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create the color buffer - notice that is uses higher
        // than normal precision. This allows us to write color values
        // greater than 1.0 to support things like HDR.
        buffer.albedo = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.albedo.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        buffer.albedo.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Base reflectivity buffer
        // buffer.baseReflectivity = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RED, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        // buffer.baseReflectivity.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        // buffer.baseReflectivity.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Roughness-Metallic-Ambient buffer
        buffer.roughnessMetallicReflectivity = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.roughnessMetallicReflectivity.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        buffer.roughnessMetallicReflectivity.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create the Structure buffer which contains rgba where r=partial x-derivative of camera-space depth, g=partial y-derivative of camera-space depth, b=16 bits of depth, a=final 16 bits of depth (b+a=32 bits=depth)
        //buffer.structure = Texture(TextureConfig{ TextureType::TEXTURE_RECTANGLE, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.structure = Texture(TextureConfig{ TextureType::TEXTURE_RECTANGLE, TextureComponentFormat::RED, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.structure.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        buffer.structure.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create velocity buffer
        // TODO: Determine best bit depth - apparently we tend to need higher precision since these values can be consistently super small
        buffer.velocity = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RG, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.velocity.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.velocity.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Holds mesh ids
        buffer.id = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::UINT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.id.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.id.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create the depth buffer
        buffer.depth = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.depth.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.depth.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create the frame buffer with all its texture attachments
        //buffer.fbo = FrameBuffer({buffer.position, buffer.normals, buffer.albedo, buffer.baseReflectivity, buffer.roughnessMetallicAmbient, buffer.structure, buffer.depth});
        buffer.fbo = FrameBuffer({ buffer.normals, buffer.albedo, buffer.roughnessMetallicReflectivity, buffer.structure, buffer.velocity, buffer.id, buffer.depth });
        if (!buffer.fbo.Valid()) {
            isValid_ = false;
            return;
        }

        buffer.fbo.Clear(glm::vec4(0.0f));

        buffer.depthPyramid = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RED, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, true }, NoTextureData);
        buffer.depthPyramid.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.depthPyramid.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    }
}

void RendererBackend::UpdateWindowDimensions_() {
    if ( !frame_->viewportDirty ) return;
    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    // Set up VPL tile data
    const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
    const i32 totalTiles = (frame_->viewportWidth / state_.vpls.tileXDivisor) * (frame_->viewportHeight / state_.vpls.tileYDivisor);
    std::vector<GpuVplStage2PerTileOutputs> data(totalTiles, GpuVplStage2PerTileOutputs());
    state_.vpls.vplStage1Results = GpuBuffer(nullptr, sizeof(GpuVplStage1PerTileOutputs) * totalTiles, flags);
    state_.vpls.vplVisiblePerTile = GpuBuffer((const void *)data.data(), sizeof(GpuVplStage2PerTileOutputs) * totalTiles, flags);

    // Re-initialize the GBuffer
    InitGBuffer_();

    // Initialize previous frame buffer
    Texture frame = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    frame.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    frame.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    state_.previousFrameBuffer = FrameBuffer({frame});
    if (!state_.previousFrameBuffer.Valid()) {
        isValid_ = false;
        return;
    }

    // Code to create the lighting fbo
    state_.lightingColorBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.lightingColorBuffer.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    state_.lightingColorBuffer.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    // Create the buffer we will use to add bloom as a post-processing effect
    // state_.lightingHighBrightnessBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    // state_.lightingHighBrightnessBuffer.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    // state_.lightingHighBrightnessBuffer.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    // Create the depth buffer
    state_.lightingDepthBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.lightingDepthBuffer.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);

    // Attach the textures to the FBO
    //state_.lightingFbo = FrameBuffer({state_.lightingColorBuffer, state_.lightingHighBrightnessBuffer, state_.lightingDepthBuffer});
    state_.lightingFbo = FrameBuffer({state_.lightingColorBuffer, state_.lightingDepthBuffer});
    if (!state_.lightingFbo.Valid()) {
        isValid_ = false;
        return;
    }

    state_.flatPassFboCurrentFrame = FrameBuffer({state_.lightingColorBuffer, state_.currentFrame.velocity, state_.lightingDepthBuffer});
    state_.flatPassFboPreviousFrame = FrameBuffer({state_.lightingColorBuffer, state_.currentFrame.velocity, state_.lightingDepthBuffer});
        if (!state_.flatPassFboCurrentFrame.Valid() || !state_.flatPassFboPreviousFrame.Valid()) {
        isValid_ = false;
        return;
    }

    // Code to create the SSAO fbo
    state_.ssaoOcclusionTexture = Texture(TextureConfig{TextureType::TEXTURE_RECTANGLE, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.ssaoOcclusionTexture.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
    state_.ssaoOcclusionTexture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.ssaoOcclusionBuffer = FrameBuffer({state_.ssaoOcclusionTexture});
    if (!state_.ssaoOcclusionBuffer.Valid()) {
        isValid_ = false;
        return;
    }

    // Code to create the SSAO blurred fbo
    state_.ssaoOcclusionBlurredTexture = Texture(TextureConfig{TextureType::TEXTURE_RECTANGLE, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.ssaoOcclusionBlurredTexture.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
    state_.ssaoOcclusionBlurredTexture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.ssaoOcclusionBlurredBuffer = FrameBuffer({state_.ssaoOcclusionBlurredTexture});
    if (!state_.ssaoOcclusionBlurredBuffer.Valid()) {
        isValid_ = false;
        return;
    }

    // Code to create the Virtual Point Light Global Illumination fbo
    Texture texture = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    texture.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    Texture texture2 = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    texture2.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
    texture2.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    state_.vpls.vplGIFbo = FrameBuffer({texture, texture2});
    if (!state_.vpls.vplGIFbo.Valid()) {
        isValid_ = false;
        return;
    }

    texture = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    texture.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    texture2 = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
    texture2.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture2.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    Texture texture3 = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    texture3.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture3.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    Texture texture4 = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
    texture4.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture4.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    state_.vpls.vplGIDenoisedFbo1 = FrameBuffer({ texture, texture2, texture3, texture4 });
    if (!state_.vpls.vplGIDenoisedFbo1.Valid()) {
        isValid_ = false;
        return;
    }

    texture = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    texture.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    texture2 = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
    texture2.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture2.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    texture3 = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    texture3.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture3.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    texture4 = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
    texture4.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture4.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    state_.vpls.vplGIDenoisedFbo2 = FrameBuffer({ texture, texture2, texture3, texture4 });
    if (!state_.vpls.vplGIDenoisedFbo2.Valid()) {
        isValid_ = false;
        return;
    }

    texture = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
    texture.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    texture2 = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
    texture2.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture2.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    texture3 = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    texture3.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture3.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    texture4 = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
    texture4.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    texture4.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    state_.vpls.vplGIDenoisedPrevFrameFbo = FrameBuffer({ texture, texture2, texture3, texture4 });
    if (!state_.vpls.vplGIDenoisedPrevFrameFbo.Valid()) {
        isValid_ = false;
        return;
    }

    // Code to create the Atmospheric fbo
    state_.atmosphericTexture = Texture(TextureConfig{TextureType::TEXTURE_RECTANGLE, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth / 2, frame_->viewportHeight / 2, 0, false}, NoTextureData);
    state_.atmosphericTexture.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    state_.atmosphericTexture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.atmosphericFbo = FrameBuffer({state_.atmosphericTexture});
    if (!state_.atmosphericFbo.Valid()) {
        isValid_ = false;
        return;
    }

    InitializePostFxBuffers_();
}

void RendererBackend::InitializePostFxBuffers_() {
    u32 currWidth = frame_->viewportWidth;
    u32 currHeight = frame_->viewportHeight;
    state_.numDownsampleIterations = 0;
    state_.numUpsampleIterations = 0;

    // Initialize bloom
    for (; state_.numDownsampleIterations < 8; ++state_.numDownsampleIterations) {
        currWidth /= 2;
        currHeight /= 2;
        if (currWidth < 8 || currHeight < 8) break;
        PostFXBuffer buffer;
        auto color = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, currWidth, currHeight, 0, false }, NoTextureData);
        color.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        color.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE); // TODO: Does this make sense for bloom textures?
        buffer.fbo = FrameBuffer({ color });
        if (!buffer.fbo.Valid()) {
            isValid_ = false;
            STRATUS_ERROR << "Unable to initialize bloom buffer" << std::endl;
            return;
        }
        state_.postFxBuffers.push_back(buffer);

        // Create the Gaussian Blur buffers
        PostFXBuffer dualBlurFbos[2];
        for (i32 i = 0; i < 2; ++i) {
            FrameBuffer& blurFbo = dualBlurFbos[i].fbo;
            Texture tex = Texture(color.GetConfig(), NoTextureData);
            tex.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
            tex.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
            blurFbo = FrameBuffer({tex});
            state_.gaussianBuffers.push_back(dualBlurFbos[i]);
        }
    }

    std::vector<std::pair<u32, u32>> sizes;
    for (i32 i = state_.numDownsampleIterations - 2; i >= 0; --i) {
        auto tex = state_.postFxBuffers[i].fbo.GetColorAttachments()[0];
        sizes.push_back(std::make_pair(tex.Width(), tex.Height()));
    }
    sizes.push_back(std::make_pair(frame_->viewportWidth, frame_->viewportHeight));
    
    for (auto&[width, height] : sizes) {
        PostFXBuffer buffer;
        ++state_.numUpsampleIterations;
        auto color = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, width, height, 0, false }, NoTextureData);
        color.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        color.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE); // TODO: Does this make sense for bloom textures?
        buffer.fbo = FrameBuffer({ color });
        if (!buffer.fbo.Valid()) {
            isValid_ = false;
            STRATUS_ERROR << "Unable to initialize bloom buffer" << std::endl;
            return;
        }
        state_.postFxBuffers.push_back(buffer);
    }

    // Create the atmospheric post fx buffer
    Texture atmosphericTexture = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    atmosphericTexture.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    atmosphericTexture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.atmosphericPostFxBuffer.fbo = FrameBuffer({atmosphericTexture});
    if (!state_.atmosphericPostFxBuffer.fbo.Valid()) {
        isValid_ = false;
        STRATUS_ERROR << "Unable to initialize atmospheric post fx buffer" << std::endl;
        return;
    }
    state_.postFxBuffers.push_back(state_.atmosphericPostFxBuffer);

    // Create the Gamma-Tonemap buffer
    Texture gammaTonemap = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
    gammaTonemap.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    gammaTonemap.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.gammaTonemapFbo.fbo = FrameBuffer({ gammaTonemap });
    if (!state_.gammaTonemapFbo.fbo.Valid()) {
        isValid_ = false;
        STRATUS_ERROR << "Unable to initialize gamma tonemap buffer" << std::endl;
        return;
    }

    // Create the FXAA buffers
    Texture fxaa = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    fxaa.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    fxaa.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.fxaaFbo1.fbo = FrameBuffer({fxaa});
    if (!state_.fxaaFbo1.fbo.Valid()) {
        isValid_ = false;
        STRATUS_ERROR << "Unable to initialize fxaa luminance buffer" << std::endl;
        return;
    }
    state_.postFxBuffers.push_back(state_.fxaaFbo1);

    fxaa = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    fxaa.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    fxaa.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.fxaaFbo2.fbo = FrameBuffer({fxaa});
    if (!state_.fxaaFbo2.fbo.Valid()) {
        isValid_ = false;
        STRATUS_ERROR << "Unable to initialize fxaa smoothing buffer" << std::endl;
        return;
    }
    state_.postFxBuffers.push_back(state_.fxaaFbo2);

    // Initialize TAA buffer
    Texture taa = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
    taa.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    taa.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.taaFbo.fbo = FrameBuffer({ taa });
    if (!state_.taaFbo.fbo.Valid()) {
        isValid_ = false;
        STRATUS_ERROR << "Unable to initialize taa buffer" << std::endl;
        return;
    }
    state_.postFxBuffers.push_back(state_.taaFbo);
}

void RendererBackend::ClearFramebufferData_(const bool clearScreen) {
    // Always clear the main screen buffer, but only
    // conditionally clean the custom frame buffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // default
    glDepthMask(GL_TRUE);
    glClearDepthf(1.0f);
    glClearColor(frame_->clearColor.r, frame_->clearColor.g, frame_->clearColor.b, frame_->clearColor.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (clearScreen) {
        const glm::vec4& color = frame_->clearColor;
        state_.currentFrame.fbo.Clear(color);
        state_.ssaoOcclusionBuffer.Clear(color);
        state_.ssaoOcclusionBlurredBuffer.Clear(color);
        state_.atmosphericFbo.Clear(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        state_.lightingFbo.Clear(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        state_.vpls.vplGIFbo.Clear(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        state_.vpls.vplGIDenoisedFbo1.Clear(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        state_.vpls.vplGIDenoisedFbo2.Clear(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        // Depending on when this happens we may not have generated cascadeFbo yet
        // if (frame_->vsmc.fbo.Valid()) {
        //     frame_->vsmc.fbo.Clear(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        // }

        for (auto& gaussian : state_.gaussianBuffers) {
            gaussian.fbo.Clear(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        }

        for (auto& postFx : state_.postFxBuffers) {
            postFx.fbo.Clear(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        }

        state_.atmosphericPostFxBuffer.fbo.Clear(glm::vec4(0.0f));
    }
}

void RendererBackend::InitSSAO_() {
    // Create k values 0 to 15 and randomize them
    std::vector<f32> ks(16);
    std::iota(ks.begin(), ks.end(), 0.0f);
    std::shuffle(ks.begin(), ks.end(), std::default_random_engine{});

    // Create the data for the 4x4 lookup table
    f32 table[16 * 3]; // RGB
    for (usize i = 0; i < ks.size(); ++i) {
        const f32 k = ks[i];
        const Radians r(2.0f * f32(STRATUS_PI) * k / 16.0f);
        table[i * 3    ] = cosine(r).value();
        table[i * 3 + 1] = sine(r).value();
        table[i * 3 + 2] = 0.0f;
    }

    // Create the lookup texture
    state_.ssaoOffsetLookup = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, 4, 4, 0, false}, TextureArrayData{(const void *)table});
    state_.ssaoOffsetLookup.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
    state_.ssaoOffsetLookup.SetCoordinateWrapping(TextureCoordinateWrapping::REPEAT);
}

void RendererBackend::InitAtmosphericShadowing_() {
    auto re = std::default_random_engine{};
    // On the range [0.0, 1.0) --> we technically want [0.0, 1.0] but it's close enough
    std::uniform_real_distribution<f32> real(0.0f, 1.0f);

    // Create the 64x64 noise texture
    const usize size = 64 * 64;
    std::vector<f32> table(size);
    for (usize i = 0; i < size; ++i) {
        table[i] = real(re);
    }

    const void* ptr = (const void *)table.data();
    state_.atmosphericNoiseTexture = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, 64, 64, 0, false}, TextureArrayData{ptr});
    state_.atmosphericNoiseTexture.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
    state_.atmosphericNoiseTexture.SetCoordinateWrapping(TextureCoordinateWrapping::REPEAT);
}

void RendererBackend::ClearRemovedLightData_() {
    i32 lightsCleared = 0;
    for (auto ptr : frame_->lightsToRemove) {
        RemoveLightFromShadowMapCache_(ptr);
        ++lightsCleared;
    }

    if (lightsCleared > 0) STRATUS_LOG << "Cleared " << lightsCleared << " lights this frame" << std::endl;
}

Texture RendererBackend::GetHiZOcclusionBuffer() const {
    return state_.depthPyramid;
}

void RendererBackend::Begin(const std::shared_ptr<RendererFrame>& frame, bool clearScreen) {
    CHECK_IS_APPLICATION_THREAD();

    frame_ = frame;

    // Make sure we set our context as the active one
    GraphicsDriver::MakeContextCurrent();

    // Swap current and previous frame buffers
    auto tmp = state_.currentFrame;
    state_.currentFrame = state_.previousFrame;
    state_.previousFrame = tmp;

    auto tmpFbo = state_.flatPassFboCurrentFrame;
    state_.flatPassFboCurrentFrame = state_.flatPassFboPreviousFrame;
    state_.flatPassFboPreviousFrame = tmpFbo;

    // Clear out instanced data from previous frame
    //_ClearInstancedData();

    // Clear out light data for lights that were removed
    ClearRemovedLightData_();

    // Checks to see if any framebuffers need to be generated or re-generated
    RecalculateCascadeData_();

    // Update all dimension, texture and framebuffer data if the viewport changed
    UpdateWindowDimensions_();

    // Includes screen data
    ClearFramebufferData_(clearScreen);

    // Generate the GPU data for all instanced entities
    //_InitAllInstancedData();

    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glDisable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POLYGON_SMOOTH);

    // This is important! It prevents z-fighting if you do multiple passes.
    //glDepthFunc(GL_LEQUAL);
    glDepthRangef(0.0f, 1.0f);
}

/**
 * During the lighting phase, we need each of the 6 faces of the shadow map to have its own view transform matrix.
 * This enables us to convert vertices to be in various different light coordinate spaces.
 */
//static std::vector<glm::mat4> GenerateLightViewTransforms(const glm::mat4 & projection, const glm::vec3 & lightPos) {
//    return std::vector<glm::mat4>{
//        //                       pos       pos + dir                                  up
//        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
//        projection * glm::lookAt(lightPos, lightPos + glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
//        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
//        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
//        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
//        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
//    };
//}
static std::vector<glm::mat4, StackBasedPoolAllocator<glm::mat4>> GenerateLightViewTransforms(const glm::vec3 & lightPos, const UnsafePtr<StackAllocator>& allocator) {
    return std::vector<glm::mat4, StackBasedPoolAllocator<glm::mat4>>({
        //          pos       pos + dir                                  up
        glm::lookAt(lightPos, lightPos + glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
        },

        StackBasedPoolAllocator<glm::mat4>(allocator)
    );
}

void RendererBackend::BindShader_(Pipeline * s) {
    UnbindShader_();
    s->Bind();
    state_.currentShader = s;
}

void RendererBackend::UnbindShader_() {
    if (!state_.currentShader) return;
    //_unbindAllTextures();
    state_.currentShader->Unbind();
    state_.currentShader = nullptr;
}

static void SetCullState(const RenderFaceCulling & mode) {
    // Set the culling state
    switch (mode) {
    case RenderFaceCulling::CULLING_NONE:
        glDisable(GL_CULL_FACE);
        break;
    case RenderFaceCulling::CULLING_CW:
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CW);
        break;
    default:
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CCW);
        break;
    }
}

static bool ValidateTexture(const Texture& tex, const TextureLoadingStatus& status) {
    return status == TextureLoadingStatus::LOADING_DONE;
}

void RendererBackend::RenderBoundingBoxes_(GpuCommandBufferPtr& buffer) {
    if (buffer->NumDrawCommands() == 0) return;

    SetCullState(RenderFaceCulling::CULLING_NONE);

    buffer->BindModelTransformBuffer(CURR_FRAME_MODEL_MATRICES_BINDING_POINT);
    buffer->BindAabbBuffer(AABB_BINDING_POINT);

    BindShader_(state_.aabbDraw.get());

    // _state.aabbDraw->setMat4("projection", _frame->projection);
    // _state.aabbDraw->setMat4("view", _frame->camera->getViewTransform());
    state_.aabbDraw->SetMat4("projectionView", frame_->projectionView);

    for (i32 i = 0; i < buffer->NumDrawCommands(); ++i) {
        state_.aabbDraw->SetInt("modelIndex", i);
        glDrawArrays(GL_LINES, 0, 24);
    }

    UnbindShader_();
}

void RendererBackend::RenderBoundingBoxes_(std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>& map) {
    for (auto& entry : map) {
        RenderBoundingBoxes_(entry.second);
    }
}

void RendererBackend::RenderImmediate_(const RenderFaceCulling cull, GpuCommandBufferPtr& buffer, const CommandBufferSelectionFunction& select, usize offset) {
    if (buffer->NumDrawCommands() == 0) return;

    frame_->materialInfo->GetMaterialBuffer().BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, MATERIAL_BINDING_POINT);
    buffer->BindMaterialIndicesBuffer(MATERIAL_INDICES_BINDING_POINT);
    buffer->BindModelTransformBuffer(CURR_FRAME_MODEL_MATRICES_BINDING_POINT);
    buffer->BindPrevFrameModelTransformBuffer(PREV_FRAME_MODEL_MATRICES_BINDING_POINT);
    select(buffer).Bind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);

    SetCullState(cull);

    const void * offsetBytes = (const void *)(sizeof(GpuDrawElementsIndirectCommand) * offset * buffer->CommandCapacity());
    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, offsetBytes, (GLsizei)buffer->NumDrawCommands(), (GLsizei)0);

    select(buffer).Unbind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);
}

void RendererBackend::Render_(
    Pipeline& s, const RenderFaceCulling cull, GpuCommandBufferPtr& buffer, const CommandBufferSelectionFunction& select, usize offset, 
    bool isLightInteracting, bool removeViewTranslation) {

    if (buffer->NumDrawCommands() == 0) return;

    glEnable(GL_DEPTH_TEST);
    //glDepthMask(GL_TRUE);

    const Camera& camera = *frame_->camera;
    const glm::mat4 & projection = frame_->projection;
    //const glm::mat4 & view = c.getViewTransform();
    glm::mat4 view = frame_->view;
    glm::mat4 projectionView = frame_->projectionView;
    // if (removeViewTranslation) {
    //     // Remove the translation component of the view matrix
    //     view = glm::mat4(glm::mat3(_frame->view));
    //     projectionView = _frame->projection * view;
    // }
    // else {
    //     view = _frame->view;
    //     projectionView = _frame->projectionView;
    // }

    //// Set up the shader we will use for this batch of entities
    //Pipeline * s;
    //if (isLightInteracting == false) {
    //    s = state_.forward.get();
    //}
    //else {
    //    s = state_.geometry.get();
    //}

    //s->print();

    if (isLightInteracting) {
        s.SetVec3("viewPosition", &camera.GetPosition()[0]);
    }

    s.SetFloat("emissiveTextureMultiplier", frame_->settings.GetEmissiveTextureMultiplier());

    s.SetMat4("projection", projection);
    s.SetMat4("view", view);
    s.SetMat4("projectionView", projectionView);
    s.SetMat4("prevProjectionView", frame_->prevProjectionView);

    s.SetInt("viewWidth", frame_->viewportWidth);
    s.SetInt("viewHeight", frame_->viewportHeight);
    s.SetFloat("alphaDepthTestThreshold", frame_->settings.GetAlphaDepthTestThreshold());
    //s->setMat4("projectionView", &projection[0][0]);
    //s->setMat4("view", &view[0][0]);

    RenderImmediate_(cull, buffer, select, offset);

    glDisable(GL_DEPTH_TEST);
    //glDepthMask(GL_FALSE);
}

void RendererBackend::Render_(
    Pipeline& s, std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>& map, const CommandBufferSelectionFunction& select, usize offset,
    bool isLightInteracting, bool removeViewTranslation) {

    for (auto& entry : map) {
        Render_(s, entry.first, entry.second, select, offset, isLightInteracting, removeViewTranslation);
    }
}

void RendererBackend::RenderImmediate_(
    std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>& map, const CommandBufferSelectionFunction& select, usize offset, const bool reverseCullFace) {

    for (auto& entry : map) {
        auto cull = entry.first;
        if (reverseCullFace) {
            if (cull == RenderFaceCulling::CULLING_CCW) {
                cull = RenderFaceCulling::CULLING_CW;
            }
            else if (cull == RenderFaceCulling::CULLING_CW) {
                cull = RenderFaceCulling::CULLING_CCW;
            }
        }
        RenderImmediate_(cull, entry.second, select, offset);
    }
}

void RendererBackend::RenderSkybox_(Pipeline * s, const glm::mat4& projectionView) { 
    glDepthMask(GL_FALSE);

    TextureLoadingStatus status;
    Texture sky = INSTANCE(ResourceManager)->LookupTexture(frame_->settings.skybox, status);
    if (ValidateTexture(sky, status)) {
        //const glm::mat4& projection = frame_->projection;
        //const glm::mat4 view = glm::mat4(glm::mat3(frame_->camera->GetViewTransform()));
        //const glm::mat4 projectionView = projection * view;

        // _state.skybox->setMat4("projection", projection);
        // _state.skybox->setMat4("view", view);
        s->SetMat4("projectionView", projectionView);

        s->SetVec3("colorMask", frame_->settings.GetSkyboxColorMask());
        s->SetFloat("intensity", frame_->settings.GetSkyboxIntensity());
        s->BindTexture("skybox", sky);

        GetMesh(state_.skyboxCube, 0)->GetMeshlet(0)->Render(1, GpuArrayBuffer());
        //_state.skyboxCube->GetMeshContainer(0)->mesh->Render(1, GpuArrayBuffer());
    }

    glDepthMask(GL_TRUE);
}

void RendererBackend::RenderSkybox_() {
    const glm::mat4& projection = frame_->projection;
    const glm::mat4 view = glm::mat4(glm::mat3(frame_->camera->GetViewTransform()));
    const glm::mat4 projectionView = projection * view;

    glDepthFunc(GL_LEQUAL);

    BindShader_(state_.skybox.get());
    RenderSkybox_(state_.skybox.get(), projectionView);
    UnbindShader_();
}

void RendererBackend::PerformVSMCulling(
    Pipeline& pipeline,
    const std::function<GpuCommandBufferPtr(const RenderFaceCulling&)>& selectInput,
    const std::function<GpuCommandReceiveBufferPtr(const RenderFaceCulling&)>& selectOutput,
    std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>& commands
) {
    for (auto& [cull, buffer] : commands) {
        if (buffer->NumDrawCommands() == 0) continue;

        //STRATUS_LOG << buffer->NumDrawCommands() << " " << buffer->CommandCapacity() << std::endl;

        pipeline.SetUint("numDrawCalls", (u32)buffer->NumDrawCommands());
        pipeline.SetUint("maxDrawCommands", (u32)buffer->CommandCapacity());

        buffer->BindModelTransformBuffer(CURR_FRAME_MODEL_MATRICES_BINDING_POINT);
        buffer->BindAabbBuffer(AABB_BINDING_POINT);

        selectInput(cull)->GetIndirectDrawCommandsBuffer(0).BindBase(
            GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VISCULL_VSM_IN_DRAW_CALLS_BINDING_POINT);

        auto receivePtr = selectOutput(cull);
        receivePtr->GetCommandBuffer().BindBase(
            GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VISCULL_VSM_OUT_DRAW_CALLS_BINDING_POINT);

        // i32 value = 0;
        // frame_->vsmc.numDrawCalls.CopyDataToBuffer(0, sizeof(i32), (const void *)&value);

        //const auto numPagesAvailable = frame_->vsmc.cascadeResolutionXY / Texture::VirtualPageSizeXY();
        const auto numPageGroupsX = frame_->vsmc.numPageGroupsX;
        const auto numPageGroupsY = frame_->vsmc.numPageGroupsY;

        //pipeline.DispatchCompute(numPageGroupsX, numPageGroupsY, 1);
        pipeline.DispatchCompute(1, 1, 1);
        pipeline.SynchronizeMemory();

        // const i32 * result = (const i32 *)frame_->vsmc.numDrawCalls.MapMemory(GPU_MAP_READ);
        // STRATUS_LOG << "Before, After: " << buffer->NumDrawCommands() << ", " << *result << std::endl;
        // frame_->vsmc.numDrawCalls.UnmapMemory();
    }
}

void RendererBackend::ProcessShadowMemoryRequests_() {
    // If not using hardware virtual texturing it means the backing memory is already
    // allocated and the GPU can handle everything
    if (frame_->vsmc.vsm.GetConfig().virtualTexture == true) {
        HostFenceSync(frame_->vsmc.prevFrameFence);

        const i32 numPagesToUpdate = *(const i32 *)frame_->vsmc.numPagesToCommit.MapMemory(GPU_MAP_READ);
        frame_->vsmc.numPagesToCommit.UnmapMemory();

        if (numPagesToUpdate > 0) {
            //STRATUS_LOG << "Processing: " << numPagesToUpdate << std::endl;

            const i32* pageIndices = (const i32 *)frame_->vsmc.pagesToCommitList.MapMemory(GPU_MAP_READ, 0, 2 * numPagesToUpdate * sizeof(int));

            for (i32 i = 0; i < numPagesToUpdate; ++i) {
                //STRATUS_LOG << i << std::endl;

                const i32 cascade = pageIndices[3 * i];
                const i32 x = pageIndices[3 * i + 1];
                const i32 y = pageIndices[3 * i + 2];

                //STRATUS_LOG << x << " " << y << std::endl;

                //if (x > 0 && y > 0) {
                {
                    frame_->vsmc.vsm.CommitOrUncommitVirtualPage(
                        std::abs(x) - 1,
                        std::abs(y) - 1,
                        cascade,
                        1,
                        1,
                        (x < 0 || y < 0) ? false : true
                    );
                }
            }

            frame_->vsmc.pagesToCommitList.UnmapMemory();
        }
    }
}

void RendererBackend::ProcessShadowVirtualTexture_() {
    // int value = 0;
    // frame_->vsmc.numPagesToCommit.CopyDataToBuffer(0, sizeof(int), (const void *)&value);

    frame_->vsmc.pageBoundingBox.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_PAGE_BOUNDING_BOX_BINDING_POINT);
    frame_->vsmc.pageGroupsToRender.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_PAGE_GROUPS_TO_RENDER_BINDING_POINT);
    frame_->vsmc.numPagesFree.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_NUM_PAGES_FREE_BINDING_POINT);
    frame_->vsmc.pagesFreeList.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_PAGES_FREE_LIST_BINDING_POINT);
    
    // Clear last frame's requests
    ProcessShadowMemoryRequests_();

    u32 frameCount = u32(INSTANCE(Engine)->FrameCount());
    const auto numPagesAvailable = frame_->vsmc.cascadeResolutionXY / Texture::VirtualPageSizeXY();

    //int clearValue = 0;
    //frame_->vsmc.currFramePageResidencyTable.Clear(0, (const void *)&clearValue);

    //frame_->vsmc.pagesToRender.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 2);

    state_.vsmAnalyzeDepth->Bind();
    
    // state_.vsmAnalyzeDepth->SetMat4("cascadeProjectionView", frame_->vsmc.projectionViewSample);
    state_.vsmAnalyzeDepth->SetMat4("invProjectionView", frame_->invProjectionView);
    state_.vsmAnalyzeDepth->SetMat4("vsmClipMap0ProjectionView", frame_->vsmc.cascades[0].projectionViewRender);
    state_.vsmAnalyzeDepth->SetUint("vsmNumCascades", (u32)frame_->vsmc.cascades.size());
    state_.vsmAnalyzeDepth->SetUint("vsmNumMemoryPools", (u32)frame_->vsmc.vsm.Depth());

    //state_.vsmAnalyzeDepth->BindTextureAsImage("prevFramePageResidencyTable", frame_->vsmc.prevFramePageResidencyTable, true, 0, ImageTextureAccessMode::IMAGE_READ_ONLY);
    //state_.vsmAnalyzeDepth->BindTextureAsImage("currFramePageResidencyTable", frame_->vsmc.currFramePageResidencyTable, true, 0, ImageTextureAccessMode::IMAGE_READ_WRITE);
    frame_->vsmc.pageResidencyTable.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING);
    state_.vsmAnalyzeDepth->SetUint("numPagesXY", numPagesAvailable);

    state_.vsmAnalyzeDepth->SetUint("frameCount", frameCount);

    state_.vsmAnalyzeDepth->BindTexture("depthTexture", state_.currentFrame.depth);

    frame_->vsmc.numPagesToCommit.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_NUM_PAGES_TO_UPDATE_BINDING_POINT);
    frame_->vsmc.pagesToCommitList.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_PAGE_INDICES_BINDING_POINT);

    u32 sizeX = frame_->viewportWidth / 16;
    u32 sizeY = frame_->viewportHeight / 9;

    state_.vsmAnalyzeDepth->DispatchCompute(sizeX, sizeY, 1);
    state_.vsmAnalyzeDepth->SynchronizeMemory();

    state_.vsmAnalyzeDepth->Unbind();

    i32 value = 0;
    frame_->vsmc.numPagesToCommit.CopyDataToBuffer(0, sizeof(int), (const void *)&value);

    state_.vsmMarkPages->Bind();

    state_.vsmMarkPages->SetUint("frameCount", frameCount);

    state_.vsmMarkPages->SetUint("numPagesXY", numPagesAvailable);
    state_.vsmMarkPages->SetUint("sunChanged", frame_->vsmc.worldLight->ChangedWithinLastFrame() ? (u32)1 : (u32)0);
    state_.vsmMarkPages->SetMat4("vsmClipMap0ProjectionView", frame_->vsmc.cascades[0].projectionViewRender);
    state_.vsmMarkPages->SetUint("vsmNumCascades", (u32)frame_->vsmc.cascades.size());
    state_.vsmMarkPages->SetUint("vsmNumMemoryPools", (u32)frame_->vsmc.vsm.Depth());

    frame_->vsmc.numPagesToCommit.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_NUM_PAGES_TO_UPDATE_BINDING_POINT);
    frame_->vsmc.pagesToCommitList.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_PAGE_INDICES_BINDING_POINT);

    sizeX = numPagesAvailable / 8;
    sizeY = numPagesAvailable / 8;

    state_.vsmMarkPages->DispatchCompute(sizeX, sizeY, 1);
    state_.vsmMarkPages->SynchronizeMemory();

    state_.vsmMarkPages->Unbind();

    state_.vsmFreePages->Bind();

    state_.vsmFreePages->SetUint("numPagesXY", numPagesAvailable);
    state_.vsmFreePages->SetMat4("vsmClipMap0ProjectionView", frame_->vsmc.cascades[0].projectionViewRender);
    state_.vsmFreePages->SetUint("vsmNumCascades", (u32)frame_->vsmc.cascades.size());
    state_.vsmFreePages->SetUint("vsmNumMemoryPools", (u32)frame_->vsmc.vsm.Depth());

    state_.vsmFreePages->DispatchCompute(1, 1, 1);
    state_.vsmFreePages->SynchronizeMemory();

    frame_->vsmc.prevFrameFence = HostInsertFence();

    state_.vsmFreePages->Unbind();

    TextureAccess depthBindConfig{
        TextureComponentFormat::RED,
        TextureComponentSize::BITS_32,
        TextureComponentType::UINT
    };

    state_.vsmMarkScreen->Bind();

    // state_.vsmMarkScreen->SetMat4("invCascadeProjectionView", frame_->vsmc.cascades[cascade].invProjectionViewRender);
    // state_.vsmMarkScreen->SetMat4("vsmProjectionView", frame_->vsmc.cascades[cascade].projectionViewSample);
    state_.vsmMarkScreen->SetUint("numPagesXY", frame_->vsmc.numPageGroupsX);
    state_.vsmMarkScreen->SetMat4("vsmClipMap0ProjectionView", frame_->vsmc.cascades[0].projectionViewRender);
    state_.vsmMarkScreen->SetUint("vsmNumCascades", (u32)frame_->vsmc.cascades.size());
    state_.vsmMarkScreen->SetUint("vsmNumMemoryPools", (u32)frame_->vsmc.vsm.Depth());
    state_.vsmMarkScreen->SetVec2("ndcClipOriginChange", frame_->vsmc.ndcClipOriginDifference);
    state_.vsmMarkScreen->BindTextureAsImage(
        "vsm", frame_->vsmc.vsm, 0, true, 0, ImageTextureAccessMode::IMAGE_READ_WRITE, depthBindConfig);
    frame_->vsmc.pageResidencyTable.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING);

    state_.vsmMarkScreen->DispatchCompute(frame_->vsmc.numPageGroupsX / 8, frame_->vsmc.numPageGroupsY / 8, 1);
    state_.vsmMarkScreen->SynchronizeMemory();

    state_.vsmCull->Bind();

    // state_.vsmCull->SetMat4("cascadeProjectionView", frame_->vsmc.projectionViewRender);
    // state_.vsmCull->SetMat4("invCascadeProjectionView", frame_->vsmc.invProjectionViewRender);
    // state_.vsmCull->SetMat4("vsmProjectionView", frame_->vsmc.projectionViewSample);
    state_.vsmCull->SetMat4("vsmClipMap0ProjectionView", frame_->vsmc.cascades[0].projectionViewRender);
    state_.vsmCull->SetUint("vsmNumCascades", (u32)frame_->vsmc.cascades.size());
    state_.vsmCull->SetUint("vsmNumMemoryPools", (u32)frame_->vsmc.vsm.Depth());
    state_.vsmCull->SetUint("frameCount", frameCount);
    state_.vsmCull->SetUint("numPageGroupsX", (u32)frame_->vsmc.numPageGroupsX);
    state_.vsmCull->SetUint("numPageGroupsY", (u32)frame_->vsmc.numPageGroupsY);
    //state_.vsmCull->BindTextureAsImage("currFramePageResidencyTable", frame_->vsmc.currFramePageResidencyTable, true, 0, ImageTextureAccessMode::IMAGE_READ_ONLY);
    frame_->vsmc.pageResidencyTable.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING);
    state_.vsmCull->SetUint("numPagesXY", numPagesAvailable);
    state_.vsmCull->SetUint("numPixelsXY", (u32)frame_->vsmc.cascadeResolutionXY);

    PerformVSMCulling(
        *state_.vsmCull,
        [this](const RenderFaceCulling& cull) {
            return frame_->drawCommands->dynamicPbrMeshes.find(cull)->second;
        },
        [this](const RenderFaceCulling& cull) {
            return frame_->vsmc.drawCommandsFinal->dynamicPbrMeshes.find(cull)->second;
        },
        frame_->drawCommands->dynamicPbrMeshes
    );

    PerformVSMCulling(
        *state_.vsmCull,
        [this](const RenderFaceCulling& cull) {
            return frame_->drawCommands->staticPbrMeshes.find(cull)->second;
        },
        [this](const RenderFaceCulling& cull) {
            return frame_->vsmc.drawCommandsFinal->staticPbrMeshes.find(cull)->second;
        },
        frame_->drawCommands->staticPbrMeshes
    );

    state_.vsmCull->Unbind();
}

void RendererBackend::RenderCSMDepth_() {
    //if (frame_->vsmc.cascades.size() > state_.csmDepth.size()) {
    //    throw std::runtime_error("Max cascades exceeded (> 6)");
    //}

    ProcessShadowVirtualTexture_();

    // const auto n = *(const i32*)frame_->vsmc.numPagesFree.MapMemory(GPU_MAP_READ);
    // frame_->vsmc.numPagesFree.UnmapMemory();

    // STRATUS_LOG << n << std::endl;

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);
    // Allows GPU to perform angle-dependent depth offset to help reduce artifacts such as shadow acne
    // See https://blogs.igalia.com/itoral/2017/10/02/working-with-lights-and-shadows-part-iii-rendering-the-shadows/
    // See https://community.khronos.org/t/can-anyone-explain-glpolygonoffset/35382
    //glEnable(GL_POLYGON_OFFSET_FILL);
    // See https://paroj.github.io/gltut/Positioning/Tut05%20Depth%20Clamping.html
    glEnable(GL_DEPTH_CLAMP);
    // First value is conditional on slope
    // Second value is a constant unconditional offset
    //glPolygonOffset(2.0f, 0.0f);
    //glBlendFunc(GL_ONE, GL_ONE);
    // glDisable(GL_CULL_FACE);

    frame_->vsmc.fbo.Bind();
    const Texture * depth = &frame_->vsmc.vsm; //frame_->vsmc.fbo.GetDepthStencilAttachment();
    if (!depth) {
        throw std::runtime_error("Critical error: depth attachment not present");
    }

    //const int * minXY = (const int *)frame_->vsmc.minViewportXY.MapMemory(GPU_MAP_READ);
    //const int * maxXY = (const int *)frame_->vsmc.maxViewportXY.MapMemory(GPU_MAP_READ);

    //STRATUS_LOG << "Min XY: " << minXY[0] << " " << minXY[1] << std::endl;
    //STRATUS_LOG << "Max XY: " << maxXY[0] << " " << maxXY[1] << std::endl;

    glViewport(0, 0, depth->Width(), depth->Height());
    //glViewport(minXY[0], minXY[1], maxXY[0], maxXY[1]);

    //frame_->vsmc.minViewportXY.UnmapMemory();
    //frame_->vsmc.maxViewportXY.UnmapMemory();

    constexpr usize cascade = 0;

    const auto& csm = frame_->vsmc;

    //for (usize cascade = 0; cascade < frame_->vsmc.cascades.size(); ++cascade) {
    // for (usize x = 0; x < 128; ++x) {
    //     for (usize y = 0; y < 128; ++y) {
    //         glViewport(x * 128, y * 128, x * 128 + 128, y * 128 + 128);
    //         RenderImmediate_(frame_->drawCommands->dynamicPbrMeshes, selectDynamic, true);
    //         RenderImmediate_(frame_->drawCommands->staticPbrMeshes, selectStatic, true);
    //     }
    // }
    //glViewport(30 * 128, 30 * 128, 70 * 128, 70 * 128);
    const auto numPageGroups = frame_->vsmc.numPageGroupsX * frame_->vsmc.numPageGroupsY;
    const auto pageGroupWindowWidth = depth->Width() / frame_->vsmc.numPageGroupsX;
    const auto pageGroupWindowHeight = depth->Height() / frame_->vsmc.numPageGroupsY;

    const u32 * pageGroupsToRender = (const u32 *)frame_->vsmc.pageGroupsToRender.MapMemory(GPU_MAP_READ);

    const u32 maxPageGroupsToUpdate = frame_->vsmc.numPageGroupsX / 8;// / 2;// / 8;// / 2;// / 8;

    // STRATUS_LOG << frame_->vsmc.numPageGroupsX << " " << maxPageGroupsToUpdate << std::endl;

    using VirtualIndexSet = std::unordered_set<
        u32,
        std::hash<u32>,
        std::equal_to<u32>,
        StackBasedPoolAllocator<u32>>;

    using VirtualIndexSetArray = std::vector<
        VirtualIndexSet,
        StackBasedPoolAllocator<VirtualIndexSet>>;

    //VirtualIndexSet changeSet(frame_->vsmc.numPageGroupsX * frame_->vsmc.numPageGroupsY, StackBasedPoolAllocator<u32>(frame_->perFrameScratchMemory));
    VirtualIndexSetArray changeSetArray(
        frame_->vsmc.cascades.size(),
        VirtualIndexSet(frame_->vsmc.numPageGroupsX * frame_->vsmc.numPageGroupsY, StackBasedPoolAllocator<u32>(frame_->perFrameScratchMemory)),
        StackBasedPoolAllocator<VirtualIndexSet>(frame_->perFrameScratchMemory)
    );

    const auto pageGroupStride = frame_->vsmc.numPageGroupsX * frame_->vsmc.numPageGroupsY;

    for (u32 cascade = 0; cascade < frame_->vsmc.cascades.size(); ++cascade) {
        for (u32 y = 0; y < frame_->vsmc.numPageGroupsY; ++y) {
            for (u32 x = 0; x < frame_->vsmc.numPageGroupsX; ++x) {
                const usize pageGroupIndex = x + y * frame_->vsmc.numPageGroupsX + cascade * pageGroupStride;
                if (pageGroupsToRender[pageGroupIndex] > 0) {
                    frame_->vsmc.pageGroupUpdateQueue[cascade]->PushBack(x, y);
                    changeSetArray[cascade].insert(ComputeFlatVirtualIndex(x, y, frame_->vsmc.numPageGroupsX));
                }
            }
        }

        // Update the page group update queue with only what is visible on screen
        frame_->vsmc.pageGroupUpdateQueue[cascade]->SetIntersection(changeSetArray[cascade]);

        // Clear the back buffer
        frame_->vsmc.backPageGroupUpdateQueue[cascade]->Clear();
    }

    //STRATUS_LOG << changeSetArray[cascade].size() << std::endl;

    frame_->vsmc.pageGroupsToRender.UnmapMemory();
    pageGroupsToRender = nullptr;

    // Now compute a page group x/y for glViewport
    // while (frame_->vsmc.pageGroupUpdateQueue->Size() > 0) {
    //     const auto xy = frame_->vsmc.pageGroupUpdateQueue->PopFront();
    //     const auto x = xy.first;
    //     const auto y = xy.second;

    //     ++numPageGroupsToRender;

    //     auto newMinPageGroupX = std::min<u32>(minPageGroupX, x);
    //     auto newMinPageGroupY = std::min<u32>(minPageGroupY, y);

    //     auto newMaxPageGroupX = std::max<u32>(maxPageGroupX, x + 1);
    //     auto newMaxPageGroupY = std::max<u32>(maxPageGroupY, y + 1);

    //     bool failedCheckX = (newMaxPageGroupX - newMinPageGroupX) > maxPageGroupsToUpdate;
    //     bool failedCheckY = (newMaxPageGroupY - newMinPageGroupY) > maxPageGroupsToUpdate;

    //     if (failedCheckX || failedCheckY) {
    //         frame_->vsmc.backPageGroupUpdateQueue->PushFront(x, y);

    //         const auto differenceX = (newMaxPageGroupX - newMinPageGroupX) - maxPageGroupsToUpdate;
    //         const auto differenceY = (newMaxPageGroupY - newMinPageGroupY) - maxPageGroupsToUpdate;

    //         if (newMinPageGroupX < )

    //         continue;
    //     }

    //     minPageGroupX = newMinPageGroupX;
    //     minPageGroupY = newMinPageGroupY;

    //     maxPageGroupX = newMaxPageGroupX;
    //     maxPageGroupY = newMaxPageGroupY;
    // }

    //STRATUS_LOG << frame_->vsmc.projectionViewSample * glm::vec4(frame_->camera->GetPosition(), 1.0f) << std::endl;

    // auto ndc1 = VsmCalculateOriginClipValueFromWorldPos(frame_->vsmc.cascades[0].projectionViewRender, frame_->camera->GetPosition(), 0);
    // auto ndc2 = glm::vec3(frame_->vsmc.projectionViewSample * glm::vec4(frame_->camera->GetPosition(), 1.0f));
    // STRATUS_LOG << "1: " << ndc1 << std::endl;
    // STRATUS_LOG << "2: " << ndc2 << std::endl;

    // for (int x = 0; x < frame_->vsmc.numPageGroupsX; ++x) {
    //     const float xy = float(x) + 0.5f;
    //     glm::vec2 virtualCoords = glm::vec2(xy, xy);

    //     glm::vec2 physicalCoords = ConvertVirtualCoordsToPhysicalCoords(
    //         virtualCoords,
    //         glm::vec2(frame_->vsmc.numPageGroupsX - 1),
    //         frame_->vsmc.cascades[0].projectionViewRender
    //     );

    //     // STRATUS_LOG << xy << ": " << physicalCoords << std::endl;

    //     glm::vec2 ndc = glm::vec2(2.0f * virtualCoords) / glm::vec2(frame_->vsmc.numPageGroupsX) - 1.0f;
    //     glm::vec3 worldPos = glm::vec3(frame_->vsmc.cascades[0].invProjectionViewRender * glm::vec4(ndc, 0.0f, 1.0f));
    //     glm::vec2 ndcOrigin = glm::vec2(frame_->vsmc.projectionViewSample * glm::vec4(worldPos, 1.0f));

    //     glm::vec2 physicalTexCoords = ndcOrigin * 0.5f + glm::vec2(0.5f);

    //     //glm::vec2 physicalCoords2 = physicalTexCoords * glm::vec2(frame_->vsmc.numPageGroupsX) - 0.5f;
    //     glm::vec2 physicalCoords2 = WrapIndex(
    //         physicalTexCoords * glm::vec2(frame_->vsmc.numPageGroupsX) - 0.5f,
    //         glm::vec2(frame_->vsmc.numPageGroupsX)
    //     );

    //     STRATUS_LOG << xy << ": " << physicalCoords << ", " << physicalCoords2 << std::endl;
    // }

    // for (i32 x = 0; x < frame_->vsmc.numPageGroupsX; ++x) {
    //     const float xy = float(x) + 0.5f;

    //     glm::vec2 virtualNdc = (2.0f * glm::vec2(xy, xy)) / float(frame_->vsmc.cascadeResolutionXY) - 1.0f;

    //     glm::vec3 worldPos = glm::vec3(frame_->vsmc.cascades[0].invProjectionViewRender * glm::vec4(virtualNdc, 0.0f, 1.0f));
    //     glm::vec2 physicalNdc = glm::vec2(frame_->vsmc.projectionViewSample * glm::vec4(worldPos, 1.0f));

    //     glm::vec2 physicalUvCoords = physicalNdc * 0.5f + glm::vec2(0.5f);

    //     glm::vec2 wrappedUvCoords = Fract(physicalUvCoords);

    //     STRATUS_LOG << xy << ": "
    //         << WrapIndex(physicalUvCoords * glm::vec2(frame_->vsmc.numPageGroupsX), glm::vec2(frame_->vsmc.numPageGroupsX))
    //         << wrappedUvCoords * glm::vec2(frame_->vsmc.numPageGroupsX) << std::endl;
    // }

    //const GpuPageResidencyEntry * pageTable = (const GpuPageResidencyEntry *)frame_->vsmc.pageResidencyTable.MapMemory(GPU_MAP_READ);

    for (usize cascade = 0; cascade < frame_->vsmc.cascades.size(); ++cascade) {

        //const GpuPageResidencyEntry * currTable = pageTable + cascade * frame_->vsmc.numPageGroupsX * frame_->vsmc.numPageGroupsY;

        //std::unordered_set<usize> entries;

        //const auto cascadeStepSize = cascade * frame_->vsmc.numPageGroupsX * frame_->vsmc.numPageGroupsY;
        //usize dx;
        //usize dy;
        //usize dz;
        //bool foundDuplicate = false;
        //usize duplicates = 0;

        //for (usize x = 0; x < frame_->vsmc.numPageGroupsX; ++x) {
        //    for (usize y = 0; y < frame_->vsmc.numPageGroupsY; ++y) {
        //        const auto data = currTable[x + y * frame_->vsmc.numPageGroupsX + cascadeStepSize].frameMarker;
        //        const auto px = (data & 0x0FF00000) >> 20;
        //        const auto py = (data & 0x000FF000) >> 12;
        //        const auto mem = (data & 0x00000FF0) >> 4;
        //        const auto res = (data & 0x0000000F);

        //        if (res == 0) continue;

        //        const auto index = px + py * frame_->vsmc.numPageGroupsX + mem * frame_->vsmc.numPageGroupsX * frame_->vsmc.numPageGroupsY;
        //        if (entries.find(index) != entries.end()) {
        //            foundDuplicate = true;
        //            dx = px;
        //            dy = py;
        //            dz = mem;
        //            ++duplicates;
        //        }

        //        entries.insert(index);
        //    }
        //}
        //
        //STRATUS_LOG << entries.size() << std::endl;

        //if (foundDuplicate) {
        //    STRATUS_LOG << "Found duplicate: " << duplicates << ", " << dx << ", " << dy << ", " << dz << std::endl;
        //}

        u32 minPageGroupX = frame_->vsmc.numPageGroupsX + 1;
        u32 minPageGroupY = frame_->vsmc.numPageGroupsY + 1;
        u32 maxPageGroupX = 0;
        u32 maxPageGroupY = 0;
        usize numPageGroupsToRender = 0;

        while (frame_->vsmc.pageGroupUpdateQueue[cascade]->Size() > 0) {
            const auto xy = frame_->vsmc.pageGroupUpdateQueue[cascade]->PopFront();
            const auto x = xy.first.first;
            const auto y = xy.first.second;
            const auto count = xy.second;

            auto newMinPageGroupX = std::min<u32>(minPageGroupX, x);
            auto newMinPageGroupY = std::min<u32>(minPageGroupY, y);

            auto newMaxPageGroupX = std::max<u32>(maxPageGroupX, x + 1);
            auto newMaxPageGroupY = std::max<u32>(maxPageGroupY, y + 1);

            const bool failedCheckX = (newMaxPageGroupX - newMinPageGroupX) > maxPageGroupsToUpdate;
            const bool failedCheckY = (newMaxPageGroupY - newMinPageGroupY) > maxPageGroupsToUpdate;

            // We want to always fully render the final cascade since it will be minimum mip level
            if ((failedCheckX || failedCheckY) && cascade != (frame_->vsmc.cascades.size() - 1)) {
                frame_->vsmc.backPageGroupUpdateQueue[cascade]->PushBack(x, y, count);
                continue;
            }

            ++numPageGroupsToRender;

            minPageGroupX = newMinPageGroupX;
            minPageGroupY = newMinPageGroupY;

            maxPageGroupX = newMaxPageGroupX;
            maxPageGroupY = newMaxPageGroupY;

            //if (count < 3) {
            //    frame_->vsmc.backPageGroupUpdateQueue[cascade]->PushBack(x, y, count + 1);
            //}
        }

        // STRATUS_LOG << "Awaiting processing: " << frame_->vsmc.backPageGroupUpdateQueue[0]->Size() << std::endl;
        // STRATUS_LOG << minPageGroupX << " " << minPageGroupY << " " << maxPageGroupX << " " << maxPageGroupY << std::endl;

        // Swap front and back buffers
        auto tmp = frame_->vsmc.pageGroupUpdateQueue[cascade];
        frame_->vsmc.pageGroupUpdateQueue[cascade] = frame_->vsmc.backPageGroupUpdateQueue[cascade];
        frame_->vsmc.backPageGroupUpdateQueue[cascade] = tmp;

        //STRATUS_LOG << "PAGE GROUPS TO RENDER: " << numPageGroupsToRender << std::endl;

        // auto test = frame_->vsmc.ndcClipOriginDifference * 32.0f;
        // if (test.x > 0 || test.y > 0) {
        //     STRATUS_LOG << test << std::endl;
        // }

        if (numPageGroupsToRender > 0) {
            // minPageGroupX = 0;
            // minPageGroupY = 0;
            // maxPageGroupX = frame_->vsmc.numPageGroupsX;
            // maxPageGroupY = frame_->vsmc.numPageGroupsY;

            // Add a 2 page group border around the whole update region
            if (minPageGroupX > 0) {
                --minPageGroupX;
            }
            if (minPageGroupY > 0) {
                --minPageGroupY;
            }
            // if (minPageGroupX > 0) {
            //     --minPageGroupX;
            // }
            // if (minPageGroupY > 0) {
            //     --minPageGroupY;
            // }

            if (maxPageGroupX < frame_->vsmc.numPageGroupsX) {
                ++maxPageGroupX;
            }
            if (maxPageGroupY < frame_->vsmc.numPageGroupsY) {
                ++maxPageGroupY;
            }
            // if (maxPageGroupX < frame_->vsmc.numPageGroupsX) {
            //     ++maxPageGroupX;
            // }
            // if (maxPageGroupY < frame_->vsmc.numPageGroupsY) {
            //     ++maxPageGroupY;
            // }

            // if (minPageGroupX % 2 != 0 && minPageGroupX > 0) {
            //     --minPageGroupX;
            // }

            // if (minPageGroupY % 2 != 0 && minPageGroupY > 0) {
            //     --minPageGroupY;
            // }

            // if (maxPageGroupX % 2 != 0 && maxPageGroupX < frame_->vsmc.numPageGroupsX) {
            //     ++maxPageGroupX;
            // }

            // if (maxPageGroupY % 2 != 0 && maxPageGroupY < frame_->vsmc.numPageGroupsY) {
            //     ++maxPageGroupY;
            // }

            u32 sizeX = maxPageGroupX - minPageGroupX;
            u32 sizeY = maxPageGroupY - minPageGroupY;
            const u32 frameCount = (u32)INSTANCE(Engine)->FrameCount();

            // Constrain to be a perfect square
            // for (int i = 0; std::fmod(f32(frame_->vsmc.numPageGroupsX) / sizeX, 2) != 0; ++i) {
            //     if (i % 2 == 0 && minPageGroupX > 0) {
            //         --minPageGroupX;
            //     }
            //     else if (maxPageGroupX < frame_->vsmc.numPageGroupsX) {
            //         ++maxPageGroupX;
            //     }

            //     sizeX = maxPageGroupX - minPageGroupX;
            // }

            // for (int i = 0; std::fmod(f32(frame_->vsmc.numPageGroupsY) / sizeY, 2) != 0; ++i) {
            //     if (i % 2 == 0 && minPageGroupY > 0) {
            //         --minPageGroupY;
            //     }
            //     else if (maxPageGroupY < frame_->vsmc.numPageGroupsY) {
            //         ++maxPageGroupY;
            //     }

            //     sizeY = maxPageGroupY - minPageGroupY;
            // }

            // if (sizeX < maxPageGroupsToUpdate) {
            //     auto difference = maxPageGroupsToUpdate - sizeX;
            //     maxPageGroupX = (maxPageGroupX + difference) < frame_->vsmc.numPageGroupsX ? maxPageGroupX + difference : frame_->vsmc.numPageGroupsX;

            //     sizeX = maxPageGroupX - minPageGroupX;
            //     if (sizeX < maxPageGroupsToUpdate) {
            //         difference = maxPageGroupsToUpdate - sizeX;
            //         minPageGroupX -= difference;

            //         sizeX = maxPageGroupX - minPageGroupX;
            //     }
            // }

            // if (sizeY < maxPageGroupsToUpdate) {
            //     auto difference = maxPageGroupsToUpdate - sizeY;
            //     maxPageGroupY = (maxPageGroupY + difference) < frame_->vsmc.numPageGroupsY ? maxPageGroupY + difference : frame_->vsmc.numPageGroupsY;

            //     sizeY = maxPageGroupY - minPageGroupY;
            //     if (sizeY < maxPageGroupsToUpdate) {
            //         difference = maxPageGroupsToUpdate - sizeY;
            //         minPageGroupY -= difference;

            //         sizeY = maxPageGroupY - minPageGroupY;
            //     }
            // }

            // Constrain the update window to be divisble by 2
            if (sizeX % 2 != 0) {
                if (maxPageGroupX < frame_->vsmc.numPageGroupsX) {
                    ++maxPageGroupX;
                }
                else if (minPageGroupX > 0) {
                    --minPageGroupX;
                }

                sizeX = maxPageGroupX - minPageGroupX;
            }

            if (sizeY % 2 != 0) {
                if (maxPageGroupY < frame_->vsmc.numPageGroupsY) {
                    ++maxPageGroupY;
                }
                else if (minPageGroupY > 0) {
                    --minPageGroupY;
                }

                sizeY = maxPageGroupY - minPageGroupY;
            }

            //STRATUS_LOG << minPageGroupX << " " << minPageGroupY << " " << sizeX << " " << sizeY << std::endl;

            //STRATUS_LOG << cascade << ": " << minPageGroupX << " " << minPageGroupY << ", " << maxPageGroupX << " " << maxPageGroupY << " " << sizeX << " " << sizeY << std::endl;

            // Repartition pages into new set of groups
            // See https://stackoverflow.com/questions/28155749/opengl-matrix-setup-for-tiled-rendering
            //const u32 newNumPageGroupsX = frame_->vsmc.numPageGroupsX / sizeX;
            //const u32 newNumPageGroupsY = frame_->vsmc.numPageGroupsY / sizeY;
            const f32 newNumPageGroupsX = f32(frame_->vsmc.numPageGroupsX) / f32(sizeX);
            const f32 newNumPageGroupsY = f32(frame_->vsmc.numPageGroupsY) / f32(sizeY);

            // Normalize the min/max page groups
            // Converts first to [-1, 1] and then t [0, 1]
            const f32 normMinPageGroupX = (f32(2 * minPageGroupX) / f32(frame_->vsmc.numPageGroupsX) - 1.0f) * 0.5f + 0.5f;
            const f32 normMinPageGroupY = (f32(2 * minPageGroupY) / f32(frame_->vsmc.numPageGroupsY) - 1.0f) * 0.5f + 0.5f;
            // const f32 normMaxPageGroupX = f32(maxPageGroupX) / f32(frame_->vsmc.numPageGroupsX);
            // const f32 normMaxPageGroupY = f32(maxPageGroupY) / f32(frame_->vsmc.numPageGroupsY);

            // Translate the normalized min/max page groups into fractional locations within
            // the new repartitioned page groups
            const f32 newMinPageGroupX = normMinPageGroupX * newNumPageGroupsX;
            const f32 newMinPageGroupY = normMinPageGroupY * newNumPageGroupsY;
            // const f32 newMaxPageGroupX = normMaxPageGroupX * newNumPageGroupsX;
            // const f32 newMaxPageGroupY = normMaxPageGroupY * newNumPageGroupsY;

            const glm::ivec2 maxVirtualIndex(frame_->vsmc.cascadeResolutionXY - 1);
            //const glm::mat4 invProjectionView = frame_->vsmc.cascades[0].invProjectionViewRender;
            //const glm::mat4 vsmProjectionView = frame_->vsmc.projectionViewSample;

            // for (int i = 0; i < 128; ++i) {
            // STRATUS_LOG << ConvertVirtualCoordsToPhysicalCoords(glm::ivec2(i, i), maxVirtualIndex, invProjectionView, vsmProjectionView) << std::endl;
            // }

            //const auto scaleX = f32(frame_->vsmc.numPageGroupsX - sizeX + 1.0f);
            //const auto scaleY = f32(frame_->vsmc.numPageGroupsY - sizeY + 1.0f);

            // Perform scaling equal to the new number of page groups (prevents entire
            // scene from being written into subset of texture - only relevant part
            // of the scene should go into the texture subset)
            const f32 scaleX = newNumPageGroupsX;
            const f32 scaleY = newNumPageGroupsY;

            const f32 invX = 1.0f / newNumPageGroupsX;
            const f32 invY = 1.0f / newNumPageGroupsY;
            // const f32 invX = 1.0f / f32(frame_->vsmc.numPageGroupsX);
            // const f32 invY = 1.0f / f32(frame_->vsmc.numPageGroupsY);

            //STRATUS_LOG << newNumPageGroupsX << " " << newNumPageGroupsY << std::endl;

            // Perform translation to orient our vertex outputs to only the ones relevant
            // to the subset of the texture we are interested in
            const f32 tx = - (-1.0f + invX + 2.0f * invX * f32(newMinPageGroupX));
            const f32 ty = - (-1.0f + invY + 2.0f * invY * f32(newMinPageGroupY));
            // const f32 tx = - (-1.0f + invX + 2.0f * invX * f32(minPageGroupX));
            // const f32 ty = - (-1.0f + invY + 2.0f * invY * f32(minPageGroupY));

            glm::mat4 scale(1.0f);
            matScale(scale, glm::vec3(scaleX, scaleY, 1.0f));

            glm::mat4 translate(1.0f);
            matTranslate(translate, glm::vec3(tx, ty, 0.0f));
            
            const glm::mat4 cascadeOrthoProjection = csm.cascades[cascade].projectionViewRender;

            //STRATUS_LOG << scaleX << " " << scaleY << " " << tx << " " << ty << std::endl;

            // STRATUS_LOG << "1: " << (
            //     scale * translate * frame_->vsmc.cascades[cascade].projection * frame_->vsmc.viewTransform) << std::endl;

            //STRATUS_LOG << "1: " << cascadeOrthoProjection << std::endl;
            //glm::mat4 cascadeOrthoProjectionModified = csm.viewTransform;
            // cascadeOrthoProjectionModified[3] = glm::vec4(
            //     -glm::vec3(frame_->camera->GetPosition().x, 0.0f, frame_->camera->GetPosition().z),
            //     1.0f
            // );
            // cascadeOrthoProjectionModified[3] = glm::vec4(
            //     -glm::vec3(441.529f, 19.608f, 221.228f),
            //     1.0f
            // );
            //STRATUS_LOG << cascadeOrthoProjectionModified[3] << std::endl;
            //cascadeOrthoProjectionModified = csm.projection * cascadeOrthoProjectionModified;
            const glm::mat4 projectionView = scale * translate * cascadeOrthoProjection;
            // STRATUS_LOG << "2: " << projectionView << std::endl;
            const i32 numPagesXY = i32(frame_->vsmc.cascadeResolutionXY / Texture::VirtualPageSizeXY());

            u32 startX = minPageGroupX;// * pageGroupWindowWidth;
            u32 startY = minPageGroupY;// * pageGroupWindowHeight;

            u32 endX = maxPageGroupX;// * pageGroupWindowWidth;
            u32 endY = maxPageGroupY;// * pageGroupWindowHeight;

            u32 numComputeGroupsX = frame_->vsmc.numPageGroupsX;
            u32 numComputeGroupsY = frame_->vsmc.numPageGroupsY;

            TextureAccess depthBindConfig{
                TextureComponentFormat::RED,
                TextureComponentSize::BITS_32,
                TextureComponentType::UINT
            };

            state_.vsmClear->Bind();

            // state_.vsmClear->SetMat4("invCascadeProjectionView", frame_->vsmc.cascades[cascade].invProjectionViewRender);
            // state_.vsmClear->SetMat4("vsmProjectionView", frame_->vsmc.cascades[cascade].projectionViewSample);
            state_.vsmClear->SetIVec2("startXY", glm::ivec2(startX, startY));
            state_.vsmClear->SetIVec2("endXY", glm::ivec2(endX, endY));
            state_.vsmClear->SetUint("numPagesXY", numPagesXY);
            state_.vsmClear->SetMat4("vsmClipMap0ProjectionView", frame_->vsmc.cascades[0].projectionViewRender);
            state_.vsmClear->SetUint("vsmNumCascades", (u32)frame_->vsmc.cascades.size());
            state_.vsmClear->SetUint("vsmNumMemoryPools", (u32)frame_->vsmc.vsm.Depth());
            state_.vsmClear->SetInt("vsmClipMapIndex", cascade);
            state_.vsmClear->BindTextureAsImage(
                "vsm", *depth, 0, true, 0, ImageTextureAccessMode::IMAGE_READ_WRITE, depthBindConfig);
            frame_->vsmc.pageResidencyTable.BindBase(
                GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING);

            state_.vsmClear->DispatchCompute(numComputeGroupsX, numComputeGroupsY, 1);
            state_.vsmClear->SynchronizeMemory();

            Pipeline * shader = frame_->vsmc.worldLight->GetAlphaTest() && cascade < 2 ?
                state_.csmDepthRunAlphaTest[cascade].get() :
                state_.csmDepth[cascade].get();

            BindShader_(shader);

            //shader->BindTextureAsImage("vsm", *depth, false, cascade, ImageTextureAccessMode::IMAGE_WRITE_ONLY);
            //shader->SetUint("vsmSizeX", (u32)depth->Width());
            //shader->SetUint("vsmSizeY", (u32)depth->Height());
            shader->SetVec3("lightDir", &frame_->vsmc.worldLightCamera->GetDirection()[0]);
            shader->SetFloat("nearClipPlane", frame_->znear);
            shader->SetFloat("alphaDepthTestThreshold", frame_->settings.GetAlphaDepthTestThreshold());
            shader->SetUint("frameCount", frameCount);
            shader->SetIVec2("startXY", glm::ivec2(startX, startY));
            shader->SetIVec2("endXY", glm::ivec2(endX, endY));

            // Set up each individual view-projection matrix
            // for (i32 i = 0; i < _frame->csc.cascades.size(); ++i) {
            //     auto& csm = _frame->csc.cascades[i];
            //     _state.csmDepth->setMat4("shadowMatrices[" + std::to_string(i) + "]", &csm.projectionViewRender[0][0]);
            // }

            // Select face (one per frame)
            //const i32 face = Engine::Instance()->FrameCount() % 4;
            //_state.csmDepth->setInt("face", face);

            // Render everything
            // See https://www.gamedev.net/forums/topic/695063-is-there-a-quick-way-to-fix-peter-panning-shadows-detaching-from-objects/5370603/
            // for the tip about enabling reverse culling for directional shadow maps to reduce peter panning
            auto& csm = frame_->vsmc;
            shader->SetMat4("shadowMatrix", csm.cascades[cascade].projectionViewRender);
            shader->SetUint("numPagesXY", (u32)(frame_->vsmc.cascadeResolutionXY / Texture::VirtualPageSizeXY()));
            shader->SetUint("virtualShadowMapSizeXY", (u32)depth->Width());
            shader->SetMat4("vsmClipMap0ProjectionView", frame_->vsmc.cascades[0].projectionViewRender);
            shader->SetUint("vsmNumCascades", (u32)frame_->vsmc.cascades.size());
            shader->SetUint("vsmNumMemoryPools", (u32)frame_->vsmc.vsm.Depth());
            shader->BindTextureAsImage("vsm", *depth, 0, true, 0, ImageTextureAccessMode::IMAGE_READ_WRITE, depthBindConfig);
            shader->SetInt("vsmClipMapIndex", cascade);

            CommandBufferSelectionFunction selectDynamic = [this, cascade](GpuCommandBufferPtr& b) {
                const auto cull = b->GetFaceCulling();
                return frame_->vsmc.drawCommandsFinal->dynamicPbrMeshes.find(cull)->second->GetCommandBuffer();
            };

            CommandBufferSelectionFunction selectStatic = [this, cascade](GpuCommandBufferPtr& b) {
                const auto cull = b->GetFaceCulling();
                return frame_->vsmc.drawCommandsFinal->staticPbrMeshes.find(cull)->second->GetCommandBuffer();
            };

            shader->SetMat4("shadowMatrix", projectionView);
            // shader->SetMat4("globalVsmShadowMatrix", csm.projectionViewSample);

            //STRATUS_LOG << glm::vec2(frame_->vsmc.projectionViewSample * glm::vec4(frame_->camera->GetPosition(), 1.0f)) << std::endl;
            
            // We need to use the old page partitioning scheme to calculate the viewport
            // info
            // const u32 clearValue = FloatBitsToUint(1.0f);
            // depth->ClearLayerRegion(
            //     0, 
            //     0, 
            //     startX, 
            //     startY, 
            //     sizeX * pageGroupWindowWidth, 
            //     sizeY * pageGroupWindowHeight, 
            //     (const void *)&clearValue
            // );
            glViewport(
                minPageGroupX * pageGroupWindowWidth, 
                minPageGroupY * pageGroupWindowHeight, 
                sizeX * pageGroupWindowWidth, 
                sizeY * pageGroupWindowHeight
            );

            RenderImmediate_(frame_->drawCommands->dynamicPbrMeshes, selectDynamic, cascade, true);
            RenderImmediate_(frame_->drawCommands->staticPbrMeshes, selectStatic, cascade, true);

            // STRATUS_LOG << minPageGroupX * pageGroupWindowWidth << " "
            //             << minPageGroupY * pageGroupWindowHeight << " "
            //             << sizeX * pageGroupWindowWidth << " "
            //             << sizeY * pageGroupWindowHeight << std::endl;

            UnbindShader_();
        }
    }

    //frame_->vsmc.pageResidencyTable.UnmapMemory();
    
    frame_->vsmc.fbo.Unbind();

    glDisable(GL_POLYGON_OFFSET_FILL);
    glDisable(GL_DEPTH_CLAMP);
}

void RendererBackend::RenderSsaoOcclude_() {
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    // Aspect ratio
    const f32 ar        = f32(frame_->viewportWidth) / f32(frame_->viewportHeight);
    // Distance to the view projection plane
    const f32 g         = 1.0f / glm::tan(frame_->fovy.value() / 2.0f);
    const f32 w         = frame_->viewportWidth;
    // Gets fed into sigma value
    const f32 intensity = 5.0f;

    BindShader_(state_.ssaoOcclude.get());
    state_.ssaoOcclusionBuffer.Bind();
    state_.ssaoOcclude->BindTexture("structureBuffer", state_.currentFrame.structure);
    state_.ssaoOcclude->BindTexture("rotationLookup", state_.ssaoOffsetLookup);
    state_.ssaoOcclude->SetFloat("aspectRatio", ar);
    state_.ssaoOcclude->SetFloat("projPlaneZDist", g);
    state_.ssaoOcclude->SetFloat("windowHeight", frame_->viewportHeight);
    state_.ssaoOcclude->SetFloat("windowWidth", w);
    state_.ssaoOcclude->SetFloat("intensity", intensity);
    RenderQuad_();
    state_.ssaoOcclusionBuffer.Unbind();
    UnbindShader_();

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

void RendererBackend::RenderSsaoBlur_() {
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    BindShader_(state_.ssaoBlur.get());
    state_.ssaoOcclusionBlurredBuffer.Bind();
    state_.ssaoBlur->BindTexture("structureBuffer", state_.currentFrame.structure);
    state_.ssaoBlur->BindTexture("occlusionBuffer", state_.ssaoOcclusionTexture);
    state_.ssaoBlur->SetFloat("windowWidth", frame_->viewportWidth);
    state_.ssaoBlur->SetFloat("windowHeight", frame_->viewportHeight);
    RenderQuad_();
    state_.ssaoOcclusionBlurredBuffer.Unbind();
    UnbindShader_();

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

void RendererBackend::RenderAtmosphericShadowing_() {
    if (!frame_->vsmc.worldLight->GetEnabled()) return;

    constexpr f32 preventDivByZero = std::numeric_limits<f32>::epsilon();

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    auto re = std::default_random_engine{};
    const f32 n = frame_->vsmc.worldLight->GetAtmosphericNumSamplesPerPixel();
    // On the range [0.0, 1/n)
    std::uniform_real_distribution<f32> real(0.0f, 1.0f / n);
    const glm::vec2 noiseShift(real(re), real(re));
    const f32 dmin     = frame_->znear;
    const f32 dmax     = frame_->vsmc.zfar;
    const f32 lambda   = frame_->vsmc.worldLight->GetAtmosphericParticleDensity();
    // cbrt = cube root
    const f32 cubeR    = std::cbrt(frame_->vsmc.worldLight->GetAtmosphericScatterControl());
    const f32 g        = (1.0f - cubeR) / (1.0f + cubeR + preventDivByZero);
    // aspect ratio
    const f32 ar       = f32(frame_->viewportWidth) / f32(frame_->viewportHeight);
    // g in frustum parameters
    const f32 projDist = 1.0f / glm::tan(frame_->fovy.value() / 2.0f);
    const glm::vec3 frustumParams(ar / projDist, 1.0f / projDist, dmin);
    const glm::mat4 shadowMatrix = frame_->vsmc.projectionViewSample * frame_->camera->GetWorldTransform();
    //const glm::mat4 shadowMatrix = frame_->vsmc.cascades[0].projectionViewRender * frame_->camera->GetWorldTransform();
    const glm::vec3 anisotropyConstants(1 - g, 1 + g * g, 2 * g);
    const glm::vec4 shadowSpaceCameraPos = frame_->vsmc.projectionViewSample * glm::vec4(frame_->camera->GetPosition(), 1.0f);
    const glm::vec3 normalizedCameraLightDirection = frame_->vsmc.worldLightDirectionCameraSpace;

    const auto timePoint = std::chrono::high_resolution_clock::now();
    const f32 milliseconds = f32(std::chrono::time_point_cast<std::chrono::milliseconds>(timePoint).time_since_epoch().count());

    BindShader_(state_.atmospheric.get());
    InitCoreCSMData_(state_.atmospheric.get());
    state_.atmosphericFbo.Bind();
    state_.atmospheric->SetVec3("frustumParams", frustumParams);
    state_.atmospheric->SetMat4("shadowMatrix", shadowMatrix);
    state_.atmospheric->BindTexture("structureBuffer", state_.currentFrame.structure);
    // state_.atmospheric->BindTexture("infiniteLightShadowMap", frame_->vsmc.vsm); //*frame_->vsmc.fbo.GetDepthStencilAttachment());
    // state_.atmospheric->BindTexture("infiniteLightShadowMapNonFiltered", frame_->vsmc.vsm);
    state_.atmospheric->SetFloat("time", milliseconds);
    state_.atmospheric->SetFloat("baseCascadeMaxDepth", frame_->vsmc.baseCascadeDiameter / 2.0f);
    state_.atmospheric->SetFloat("maxCascadeDepth", frame_->vsmc.zfar);
    
    // Set up cascade data
    // for (i32 i = 0; i < 4; ++i) {
    //     const auto& cascade = frame_->vsmc;
    //     const std::string si = "[" + std::to_string(i) + "]";
    //     state_.atmospheric->SetFloat("maxCascadeDepth" + si, cascade.zfar);
    //     //if (i > 0) {
    //     //    const std::string sim1 = "[" + std::to_string(i - 1) + "]";
    //     //    state_.atmospheric->SetMat4("cascade0ToCascadeK" + sim1, cascade.sampleCascade0ToCurrent);
    //     //}
    // }

    state_.atmospheric->BindTexture("noiseTexture", state_.atmosphericNoiseTexture);
    state_.atmospheric->SetFloat("minAtmosphereDepth", dmin);
    state_.atmospheric->SetFloat("atmosphereDepthDiff", dmax - dmin);
    state_.atmospheric->SetFloat("atmosphereDepthRatio", dmax / dmin);
    state_.atmospheric->SetFloat("atmosphereFogDensity", lambda);
    state_.atmospheric->SetVec3("anisotropyConstants", anisotropyConstants);
    state_.atmospheric->SetVec4("shadowSpaceCameraPos", shadowSpaceCameraPos);
    state_.atmospheric->SetVec3("normalizedCameraLightDirection", normalizedCameraLightDirection);
    state_.atmospheric->SetVec2("noiseShift", noiseShift);
    const Texture& colorTex = state_.atmosphericFbo.GetColorAttachments()[0];
    state_.atmospheric->SetFloat("windowWidth", f32(colorTex.Width()));
    state_.atmospheric->SetFloat("windowHeight", f32(colorTex.Height()));

    glViewport(0, 0, colorTex.Width(), colorTex.Height());
    RenderQuad_();
    state_.atmosphericFbo.Unbind();
    UnbindShader_();

    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

void RendererBackend::InitVplFrameData_(const VplDistVector_& perVPLDistToViewer) {
    std::vector<GpuVplData> vplData(perVPLDistToViewer.size());
    for (usize i = 0; i < perVPLDistToViewer.size(); ++i) {
        const VirtualPointLight* point = (const VirtualPointLight *)perVPLDistToViewer[i].key.get();
        GpuVplData& data = vplData[i];
        data.position = GpuVec(glm::vec4(point->GetPosition(), 1.0f));
        data.farPlane = point->GetFarPlane();
        data.radius = point->GetRadius();
        data.intensity = point->GetIntensity();
    }
    state_.vpls.vplData.CopyDataToBuffer(0, sizeof(GpuVplData) * vplData.size(), (const void *)vplData.data());
}

static inline void PerformPointLightGeometryCulling(
    Pipeline& pipeline,
    const usize lod,
    const std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>& commands,
    const std::vector<GpuCommandReceiveManagerPtr>& receivers,
    const std::function<GpuBuffer (const GpuCommandReceiveManagerPtr&, const RenderFaceCulling& cull)>& select,
    const std::vector<glm::mat4, StackBasedPoolAllocator<glm::mat4>>& viewProj
) {
    for (usize i = 0; i < viewProj.size(); ++i) {
        pipeline.SetMat4("viewProj[" + std::to_string(i) + "]", viewProj[i]);
    }

    for (auto& [cull, buffer] : commands) {
        if (buffer->NumDrawCommands() == 0) continue;

        pipeline.SetUint("numDrawCalls", (u32)buffer->NumDrawCommands());
        
        buffer->GetIndirectDrawCommandsBuffer(lod).BindBase(
            GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VISCULL_POINT_IN_DRAW_CALLS_BINDING_POINT);
        buffer->BindModelTransformBuffer(CURR_FRAME_MODEL_MATRICES_BINDING_POINT);
        buffer->BindAabbBuffer(AABB_BINDING_POINT);

        select(receivers[0], cull).BindBase(
            GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VISCULL_POINT_OUT_DRAW_CALLS_FACE0_BINDING_POINT);
        select(receivers[1], cull).BindBase(
            GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VISCULL_POINT_OUT_DRAW_CALLS_FACE1_BINDING_POINT);
        select(receivers[2], cull).BindBase(
            GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VISCULL_POINT_OUT_DRAW_CALLS_FACE2_BINDING_POINT);
        select(receivers[3], cull).BindBase(
            GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VISCULL_POINT_OUT_DRAW_CALLS_FACE3_BINDING_POINT);
        select(receivers[4], cull).BindBase(
            GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VISCULL_POINT_OUT_DRAW_CALLS_FACE4_BINDING_POINT);
        select(receivers[5], cull).BindBase(
            GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VISCULL_POINT_OUT_DRAW_CALLS_FACE5_BINDING_POINT);

        pipeline.DispatchCompute(1, 1, 1);
        pipeline.SynchronizeMemory();
    }
}

void RendererBackend::UpdatePointLights_(
    VplDistMultiSet_& perLightDistToViewerSet,
    VplDistVector_& perLightDistToViewerVec,
    VplDistMultiSet_& perLightShadowCastingDistToViewerSet,
    VplDistVector_& perLightShadowCastingDistToViewerVec,
    VplDistMultiSet_& perVPLDistToViewerSet,
    VplDistVector_& perVPLDistToViewerVec,
    std::vector<i32, StackBasedPoolAllocator<i32>>& visibleVplIndices) {

    const Camera& c = *frame_->camera;

    const bool worldLightEnabled = frame_->vsmc.worldLight->GetEnabled();
    const bool giEnabled = worldLightEnabled && frame_->settings.globalIlluminationEnabled;

    perLightDistToViewerSet.clear();
    perLightDistToViewerVec.clear();

    perLightShadowCastingDistToViewerSet.clear();
    perLightShadowCastingDistToViewerVec.clear();

    perVPLDistToViewerSet.clear();
    perVPLDistToViewerVec.clear();

    visibleVplIndices.clear();

    perLightDistToViewerVec.reserve(state_.maxTotalRegularLightsPerFrame);
    perLightShadowCastingDistToViewerVec.reserve(state_.maxShadowCastingLightsPerFrame);
    if (worldLightEnabled) {
        perVPLDistToViewerVec.reserve(MAX_TOTAL_VPLS_BEFORE_CULLING);
    }

    // Init per light instance data
    for (auto& light : frame_->lights) {
        const f64 distance = glm::distance(c.GetPosition(), light->GetPosition());
        if (light->IsVirtualLight()) {
            //if (giEnabled && distance <= MAX_VPL_DISTANCE_TO_VIEWER) {
            //if (giEnabled && IsSphereInFrustum(light->GetPosition(), light->GetRadius(), frame_->viewFrustumPlanes)) {
            if (giEnabled) {
                perVPLDistToViewerSet.insert(VplDistKey_(light, distance));
            }
        }
        else {
            perLightDistToViewerSet.insert(VplDistKey_(light, distance));
        }

        if ( !light->IsVirtualLight() && light->CastsShadows() ) {
            perLightShadowCastingDistToViewerSet.insert(VplDistKey_(light, distance));
        }
    }

    for (auto& entry : perLightDistToViewerSet) {
        perLightDistToViewerVec.push_back(entry);
        if (perLightDistToViewerVec.size() >= perLightDistToViewerVec.capacity()) break;
    }

    for (auto& entry : perLightShadowCastingDistToViewerSet) {
        perLightShadowCastingDistToViewerVec.push_back(entry);
        if (perLightShadowCastingDistToViewerVec.size() >= perLightShadowCastingDistToViewerVec.capacity()) break;
    }

    // Remove vpls exceeding absolute maximum
    if (giEnabled) {
        for (auto& entry : perVPLDistToViewerSet) {
            perVPLDistToViewerVec.push_back(entry);
            if (perVPLDistToViewerVec.size() >= perVPLDistToViewerVec.capacity()) break;
        }

        InitVplFrameData_(perVPLDistToViewerVec);
        PerformVirtualPointLightCullingStage1_(perVPLDistToViewerVec, visibleVplIndices);
    }

    // Check if any need to have a new shadow map pulled from the cache
    for (const auto&[light, _] : perLightShadowCastingDistToViewerVec) {
        if (!ShadowMapExistsForLight_(light)) {
            frame_->lightsToUpdate.PushBack(light);
        }
    }

    for (usize i = 0, updates = 0; i < visibleVplIndices.size() && updates < state_.maxShadowUpdatesPerFrame; ++i) {
        const i32 index = visibleVplIndices[i];
        auto light = perVPLDistToViewerVec[index].key;
        if (!ShadowMapExistsForLight_(light)) {
            // Pushing to front will cause the light update queue to be reordered biased towards VPLs
            // close to the camera
            frame_->lightsToUpdate.PushFront(light);
            ++updates;
        }
    }

    // Make sure we have enough space to generate the point light draw calls
    for (usize i = 0; i < state_.dynamicPerPointLightDrawCalls.size(); ++i) {
        state_.dynamicPerPointLightDrawCalls[i]->EnsureCapacity(frame_->drawCommands);
        state_.staticPerPointLightDrawCalls[i]->EnsureCapacity(frame_->drawCommands);
    }

    // Set blend func just for shadow pass
    // glBlendFunc(GL_ONE, GL_ONE);
    glEnable(GL_DEPTH_TEST);
    // Perform the shadow volume pre-pass
    for (i32 shadowUpdates = 0; shadowUpdates < state_.maxShadowUpdatesPerFrame && frame_->lightsToUpdate.Size() > 0; ++shadowUpdates) {
        auto light = frame_->lightsToUpdate.PopFront();
        // Ideally this won't be needed but just in case
        if ( !light->CastsShadows() ) continue;
        //const f64 distance = perLightShadowCastingDistToViewer.find(light)->second;
    
        // TODO: Make this work with spotlights
        //PointLightPtr point = (PointLightPtr)light;
        PointLight * point = (PointLight *)light.get();

        // if (point->IsVirtualLight() && !IsPointInFrustum(point->GetPosition(), frame_->viewFrustumPlanes)) {
        //     frame_->lightsToUpdate.PushBack(light);
        //     continue;
        // }
        if (point->IsVirtualLight() && !IsSphereInFrustum(point->GetPosition(), point->GetRadius(), frame_->viewFrustumPlanes)) {
            frame_->lightsToUpdate.PushBack(light);
            continue;
        }

        auto& cache = GetSmapCacheForLight_(light);
        GpuAtlasEntry smap = GetOrAllocateShadowMapForLight_(light);

        const auto cubeMapWidth = cache.buffers[smap.index].GetDepthStencilAttachment()->Width();
        const auto cubeMapHeight = cache.buffers[smap.index].GetDepthStencilAttachment()->Height();
        const glm::mat4 lightPerspective = glm::perspective<f32>(glm::radians(90.0f), f32(cubeMapWidth) / f32(cubeMapHeight), point->GetNearPlane(), point->GetFarPlane());

        // glBindFramebuffer(GL_FRAMEBUFFER, smap.frameBuffer);
        if (cache.buffers[smap.index].GetColorAttachments().size() > 0) {
            cache.buffers[smap.index].GetColorAttachments()[0].ClearLayer(0, smap.layer, nullptr);
        }
        f32 depthClear = 1.0f;
        cache.buffers[smap.index].GetDepthStencilAttachment()->ClearLayer(0, smap.layer, &depthClear);

        cache.buffers[smap.index].Bind();
        glViewport(0, 0, cubeMapWidth, cubeMapHeight);
        // Current pass only cares about depth buffer
        // glClear(GL_DEPTH_BUFFER_BIT);

        Pipeline * shader = light->IsVirtualLight() ? state_.vplShadows.get() : state_.shadows.get();
        auto transforms = GenerateLightViewTransforms(point->GetPosition(), frame_->perFrameScratchMemory);

        std::vector<glm::mat4, StackBasedPoolAllocator<glm::mat4>> lightViewProj(
            {
                lightPerspective * transforms[0],
                lightPerspective * transforms[1],
                lightPerspective * transforms[2],
                lightPerspective * transforms[3],
                lightPerspective * transforms[4],
                lightPerspective * transforms[5],
            },

            StackBasedPoolAllocator<glm::mat4>(frame_->perFrameScratchMemory)
        );

        // Perform visibility culling

        state_.viscullPointLights->Bind();

        PerformPointLightGeometryCulling(
            *state_.viscullPointLights.get(),
            light->IsVirtualLight() ? frame_->drawCommands->NumLods() - 1 : 0, // lod
            frame_->drawCommands->staticPbrMeshes,
            state_.staticPerPointLightDrawCalls,
            [](const GpuCommandReceiveManagerPtr& manager, const RenderFaceCulling& cull) {
                return manager->staticPbrMeshes.find(cull)->second->GetCommandBuffer();
            },
            lightViewProj
        );

        if (!light->IsStaticLight() && !light->IsVirtualLight()) {
            PerformPointLightGeometryCulling(
                *state_.viscullPointLights.get(),
                0, // lod
                frame_->drawCommands->dynamicPbrMeshes,
                state_.dynamicPerPointLightDrawCalls,
                [](const GpuCommandReceiveManagerPtr& manager, const RenderFaceCulling& cull) {
                    return manager->dynamicPbrMeshes.find(cull)->second->GetCommandBuffer();
                },
                lightViewProj
            );
        }

        state_.viscullPointLights->Unbind();

        for (usize i = 0; i < lightViewProj.size(); ++i) {
            const glm::mat4& projectionView = lightViewProj[i];

            // * 6 since each cube map is accessed by a layer-face which is divisible by 6
            BindShader_(shader);
            shader->SetInt("layer", i32(smap.layer * 6 + i));
            shader->SetMat4("shadowMatrix", projectionView);
            shader->SetVec3("lightPos", light->GetPosition());
            shader->SetFloat("farPlane", point->GetFarPlane());
            shader->SetFloat("alphaDepthTestThreshold", frame_->settings.GetAlphaDepthTestThreshold());

            if (point->IsVirtualLight()) {
                // Use lower LOD
                const usize lod = frame_->drawCommands->NumLods() - 1;
                const CommandBufferSelectionFunction select = [this, lod, i](GpuCommandBufferPtr& b) {
                    //return b->GetVisibleLowestLodDrawCommandsBuffer();
                    //return b->GetIndirectDrawCommandsBuffer(lod);
                    const auto cull = b->GetFaceCulling();
                    return state_.staticPerPointLightDrawCalls[i]->staticPbrMeshes.find(cull)->second->GetCommandBuffer();
                };
                RenderImmediate_(frame_->drawCommands->staticPbrMeshes, select, 0, false);
                //RenderImmediate_(frame_->instancedDynamicPbrMeshes[frame_->instancedDynamicPbrMeshes.size() - 1]);

                const glm::mat4 projectionViewNoTranslate = lightPerspective * glm::mat4(glm::mat3(transforms[i]));

                glDepthFunc(GL_LEQUAL);

                BindShader_(state_.skyboxLayered.get());
                state_.skyboxLayered->SetInt("layer", i32(smap.layer * 6 + i));

                auto tmp = frame_->settings.GetSkyboxIntensity();
                if (tmp > 1.0f) {
                    frame_->settings.SetSkyboxIntensity(1.0f);
                }
                
                RenderSkybox_(state_.skyboxLayered.get(), projectionViewNoTranslate);

                if (tmp > 1.0f) {
                    frame_->settings.SetSkyboxIntensity(tmp);
                }

                glDepthFunc(GL_LESS);
            }
            else {
                const CommandBufferSelectionFunction selectDynamic = [this, i](GpuCommandBufferPtr& b) {
                    const auto cull = b->GetFaceCulling();
                    return state_.dynamicPerPointLightDrawCalls[i]->dynamicPbrMeshes.find(cull)->second->GetCommandBuffer();
                    //return b->GetIndirectDrawCommandsBuffer(0);
                };

                const CommandBufferSelectionFunction selectStatic = [this, i](GpuCommandBufferPtr& b) {
                    const auto cull = b->GetFaceCulling();
                    return state_.staticPerPointLightDrawCalls[i]->staticPbrMeshes.find(cull)->second->GetCommandBuffer();
                    //return b->GetIndirectDrawCommandsBuffer(0);
                };

                RenderImmediate_(frame_->drawCommands->staticPbrMeshes, selectStatic, 0, false);
                if ( !point->IsStaticLight() ) RenderImmediate_(frame_->drawCommands->dynamicPbrMeshes, selectDynamic, 0, false);
            }

            UnbindShader_();
        }

        // Unbind
        cache.buffers[smap.index].Unbind();
    }
}

void RendererBackend::PerformVirtualPointLightCullingStage1_(
    VplDistVector_& perVPLDistToViewer,
    std::vector<i32, StackBasedPoolAllocator<i32>>& visibleVplIndices) {

    if (perVPLDistToViewer.size() == 0) return;

    state_.vplCulling->Bind();

    const Camera & lightCam = *frame_->vsmc.worldLightCamera;
    // glm::mat4 lightView = lightCam.getViewTransform();
    const glm::vec3 direction = lightCam.GetDirection();

    state_.vplCulling->SetVec3("infiniteLightDirection", direction);
    state_.vplCulling->SetInt("totalNumLights", perVPLDistToViewer.size());

    // Set up # visible atomic counter
    i32 numVisible = 0;
    //state_.vpls.vplNumVisible.CopyDataToBuffer(0, sizeof(i32), (const void *)&numVisible);
    //state_.vpls.vplNumVisible.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);

    // Bind light data and visibility indices
    state_.vpls.vplVisibleIndices.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_NUM_LIGHTS_VISIBLE_BINDING_POINT);
    state_.vpls.vplData.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_LIGHT_DATA_UNEDITED_BINDING_POINT);
    state_.vpls.vplUpdatedData.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_LIGHT_DATA_BINDING_POINT);

    InitCoreCSMData_(state_.vplCulling.get());
    state_.vplCulling->DispatchCompute(1, 1, 1);
    state_.vplCulling->SynchronizeCompute();

    state_.vplCulling->Unbind();
}

// void RendererBackend::PerformVirtualPointLightCullingStage2_(
//     const std::vector<std::pair<LightPtr, f64>>& perVPLDistToViewer,
//     const std::vector<i32>& visibleVplIndices) {
void RendererBackend::PerformVirtualPointLightCullingStage2_(
    const VplDistVector_& perVPLDistToViewer) {

    // i32 totalVisible = *(i32 *)state_.vpls.vplNumVisible.MapMemory();
    // state_.vpls.vplNumVisible.UnmapMemory();

    //if (perVPLDistToViewer.size() == 0 || visibleVplIndices.size() == 0) return;
    if (perVPLDistToViewer.size() == 0) return;

    i32* visibleVplIndices = (i32*)state_.vpls.vplVisibleIndices.MapMemory(GPU_MAP_READ);
    // First index is reserved for the size of the array
    const i32 totalVisible = visibleVplIndices[0];
    visibleVplIndices += 1;
    
    if (totalVisible == 0) {
        state_.vpls.vplVisibleIndices.UnmapMemory();
        return;
    }

    // Pack data into system memory
    std::vector<GpuTextureHandle, StackBasedPoolAllocator<GpuTextureHandle>> diffuseHandles(StackBasedPoolAllocator<GpuTextureHandle>(frame_->perFrameScratchMemory));
    diffuseHandles.reserve(totalVisible);
    std::vector<GpuAtlasEntry, StackBasedPoolAllocator<GpuAtlasEntry>> shadowDiffuseIndices(StackBasedPoolAllocator<GpuAtlasEntry>(frame_->perFrameScratchMemory));
    shadowDiffuseIndices.reserve(totalVisible);
    for (usize i = 0; i < totalVisible; ++i) {
        const i32 index = visibleVplIndices[i];
        const VirtualPointLight * point = (const VirtualPointLight *)perVPLDistToViewer[index].key.get();
        auto smap = GetOrAllocateShadowMapForLight_(perVPLDistToViewer[index].key);
        shadowDiffuseIndices.push_back(smap);
    }

    state_.vpls.vplVisibleIndices.UnmapMemory();
    visibleVplIndices = nullptr;

    // Move data to GPU memory
    state_.vpls.shadowDiffuseIndices.CopyDataToBuffer(0, sizeof(GpuAtlasEntry) * shadowDiffuseIndices.size(), (const void *)shadowDiffuseIndices.data());

    const Camera & lightCam = *frame_->vsmc.worldLightCamera;
    // glm::mat4 lightView = lightCam.getViewTransform();
    const glm::vec3 direction = lightCam.GetDirection();

    state_.vplColoring->Bind();

    // Bind inputs
    auto& cache = vplSmapCache_;
    state_.vplColoring->SetVec3("infiniteLightDirection", direction);
    state_.vplColoring->SetVec3("infiniteLightColor", frame_->vsmc.worldLight->GetLuminance());
    // for (usize i = 0; i < cache.buffers.size(); ++i) {
    //     state_.vplColoring->BindTexture("diffuseCubeMaps[" + std::to_string(i) + "]", cache.buffers[i].GetColorAttachments()[0]);
    //     state_.vplColoring->BindTexture("shadowCubeMaps[" + std::to_string(i) + "]", *cache.buffers[i].GetDepthStencilAttachment());
    // }

    state_.vpls.vplVisibleIndices.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_NUM_LIGHTS_VISIBLE_BINDING_POINT);
    //state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 3);
    state_.vpls.shadowDiffuseIndices.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_SHADOW_ATLAS_INDICES_BINDING_POINT);

    InitCoreCSMData_(state_.vplColoring.get());

    // Bind outputs
    state_.vpls.vplUpdatedData.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_LIGHT_DATA_BINDING_POINT);

    // Dispatch and synchronize
    state_.vplColoring->DispatchCompute(1, 1, 1);
    state_.vplColoring->SynchronizeCompute();

    state_.vplColoring->Unbind();
}

void RendererBackend::ComputeVirtualPointLightGlobalIllumination_(const VplDistVector_& perVPLDistToViewer, const f64 deltaSeconds) {
    if (perVPLDistToViewer.size() == 0) return;

    // auto space = LogSpace<f32>(1, 512, 30);
    // for (const auto& s : space) std::cout << s << " ";
    // std::cout << std::endl;

    const auto timePoint = std::chrono::high_resolution_clock::now();
    const f32 milliseconds = f32(std::chrono::time_point_cast<std::chrono::milliseconds>(timePoint).time_since_epoch().count());

    glDisable(GL_DEPTH_TEST);
    BindShader_(state_.vplGlobalIllumination.get());
    state_.vpls.vplGIFbo.Bind();
    glViewport(0, 0, state_.vpls.vplGIFbo.GetColorAttachments()[0].Width(), state_.vpls.vplGIFbo.GetColorAttachments()[0].Height());

    // Set up infinite light color
    auto& cache = vplSmapCache_;
    const glm::vec3 lightColor = frame_->vsmc.worldLight->GetLuminance();
    state_.vplGlobalIllumination->SetVec3("infiniteLightColor", lightColor);

    state_.vplGlobalIllumination->SetInt("numTilesX", frame_->viewportWidth  / state_.vpls.tileXDivisor);
    state_.vplGlobalIllumination->SetInt("numTilesY", frame_->viewportHeight / state_.vpls.tileYDivisor);

    // All relevant rendering data is moved to the GPU during the light cull phase
    state_.vpls.vplUpdatedData.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_LIGHT_DATA_BINDING_POINT);
    state_.vpls.vplVisibleIndices.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_NUM_LIGHTS_VISIBLE_BINDING_POINT);
    //state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 2);
    state_.vpls.shadowDiffuseIndices.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_SHADOW_ATLAS_INDICES_BINDING_POINT);
    haltonSequence_.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VPL_HALTON_SEQUENCE_BINDING_POINT);
    state_.vplGlobalIllumination->SetInt("haltonSize", i32(haltonSequence.size()));

    state_.vplGlobalIllumination->SetMat4("invProjectionView", frame_->invProjectionView);
    // for (usize i = 0; i < cache.buffers.size(); ++i) {
    //     state_.vplGlobalIllumination->BindTexture("shadowCubeMaps[" + std::to_string(i) + "]", *cache.buffers[i].GetDepthStencilAttachment());
    // }

    state_.vplGlobalIllumination->SetFloat("minRoughness", frame_->settings.GetMinRoughness());
    state_.vplGlobalIllumination->SetBool("usePerceptualRoughness", frame_->settings.usePerceptualRoughness);
    state_.vplGlobalIllumination->BindTexture("screen", state_.lightingColorBuffer);
    state_.vplGlobalIllumination->BindTexture("gDepth", state_.currentFrame.depth);
    state_.vplGlobalIllumination->BindTexture("gNormal", state_.currentFrame.normals);
    state_.vplGlobalIllumination->BindTexture("gAlbedo", state_.currentFrame.albedo);
    //state_.vplGlobalIllumination->BindTexture("gBaseReflectivity", state_.currentFrame.baseReflectivity);
    state_.vplGlobalIllumination->BindTexture("gRoughnessMetallicReflectivity", state_.currentFrame.roughnessMetallicReflectivity);
    state_.vplGlobalIllumination->BindTexture("ssao", state_.ssaoOcclusionBlurredTexture);
    state_.vplGlobalIllumination->BindTexture("historyDepth", state_.vpls.vplGIDenoisedPrevFrameFbo.GetColorAttachments()[3]);
    state_.vplGlobalIllumination->SetFloat("time", milliseconds);
    state_.vplGlobalIllumination->SetInt("frameCount", i32(INSTANCE(Engine)->FrameCount()));
    state_.vplGlobalIllumination->SetFloat("minGiOcclusionFactor", frame_->settings.GetMinGiOcclusionFactor());

    state_.vplGlobalIllumination->SetVec3("fogColor", frame_->settings.GetFogColor());
    state_.vplGlobalIllumination->SetFloat("fogDensity", frame_->settings.GetFogDensity());

    const Camera& camera = *frame_->camera;
    state_.vplGlobalIllumination->SetVec3("viewPosition", camera.GetPosition());
    state_.vplGlobalIllumination->SetInt("viewportWidth", frame_->viewportWidth);
    state_.vplGlobalIllumination->SetInt("viewportHeight", frame_->viewportHeight);

    RenderQuad_();
    
    UnbindShader_();
    state_.vpls.vplGIFbo.Unbind();

    std::vector<FrameBuffer*, StackBasedPoolAllocator<FrameBuffer*>> buffers({
        &state_.vpls.vplGIDenoisedFbo1,
        &state_.vpls.vplGIDenoisedFbo2
        },

        StackBasedPoolAllocator<FrameBuffer*>(frame_->perFrameScratchMemory)
    );

    Texture indirectIllum = state_.vpls.vplGIFbo.GetColorAttachments()[0];
    Texture indirectShadows = state_.vpls.vplGIFbo.GetColorAttachments()[1];

    BindShader_(state_.vplGlobalIlluminationDenoising.get());
    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    state_.vplGlobalIlluminationDenoising->BindTexture("screen", state_.lightingColorBuffer);
    state_.vplGlobalIlluminationDenoising->BindTexture("albedo", state_.currentFrame.albedo);
    state_.vplGlobalIlluminationDenoising->BindTexture("velocity", state_.currentFrame.velocity);
    state_.vplGlobalIlluminationDenoising->BindTexture("normal", state_.currentFrame.normals);
    state_.vplGlobalIlluminationDenoising->BindTexture("ids", state_.currentFrame.id);
    state_.vplGlobalIlluminationDenoising->BindTexture("depth", state_.currentFrame.depth);
    state_.vplGlobalIlluminationDenoising->BindTexture("structureBuffer", state_.currentFrame.structure);
    state_.vplGlobalIlluminationDenoising->BindTexture("prevNormal", state_.previousFrame.normals);
    state_.vplGlobalIlluminationDenoising->BindTexture("prevIds", state_.previousFrame.id);
    state_.vplGlobalIlluminationDenoising->BindTexture("prevDepth", state_.previousFrame.depth);
    state_.vplGlobalIlluminationDenoising->BindTexture("prevIndirectIllumination", state_.vpls.vplGIDenoisedPrevFrameFbo.GetColorAttachments()[1]);
    state_.vplGlobalIlluminationDenoising->BindTexture("originalNoisyIndirectIllumination", indirectShadows);
    state_.vplGlobalIlluminationDenoising->BindTexture("historyDepth", state_.vpls.vplGIDenoisedPrevFrameFbo.GetColorAttachments()[3]);
    state_.vplGlobalIlluminationDenoising->SetBool("final", false);
    state_.vplGlobalIlluminationDenoising->SetFloat("time", milliseconds);
    state_.vplGlobalIlluminationDenoising->SetFloat("framesPerSecond", f32(1.0 / deltaSeconds));
    state_.vplGlobalIlluminationDenoising->SetMat4("invProjectionView", frame_->invProjectionView);
    state_.vplGlobalIlluminationDenoising->SetMat4("prevInvProjectionView", frame_->prevInvProjectionView);

    usize bufferIndex = 0;
    const i32 maxReservoirMergingPasses = 1;
    const i32 maxIterations = 3;
    for (; bufferIndex < maxIterations; ++bufferIndex) {

        // The first iteration(s) is used for reservoir merging so we don't
        // start increasing the multiplier until after the reservoir merging passes
        const i32 i = bufferIndex; // bufferIndex < maxReservoirMergingPasses ? 0 : bufferIndex - maxReservoirMergingPasses + 1;
        const i32 multiplier = std::pow(2, i) - 1;
        FrameBuffer * buffer = buffers[bufferIndex % buffers.size()];

        buffer->Bind();
        state_.vplGlobalIlluminationDenoising->BindTexture("indirectIllumination", indirectIllum);
        state_.vplGlobalIlluminationDenoising->BindTexture("indirectShadows", indirectShadows);
        state_.vplGlobalIlluminationDenoising->SetInt("multiplier", multiplier);
        state_.vplGlobalIlluminationDenoising->SetInt("passNumber", i);
        state_.vplGlobalIlluminationDenoising->SetBool("mergeReservoirs", bufferIndex < maxReservoirMergingPasses);

        if (bufferIndex + 1 == maxIterations) {
            state_.vplGlobalIlluminationDenoising->SetBool("final", true);
        }

        RenderQuad_();

        buffer->Unbind();

        indirectIllum = buffer->GetColorAttachments()[1];
        indirectShadows = buffer->GetColorAttachments()[2];
    }

    UnbindShader_();
    --bufferIndex;

    FrameBuffer * last = buffers[bufferIndex % buffers.size()];
    state_.lightingFbo.CopyFrom(*last, BufferBounds{0, 0, frame_->viewportWidth, frame_->viewportHeight}, BufferBounds{0, 0, frame_->viewportWidth, frame_->viewportHeight}, BufferBit::COLOR_BIT, BufferFilter::NEAREST);

    // Swap current and previous frame
    auto tmp = *last;
    *last = state_.vpls.vplGIDenoisedPrevFrameFbo;
    state_.vpls.vplGIDenoisedPrevFrameFbo = tmp;
}

void RendererBackend::RenderScene(const f64 deltaSeconds) {
    CHECK_IS_APPLICATION_THREAD();

    const Camera& c = *frame_->camera;

    // Bind buffers
    GpuMeshAllocator::BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, MESH_DATA_BINDING_POINT);
    GpuMeshAllocator::BindElementArrayBuffer();

    VplDistMultiSet_ perLightDistToViewerSet(StackBasedPoolAllocator<VplDistKey_>(frame_->perFrameScratchMemory));
    VplDistVector_ perLightDistToViewerVec(StackBasedPoolAllocator<VplDistKey_>(frame_->perFrameScratchMemory));

    // // This one is just for shadow-casting lights
    VplDistMultiSet_ perLightShadowCastingDistToViewerSet(StackBasedPoolAllocator<VplDistKey_>(frame_->perFrameScratchMemory));
    VplDistVector_ perLightShadowCastingDistToViewerVec(StackBasedPoolAllocator<VplDistKey_>(frame_->perFrameScratchMemory));

    VplDistMultiSet_ perVPLDistToViewerSet(StackBasedPoolAllocator<VplDistKey_>(frame_->perFrameScratchMemory));
    VplDistVector_ perVPLDistToViewerVec(StackBasedPoolAllocator<VplDistKey_>(frame_->perFrameScratchMemory));

    std::vector<i32, StackBasedPoolAllocator<i32>> visibleVplIndices(StackBasedPoolAllocator<i32>(frame_->perFrameScratchMemory));

    // Perform point light pass
    UpdatePointLights_(
        perLightDistToViewerSet, perLightDistToViewerVec,
        perLightShadowCastingDistToViewerSet, perLightShadowCastingDistToViewerVec,
        perVPLDistToViewerSet, perVPLDistToViewerVec, 
        visibleVplIndices
    );

    glBlendFunc(state_.blendSFactor, state_.blendDFactor);
    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    // Generate the GBuffer
    RenderForwardPassPbr_();

    // Perform world light depth pass if enabled - needed by a lot of the rest of the frame so
    // do this first
    if (frame_->vsmc.worldLight->GetEnabled()) {
        RenderCSMDepth_();
    }

    // TEMP: Set up the light source
    //glm::vec3 lightPos(0.0f, 0.0f, 0.0f);
    //glm::vec3 lightColor(10.0f); 

    // Make sure some of our global GL states are set properly for primary rendering below
    glBlendFunc(state_.blendSFactor, state_.blendDFactor);
    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    //glEnable(GL_BLEND);

    // // Begin first SSAO pass (occlusion)
    // RenderSsaoOcclude_();

    // // Begin second SSAO pass (blurring)
    // RenderSsaoBlur_();

    // Begin atmospheric pass
    RenderAtmosphericShadowing_();

    // Begin deferred lighting pass
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    state_.lightingFbo.Bind();

    //_unbindAllTextures();
    Pipeline* lighting = state_.lighting.get();
    if (frame_->vsmc.worldLight->GetEnabled()) {
        lighting = state_.lightingWithInfiniteLight.get();
    }

    BindShader_(lighting);
    InitLights_(lighting, perLightDistToViewerVec, state_.maxShadowCastingLightsPerFrame);
    lighting->BindTexture("atmosphereBuffer", state_.atmosphericTexture);
    lighting->SetMat4("invProjectionView", frame_->invProjectionView);
    lighting->BindTexture("gDepth", state_.currentFrame.depth);
    lighting->BindTexture("gNormal", state_.currentFrame.normals);
    lighting->BindTexture("gAlbedo", state_.currentFrame.albedo);
    //lighting->BindTexture("gBaseReflectivity", state_.currentFrame.baseReflectivity);
    lighting->BindTexture("gRoughnessMetallicReflectivity", state_.currentFrame.roughnessMetallicReflectivity);
    lighting->BindTexture("ssao", state_.ssaoOcclusionBlurredTexture);
    lighting->BindTexture("ids", state_.currentFrame.id);
    lighting->SetFloat("windowWidth", frame_->viewportWidth);
    lighting->SetFloat("windowHeight", frame_->viewportHeight);
    lighting->SetVec3("fogColor", frame_->settings.GetFogColor());
    lighting->SetFloat("fogDensity", frame_->settings.GetFogDensity());
    RenderQuad_();
    state_.lightingFbo.Unbind();
    UnbindShader_();
    state_.finalScreenBuffer = state_.lightingFbo; // state_.lightingColorBuffer;

    // If world light is enabled perform VPL Global Illumination pass
    if (frame_->vsmc.worldLight->GetEnabled() && frame_->settings.globalIlluminationEnabled) {
        // Handle VPLs for global illumination (can't do this earlier due to needing position data from GBuffer)
        PerformVirtualPointLightCullingStage2_(perVPLDistToViewerVec);
        ComputeVirtualPointLightGlobalIllumination_(perVPLDistToViewerVec, deltaSeconds);
    }

    // Forward pass for all objects that don't interact with light (may also be used for transparency later as well)
    // flatPassFbo is a framebuffer view of lightingFbo and Gbuffer.velocity
    state_.flatPassFboCurrentFrame.CopyFrom(state_.currentFrame.fbo, BufferBounds{0, 0, frame_->viewportWidth, frame_->viewportHeight}, BufferBounds{0, 0, frame_->viewportWidth, frame_->viewportHeight}, BufferBit::DEPTH_BIT, BufferFilter::NEAREST);
    // Blit to default framebuffer - not that the framebuffer you are writing to has to match the internal format
    // of the framebuffer you are reading to!
    glEnable(GL_DEPTH_TEST);
    state_.flatPassFboCurrentFrame.Bind();

    // Skybox is one that does not interact with light at all
    RenderSkybox_();

    // No light interaction
    // TODO: Allow to cast shadows? May be useful in scenes that want to use
    // purely diffuse non-pbr objects which still cast shadows.
    RenderForwardPassFlat_();

    // Render bounding boxes
    //RenderBoundingBoxes_(frame_->drawCommands->flatMeshes);
    //RenderBoundingBoxes_(frame_->drawCommands->dynamicPbrMeshes);
    //RenderBoundingBoxes_(frame_->drawCommands->staticPbrMeshes);

    state_.flatPassFboCurrentFrame.Unbind();
    state_.finalScreenBuffer = state_.lightingFbo;// state_.lightingColorBuffer;
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Enable post-FX effects such as bloom
    PerformPostFxProcessing_();

    // Perform final drawing to screen + gamma correction
    FinalizeFrame_();

    // Update hierarchical depth buffer
    //UpdateHiZBuffer_();

    // Unbind element array buffer
    GpuMeshAllocator::UnbindElementArrayBuffer();
}

void RendererBackend::RenderForwardPassPbr_() {
    // Make sure to bind our own frame buffer for rendering
    state_.currentFrame.fbo.Bind();

    const CommandBufferSelectionFunction select = [](GpuCommandBufferPtr& b) {
        return b->GetVisibleDrawCommandsBuffer();
    };

    const auto& jitter = frame_->settings.taaEnabled ? frame_->jitterProjectionView : frame_->projectionView;

    // Perform depth prepass
    // BindShader_(state_.depthPrepass.get());

    // glEnable(GL_DEPTH_TEST);
    // glDepthFunc(GL_LESS);
    // glDepthMask(GL_TRUE);

    // state_.depthPrepass->SetMat4("projectionView", jitter);

    // RenderImmediate_(frame_->drawCommands->dynamicPbrMeshes, select, 0, false);
    // RenderImmediate_(frame_->drawCommands->staticPbrMeshes, select, 0, false);

    // UnbindShader_();

    // Begin geometry pass
    BindShader_(state_.geometry.get());

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    // glDepthFunc(GL_EQUAL);
    // glDepthMask(GL_FALSE);

    state_.geometry->SetMat4("jitterProjectionView", jitter);

    //glDepthFunc(GL_LEQUAL);

    Render_(*state_.geometry.get(), frame_->drawCommands->dynamicPbrMeshes, select, 0, true);
    Render_(*state_.geometry.get(), frame_->drawCommands->staticPbrMeshes, select, 0, true);

    state_.currentFrame.fbo.Unbind();

    UnbindShader_();

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
}

void RendererBackend::RenderForwardPassFlat_() {
    BindShader_(state_.forward.get());

    auto& jitter = frame_->settings.taaEnabled ? frame_->jitterProjectionView : frame_->projectionView;
    state_.forward->SetMat4("jitterProjectionView", jitter);

    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    const CommandBufferSelectionFunction select = [](GpuCommandBufferPtr& b) {
        return b->GetVisibleDrawCommandsBuffer();
    };

    Render_(*state_.forward.get(), frame_->drawCommands->flatMeshes, select, 0, false);

    UnbindShader_();

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
}

void RendererBackend::PerformPostFxProcessing_() {
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    //glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    PerformTaaPostFx_();

    PerformBloomPostFx_();

    PerformAtmosphericPostFx_();

    // Needs to happen before FXAA since FXAA works on color graded LDR values (not HDR)
    PerformGammaTonemapPostFx_();

    PerformFxaaPostFx_();

    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void RendererBackend::PerformBloomPostFx_() {
    if (!frame_->settings.bloomEnabled) return;

    // We use this so that we can avoid a final copy between the downsample and blurring stages
    std::vector<PostFXBuffer, StackBasedPoolAllocator<PostFXBuffer>> finalizedPostFxFrames(
        state_.numDownsampleIterations + state_.numUpsampleIterations,
        StackBasedPoolAllocator<PostFXBuffer>(frame_->perFrameScratchMemory)
    );
   
    Pipeline* bloom = state_.bloom.get();
    BindShader_(bloom);

    Texture lightingColorBuffer = state_.finalScreenBuffer.GetColorAttachments()[0];

    // Downsample stage
    bloom->SetBool("downsamplingStage", true);
    bloom->SetBool("upsamplingStage", false);
    bloom->SetBool("finalStage", false);
    bloom->SetBool("gaussianStage", false);
    for (i32 i = 0, gaussian = 0; i < state_.numDownsampleIterations; ++i, gaussian += 2) {
        PostFXBuffer& buffer = state_.postFxBuffers[i];
        Texture colorTex = buffer.fbo.GetColorAttachments()[0];
        auto width = colorTex.Width();
        auto height = colorTex.Height();
        bloom->SetFloat("viewportX", f32(width));
        bloom->SetFloat("viewportY", f32(height));
        buffer.fbo.Bind();
        glViewport(0, 0, width, height);
        if (i == 0) {
            bloom->BindTexture("mainTexture", lightingColorBuffer);
        }
        else {
            bloom->BindTexture("mainTexture", state_.postFxBuffers[i - 1].fbo.GetColorAttachments()[0]);
        }
        RenderQuad_();
        buffer.fbo.Unbind();

        // Now apply Gaussian blurring
        bool horizontal = false;
        bloom->SetBool("downsamplingStage", false);
        bloom->SetBool("gaussianStage", true);
        BufferBounds bounds = BufferBounds{0, 0, width, height};
        for (i32 i = 0; i < 2; ++i) {
            FrameBuffer& blurFbo = state_.gaussianBuffers[gaussian + i].fbo;
            FrameBuffer copyFromFbo;
            if (i == 0) {
                copyFromFbo = buffer.fbo;
            }
            else {
                copyFromFbo = state_.gaussianBuffers[gaussian].fbo;
            }

            bloom->SetBool("horizontal", horizontal);
            bloom->BindTexture("mainTexture", copyFromFbo.GetColorAttachments()[0]);
            horizontal = !horizontal;
            blurFbo.Bind();
            RenderQuad_();
            blurFbo.Unbind();
        }

        // Copy the end result back to the original buffer
        // buffer.fbo.copyFrom(_state.gaussianBuffers[gaussian + 1].fbo, bounds, bounds, BufferBit::COLOR_BIT, BufferFilter::LINEAR);
        finalizedPostFxFrames[i] = state_.gaussianBuffers[gaussian + 1];
    }

    // Upsample stage
    bloom->SetBool("downsamplingStage", false);
    bloom->SetBool("upsamplingStage", true);
    bloom->SetBool("finalStage", false);
    bloom->SetBool("gaussianStage", false);
    i32 postFXIndex = state_.numDownsampleIterations;
    for (i32 i = state_.numDownsampleIterations - 1; i >= 0; --i, ++postFXIndex) {
        PostFXBuffer& buffer = state_.postFxBuffers[postFXIndex];
        auto width = buffer.fbo.GetColorAttachments()[0].Width();
        auto height = buffer.fbo.GetColorAttachments()[0].Height();
        bloom->SetFloat("viewportX", f32(width));
        bloom->SetFloat("viewportY", f32(height));
        buffer.fbo.Bind();
        glViewport(0, 0, width, height);
        //bloom->bindTexture("mainTexture", _state.postFxBuffers[postFXIndex - 1].fbo.getColorAttachments()[0]);
        bloom->BindTexture("mainTexture", finalizedPostFxFrames[postFXIndex - 1].fbo.GetColorAttachments()[0]);
        if (i == 0) {
            bloom->BindTexture("bloomTexture", lightingColorBuffer);
            bloom->SetBool("finalStage", true);
        }
        else {
            //bloom->bindTexture("bloomTexture", _state.postFxBuffers[i - 1].fbo.getColorAttachments()[0]);
            bloom->BindTexture("bloomTexture", finalizedPostFxFrames[i - 1].fbo.GetColorAttachments()[0]);
        }
        RenderQuad_();
        buffer.fbo.Unbind();
        
        finalizedPostFxFrames[postFXIndex] = buffer;
        state_.finalScreenBuffer = buffer.fbo; //buffer.fbo.GetColorAttachments()[0];
    }

    UnbindShader_();
}

glm::vec3 RendererBackend::CalculateAtmosphericLightPosition_() const {
    const glm::mat4& projection = frame_->projection;
    // See page 354, eqs. 10.81 and 10.82
    const glm::vec3& normalizedLightDirCamSpace = frame_->vsmc.worldLightDirectionCameraSpace;
    const Texture& colorTex = state_.atmosphericTexture;
    const f32 w = colorTex.Width();
    const f32 h = colorTex.Height();
    const f32 xlight = w * ((projection[0][0] * normalizedLightDirCamSpace.x + 
                               projection[0][1] * normalizedLightDirCamSpace.y + 
                               projection[0][2] * normalizedLightDirCamSpace.z) / (2.0f * normalizedLightDirCamSpace.z) + 0.5f);
    const f32 ylight = h * ((projection[1][0] * normalizedLightDirCamSpace.x + 
                               projection[1][1] * normalizedLightDirCamSpace.y + 
                               projection[1][2] * normalizedLightDirCamSpace.z) / (2.0f * normalizedLightDirCamSpace.z) + 0.5f);
    
    return 2.0f * normalizedLightDirCamSpace.z * glm::vec3(xlight, ylight, 1.0f);
}

void RendererBackend::PerformAtmosphericPostFx_() {
    if (!frame_->vsmc.worldLight->GetEnabled()) return;

    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    const glm::vec3 lightPosition = CalculateAtmosphericLightPosition_();
    //const f32 sinX = stratus::sine(_frame->csc.worldLight->getRotation().x).value();
    //const f32 cosX = stratus::cosine(_frame->csc.worldLight->getRotation().x).value();
    const glm::vec3 lightColor = frame_->vsmc.worldLight->GetAtmosphereColor();// * glm::vec3(cosX, cosX, sinX);

    BindShader_(state_.atmosphericPostFx.get());
    state_.atmosphericPostFxBuffer.fbo.Bind();
    state_.atmosphericPostFx->BindTexture("atmosphereBuffer", state_.atmosphericTexture);
    state_.atmosphericPostFx->BindTexture("screenBuffer", state_.finalScreenBuffer.GetColorAttachments()[0]);
    state_.atmosphericPostFx->SetVec3("lightPosition", lightPosition);
    state_.atmosphericPostFx->SetVec3("lightColor", lightColor);
    RenderQuad_();
    state_.atmosphericPostFxBuffer.fbo.Unbind();
    UnbindShader_();

    state_.finalScreenBuffer = state_.atmosphericPostFxBuffer.fbo; //state_.atmosphericPostFxBuffer.fbo.GetColorAttachments()[0];
}

void RendererBackend::PerformFxaaPostFx_() {
    if (!frame_->settings.fxaaEnabled) return;

    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    // Perform luminance calculation pass
    BindShader_(state_.fxaaLuminance.get());
    
    state_.fxaaFbo1.fbo.Bind();
    state_.fxaaLuminance->BindTexture("screen", state_.finalScreenBuffer.GetColorAttachments()[0]);
    RenderQuad_();
    state_.fxaaFbo1.fbo.Unbind();

    UnbindShader_();

    state_.finalScreenBuffer = state_.fxaaFbo1.fbo; // state_.fxaaFbo1.fbo.GetColorAttachments()[0];

    // Perform smoothing pass
    BindShader_(state_.fxaaSmoothing.get());

    state_.fxaaFbo2.fbo.Bind();
    state_.fxaaSmoothing->BindTexture("screen", state_.finalScreenBuffer.GetColorAttachments()[0]);
    RenderQuad_();
    state_.fxaaFbo2.fbo.Unbind();

    UnbindShader_();

    state_.finalScreenBuffer = state_.fxaaFbo2.fbo; // state_.fxaaFbo2.fbo.GetColorAttachments()[0];
}

void RendererBackend::PerformTaaPostFx_() {
    if (!frame_->settings.taaEnabled) return;

    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    BindShader_(state_.taa.get());

    state_.taaFbo.fbo.Bind();

    state_.taa->BindTexture("screen", state_.finalScreenBuffer.GetColorAttachments()[0]);
    state_.taa->BindTexture("prevScreen", state_.previousFrameBuffer.GetColorAttachments()[0]);
    state_.taa->BindTexture("velocity", state_.currentFrame.velocity);
    state_.taa->BindTexture("previousVelocity", state_.previousFrame.velocity);

    RenderQuad_();

    state_.taaFbo.fbo.Unbind();

    UnbindShader_();

    state_.finalScreenBuffer = state_.taaFbo.fbo;

    // Update history texture
    state_.previousFrameBuffer.CopyFrom(
        state_.finalScreenBuffer,
        BufferBounds{ 0, 0, frame_->viewportWidth, frame_->viewportHeight },
        BufferBounds{ 0, 0, frame_->viewportWidth, frame_->viewportHeight },
        BufferBit::COLOR_BIT,
        BufferFilter::NEAREST
    );
}

void RendererBackend::PerformGammaTonemapPostFx_() {
    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    BindShader_(state_.gammaTonemap.get());
    state_.gammaTonemapFbo.fbo.Bind();
    state_.gammaTonemap->BindTexture("screen", state_.finalScreenBuffer.GetColorAttachments()[0]);
    RenderQuad_();
    UnbindShader_();

    state_.finalScreenBuffer = state_.gammaTonemapFbo.fbo; //state_.gammaTonemapFbo.fbo.GetColorAttachments()[0];
}

void RendererBackend::FinalizeFrame_() {
    // Copy final frame to current frame
    //state_.gammaTonemapFbo.fbo.CopyFrom()

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_CULL_FACE);
    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);
    //glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    // Now render the screen
    BindShader_(state_.fullscreen.get());

    state_.fullscreen->BindTexture("screen", state_.finalScreenBuffer.GetColorAttachments()[0]);
    state_.fullscreen->BindTexture("inputDepth", state_.currentFrame.depth);

    state_.fullscreen->BindTextureAsImage(
        "outputDepth", 
        state_.currentFrame.depthPyramid,
        0,
        true, 
        0, 
        ImageTextureAccessMode::IMAGE_WRITE_ONLY
    );

    RenderQuad_();
    UnbindShader_();

    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // glViewport(0, 0, 350, 350);
    // BindShader_(state_.fullscreenPages.get());
    // state_.fullscreenPages->SetFloat("znear", frame_->vsmc.znear);
    // state_.fullscreenPages->SetFloat("zfar", frame_->vsmc.zfar);
    // state_.fullscreenPages->BindTexture("depth", frame_->vsmc.vsm); //*frame_->vsmc.fbo.GetDepthStencilAttachment());
    // RenderQuad_();
    // UnbindShader_();

    // const auto numPagesAvailable = frame_->vsmc.cascadeResolutionXY / Texture::VirtualPageSizeXY();

    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // glViewport(frame_->viewportWidth - 350, 0, 350, 350);
    // BindShader_(state_.fullscreenPageGroups.get());
    // state_.fullscreenPageGroups->SetUint("numPageGroupsX", frame_->vsmc.numPageGroupsX);
    // state_.fullscreenPageGroups->SetUint("numPageGroupsY", frame_->vsmc.numPageGroupsY);
    // state_.fullscreenPageGroups->SetUint("numPagesXY", (u32)numPagesAvailable);
    // frame_->vsmc.pageGroupsToRender.BindBase(
    //     GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_PAGE_GROUPS_TO_RENDER_BINDING_POINT);
    // frame_->vsmc.pageResidencyTable.BindBase(
    //     GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING);
    // RenderQuad_();
    // UnbindShader_();
}

// See https://www.rastergrid.com/blog/2010/10/hierarchical-z-map-based-occlusion-culling/
// See https://vkguide.dev/docs/gpudriven/compute_culling/
void RendererBackend::UpdateHiZBuffer_() {
    state_.depthPyramid = state_.currentFrame.depthPyramid;

    u32 w = state_.depthPyramid.Width();
    u32 h = state_.depthPyramid.Height();

    BindShader_(state_.depthPyramidConstruct.get());

    const u32 numMips = 1 + u32(std::floor(std::log2(std::max(w, h))));

    for (u32 currLevel = 1; currLevel < numMips; ++currLevel) {
        w /= 2;
        h /= 2;

        if (w == 0) w = 1;
        if (h == 0) h = 1;

        const u32 numComputeGroupsX = u32(std::ceil(w / 8) + 1);
        const u32 numComputeGroupsY = u32(std::ceil(h / 8) + 1);

        const u32 prevLevel = currLevel - 1;

        state_.depthPyramidConstruct->BindTextureAsImage(
            "depthInput", 
            state_.depthPyramid, 
            prevLevel,
            true, 
            0, 
            ImageTextureAccessMode::IMAGE_READ_ONLY
        );

        state_.depthPyramidConstruct->BindTextureAsImage(
            "depthOutput", 
            state_.depthPyramid, 
            currLevel,
            true, 
            0, 
            ImageTextureAccessMode::IMAGE_WRITE_ONLY
        );

        state_.depthPyramidConstruct->DispatchCompute(numComputeGroupsX, numComputeGroupsY, 1);
        state_.depthPyramidConstruct->SynchronizeMemory();
    }

    UnbindShader_();
}

void RendererBackend::End() {
    CHECK_IS_APPLICATION_THREAD();

    GraphicsDriver::SwapBuffers(frame_->settings.vsyncEnabled);

    frame_.reset();
}

void RendererBackend::RenderQuad_() {
    GetMesh(state_.screenQuad, 0)->GetMeshlet(0)->Render(1, GpuArrayBuffer());
    //_state.screenQuad->GetMeshContainer(0)->mesh->Render(1, GpuArrayBuffer());
}

RendererBackend::ShadowMapCache RendererBackend::CreateShadowMap3DCache_(u32 resolutionX, u32 resolutionY, u32 count, bool vpl, const TextureComponentSize& bits) {
    ShadowMapCache cache;
    
    i32 remaining = i32(count);
    while (remaining > 0 && cache.buffers.size() < MAX_TOTAL_SHADOW_ATLASES) {
        // Determine how many entries will be present in this atlas
        i32 tmp = remaining - MAX_TOTAL_SHADOWS_PER_ATLAS;
        i32 entries = MAX_TOTAL_SHADOWS_PER_ATLAS;
        if (tmp < 0) {
            entries = remaining;
        }
        remaining = tmp;

        const u32 numLayers = entries;

        std::vector<Texture> attachments;
        Texture texture = Texture(TextureConfig{ TextureType::TEXTURE_CUBE_MAP_ARRAY, TextureComponentFormat::DEPTH, bits, TextureComponentType::FLOAT, resolutionX, resolutionY, numLayers, false }, NoTextureData);
        texture.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        texture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        attachments.push_back(texture); 

        if (vpl) {
            texture = Texture(TextureConfig{ TextureType::TEXTURE_CUBE_MAP_ARRAY, TextureComponentFormat::RGB, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, resolutionX, resolutionY, numLayers, false }, NoTextureData);
            texture.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
            texture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

            attachments.push_back(texture);
        }

        auto fbo = FrameBuffer(attachments);

        if (!fbo.Valid()) {
            isValid_ = false;
            return ShadowMapCache();
        }

        cache.buffers.push_back(fbo);
    }

    // Initialize individual entries
    for (i32 index = 0; index < cache.buffers.size(); ++index) {
        const i32 depth = i32(cache.buffers[index].GetDepthStencilAttachment()->Depth());
        for (i32 layer = 0; layer < depth; ++layer) {
            GpuAtlasEntry entry;
            entry.index = index;
            entry.layer = layer;
            cache.freeShadowMaps.push_back(entry);
        }
    }

    return cache;
}

// This handles everything that's in pbr.glsl
void RendererBackend::InitCoreCSMData_(Pipeline * s) {
    const Camera & lightCam = *frame_->vsmc.worldLightCamera;
    // glm::mat4 lightView = lightCam.getViewTransform();
    const glm::vec3 direction = lightCam.GetDirection();

    s->SetVec3("infiniteLightDirection", direction);    
    s->BindTexture("infiniteLightShadowMap", frame_->vsmc.vsm); //*frame_->vsmc.fbo.GetDepthStencilAttachment());
    s->BindTexture("infiniteLightShadowMapNonFiltered", frame_->vsmc.vsm); //*frame_->vsmc.fbo.GetDepthStencilAttachment());
    for (i32 i = 0; i < frame_->vsmc.cascades.size(); ++i) {
        //s->bindTexture("infiniteLightShadowMaps[" + std::to_string(i) + "]", *_state.csms[i].fbo.getDepthStencilAttachment());
        s->SetMat4("cascadeProjViews[" + std::to_string(i) + "]", frame_->vsmc.projectionViewSample);
        // s->setFloat("cascadeSplits[" + std::to_string(i) + "]", _state.cascadeSplits[i]);
    }

    for (i32 i = 0; i < 2; ++i) {
        s->SetVec4("shadowOffset[" + std::to_string(i) + "]", frame_->vsmc.cascadeShadowOffsets[i]);
    }

    s->SetMat4("vsmClipMap0ProjectionView", frame_->vsmc.cascades[0].projectionViewRender);
    s->SetUint("vsmNumCascades", (u32)frame_->vsmc.cascades.size());
    s->SetUint("vsmNumMemoryPools", (u32)frame_->vsmc.vsm.Depth());

    //for (i32 i = 0; i < frame_->vsmc.cascades.size() - 1; ++i) {
    //    // s->setVec3("cascadeScale[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeScale[0]);
    //    // s->setVec3("cascadeOffset[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeOffset[0]);
    //    s->SetVec4("cascadePlanes[" + std::to_string(i) + "]", frame_->vsmc.cascades[i + 1].cascadePlane);
    //}
}

void RendererBackend::InitLights_(Pipeline * s, const VplDistVector_& lights, const usize maxShadowLights) {
    // Set up point lights

    // Make sure everything is set to some sort of default to prevent shader crashes or huge performance drops
    // s->setFloat("lightFarPlanes[0]", 1.0f);
    // s->bindTexture("shadowCubeMaps[0]", _LookupShadowmapTexture(_state.dummyCubeMap));
    // s->setVec3("lightPositions[0]", glm::vec3(0.0f));
    // s->setVec3("lightColors[0]", glm::vec3(0.0f));
    // s->setFloat("lightRadii[0]", 1.0f);
    // s->setBool("lightCastsShadows[0]", false);

    const Camera& c = *frame_->camera;
    glm::vec3 lightColor;
    //i32 lightIndex = 0;
    //i32 shadowLightIndex = 0;
    //for (i32 i = 0; i < lights.size(); ++i) {
    //    LightPtr light = lights[i].first;
    //    PointLight * point = (PointLight *)light.get();
    //    const f64 distance = lights[i].second; //glm::distance(c.getPosition(), light->position);
    //    // Skip lights too far from camera
    //    //if (distance > (2 * light->getRadius())) continue;

    //    // VPLs are handled as part of the global illumination compute pipeline
    //    if (point->IsVirtualLight()) {
    //        continue;
    //    }

    //    if (point->castsShadows() && shadowLightIndex < maxShadowLights) {
    //        s->setFloat("lightFarPlanes[" + std::to_string(shadowLightIndex) + "]", point->getFarPlane());
    //        //_bindShadowMapTexture(s, "shadowCubeMaps[" + std::to_string(shadowLightIndex) + "]", _GetOrAllocateShadowMapHandleForLight(light));
    //        s->bindTexture("shadowCubeMaps[" + std::to_string(shadowLightIndex) + "]", _LookupShadowmapTexture(_GetOrAllocateShadowMapHandleForLight(light)));
    //        s->setBool("lightCastsShadows[" + std::to_string(lightIndex) + "]", true);
    //        ++shadowLightIndex;
    //    }
    //    else {
    //        s->setBool("lightCastsShadows[" + std::to_string(lightIndex) + "]", false);
    //    }

    //    lightColor = point->getBaseColor() * point->getIntensity();
    //    s->setVec3("lightPositions[" + std::to_string(lightIndex) + "]", point->GetPosition());
    //    s->setVec3("lightColors[" + std::to_string(lightIndex) + "]", &lightColor[0]);
    //    s->setFloat("lightRadii[" + std::to_string(lightIndex) + "]", point->getRadius());
    //    //_bindShadowMapTexture(s, "shadowCubeMaps[" + std::to_string(lightIndex) + "]", light->getShadowMapHandle());
    //    ++lightIndex;
    //}

    auto allocator = frame_->perFrameScratchMemory;
    auto gpuLights = std::vector<GpuPointLight, StackBasedPoolAllocator<GpuPointLight>>(StackBasedPoolAllocator<GpuPointLight>(allocator));
    auto gpuShadowCubeMaps = std::vector<GpuAtlasEntry, StackBasedPoolAllocator<GpuAtlasEntry>>(StackBasedPoolAllocator<GpuAtlasEntry>(allocator));
    auto gpuShadowLights = std::vector<GpuPointLight, StackBasedPoolAllocator<GpuPointLight>>(StackBasedPoolAllocator<GpuPointLight>(allocator));
    gpuLights.reserve(lights.size());
    gpuShadowCubeMaps.reserve(maxShadowLights);
    gpuShadowLights.reserve(maxShadowLights);
    for (i32 i = 0; i < lights.size(); ++i) {
        LightPtr light = lights[i].key;
        PointLight* point = (PointLight*)light.get();

        if (point->IsVirtualLight()) {
            continue;
        }

        GpuPointLight gpuLight;
        gpuLight.position = GpuVec(glm::vec4(point->GetPosition(), 1.0f));
        gpuLight.color = GpuVec(glm::vec4(point->GetColor(), 1.0f));
        gpuLight.farPlane = point->GetFarPlane();
        gpuLight.radius = point->GetRadius();

        if (point->CastsShadows() && gpuShadowLights.size() < maxShadowLights) {
            gpuShadowLights.push_back(std::move(gpuLight));
            auto smap = GetOrAllocateShadowMapForLight_(light);
            gpuShadowCubeMaps.push_back(smap);
        }
        else {
            gpuLights.push_back(std::move(gpuLight)); 
        }
    }

    state_.nonShadowCastingPointLights.CopyDataToBuffer(0, sizeof(GpuPointLight) * gpuLights.size(), (const void*)gpuLights.data());
    state_.shadowIndices.CopyDataToBuffer(0, sizeof(GpuAtlasEntry) * gpuShadowCubeMaps.size(), (const void*)gpuShadowCubeMaps.data());
    state_.shadowCastingPointLights.CopyDataToBuffer(0, sizeof(GpuPointLight) * gpuShadowLights.size(), (const void*)gpuShadowLights.data());

    state_.nonShadowCastingPointLights.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, POINT_LIGHT_NON_SHADOW_CASTER_BINDING_POINT);
    state_.shadowIndices.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, POINT_LIGHT_SHADOW_ATLAS_INDICES_BINDING_POINT);
    state_.shadowCastingPointLights.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, POINT_LIGHT_SHADOW_CASTER_BINDING_POINT);
    frame_->vsmc.pageResidencyTable.BindBase(
        GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING);
    s->SetInt("numPagesXY", frame_->vsmc.cascadeResolutionXY / Texture::VirtualPageSizeXY());

    s->SetFloat("ambientIntensity", 0.0001f);
    /*
    if (lightIndex == 0) {
        s->setFloat("ambientIntensity", 0.0001f);
    }
    else {
        s->setFloat("ambientIntensity", 0.0f);
    }
    */

    auto& cache = smapCache_;
    s->SetInt("numLights", i32(gpuLights.size()));
    s->SetInt("numShadowLights", i32(gpuShadowLights.size()));
    s->SetVec3("viewPosition", c.GetPosition());
    s->SetFloat("emissionStrength", frame_->settings.GetEmissionStrength());
    s->SetFloat("minRoughness", frame_->settings.GetMinRoughness());
    s->SetBool("usePerceptualRoughness", frame_->settings.usePerceptualRoughness);
    for (usize i = 0; i < cache.buffers.size(); ++i) {
        s->BindTexture("shadowCubeMaps[" + std::to_string(i) + "]", *cache.buffers[i].GetDepthStencilAttachment());
    }
    const glm::vec3 lightPosition = CalculateAtmosphericLightPosition_();
    s->SetVec3("atmosphericLightPos", lightPosition);

    // Set up world light if enabled
    //glm::mat4 lightView = constructViewMatrix(_state.worldLight.getRotation(), _state.worldLight.getPosition());
    //glm::mat4 lightView = constructViewMatrix(_state.worldLight.getRotation(), glm::vec3(0.0f));
    // Camera lightCam(false);
    // lightCam.setAngle(_state.worldLight.getRotation());
    const Camera & lightCam = *frame_->vsmc.worldLightCamera;
    glm::mat4 lightWorld = lightCam.GetWorldTransform();
    // glm::mat4 lightView = lightCam.getViewTransform();
    glm::vec3 direction = lightCam.GetDirection(); //glm::vec3(-lightWorld[2].x, -lightWorld[2].y, -lightWorld[2].z);
    // STRATUS_LOG << "Light direction: " << direction << std::endl;
    lightColor = frame_->vsmc.worldLight->GetLuminance();
    s->SetVec3("infiniteLightColor", lightColor);
    s->SetFloat("infiniteLightZnear", frame_->vsmc.znear);
    s->SetFloat("infiniteLightZfar", frame_->vsmc.zfar);
    s->SetFloat("infiniteLightDepthBias", frame_->vsmc.worldLight->GetDepthBias());
    s->SetFloat("worldLightAmbientIntensity", frame_->vsmc.worldLight->GetAmbientIntensity());

    InitCoreCSMData_(s);

    // s->setMat4("cascade0ProjView", &_state.csms[0].projectionView[0][0]);
}

GpuAtlasEntry RendererBackend::GetOrAllocateShadowMapForLight_(LightPtr light) {
    auto& cache = GetSmapCacheForLight_(light);
    assert(cache.freeShadowMaps.size() > 0 || cache.cachedLights.size() > 0);

    auto it = cache.lightsToShadowMap.find(light);
    // If not found, look for an existing shadow map
    if (it == cache.lightsToShadowMap.end()) {
        GpuAtlasEntry smap;
        if (cache.freeShadowMaps.size() > 0) {
            smap = cache.freeShadowMaps.front();
            cache.freeShadowMaps.pop_front();
        }

        if (smap.index == -1) {
            // Evict oldest since we could not find an available handle
            smap = EvictOldestLightFromShadowMapCache_(cache);
            // LightPtr oldest = cache.usedShadowMapCache.front();
            // //cache.lruLightCache.pop_front();
            // smap = cache.lightsToShadowMap.find(oldest)->second;
            // EvictLightFromShadowMapCache_(oldest);
        }

        SetLightShadowMap3D_(light, smap);
        AddLightToShadowMapCache_(light);
        return smap;
    }

    // Update the LRU cache
    //AddLightToShadowMapCache_(light);
    return it->second;
}

void RendererBackend::SetLightShadowMap3D_(LightPtr light, GpuAtlasEntry smap) {
    auto& cache = GetSmapCacheForLight_(light);
    cache.lightsToShadowMap.insert(std::make_pair(light, smap));
}

GpuAtlasEntry RendererBackend::EvictLightFromShadowMapCache_(LightPtr light) {
    auto& cache = GetSmapCacheForLight_(light);
    for (auto it = cache.cachedLights.begin(); it != cache.cachedLights.end(); ++it) {
        if (*it == light) {
            cache.cachedLights.erase(it);
            GpuAtlasEntry atlas = cache.lightsToShadowMap.find(light)->second;
            cache.lightsToShadowMap.erase(light);
            return atlas;
        }
    }

    return GpuAtlasEntry();
}

GpuAtlasEntry RendererBackend::EvictOldestLightFromShadowMapCache_(ShadowMapCache& cache) {
    if (cache.cachedLights.size() == 0) {
        throw std::runtime_error("Used shadow map cache is empty");
    }
    auto oldest = cache.cachedLights.front();
    cache.cachedLights.pop_front();
    GpuAtlasEntry atlas = cache.lightsToShadowMap.find(oldest)->second;
    cache.lightsToShadowMap.erase(oldest);
    return atlas;
}

bool RendererBackend::ShadowMapExistsForLight_(LightPtr light) {
    auto& cache = GetSmapCacheForLight_(light);
    return cache.lightsToShadowMap.find(light) != cache.lightsToShadowMap.end();
}

void RendererBackend::AddLightToShadowMapCache_(LightPtr light) {
    auto& cache = GetSmapCacheForLight_(light);
    // First remove the existing light entry if it's already there
    //EvictLightFromShadowMapCache_(light);
    // Push to back so that it is seen as most recently used
    cache.cachedLights.push_back(light);
}

void RendererBackend::RemoveLightFromShadowMapCache_(LightPtr light) {
    if ( !ShadowMapExistsForLight_(light) ) return;

    auto& cache = GetSmapCacheForLight_(light);

    // Deallocate its map
    GpuAtlasEntry smap = cache.lightsToShadowMap.find(light)->second;
    cache.lightsToShadowMap.erase(light);
    cache.freeShadowMaps.push_back(smap);

    // Remove from LRU cache
    EvictLightFromShadowMapCache_(light);
}

RendererBackend::ShadowMapCache& RendererBackend::GetSmapCacheForLight_(LightPtr light) {
    return light->IsVirtualLight() ? vplSmapCache_ : smapCache_;
}
}