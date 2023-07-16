
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

namespace stratus {
bool IsRenderable(const EntityPtr& p) {
    return p->Components().ContainsComponent<RenderComponent>();
}

bool IsLightInteracting(const EntityPtr& p) {
    auto component = p->Components().GetComponent<LightInteractionComponent>();
    return component.status == EntityComponentStatus::COMPONENT_ENABLED;
}

size_t GetMeshCount(const EntityPtr& p) {
    return p->Components().GetComponent<RenderComponent>().component->GetMeshCount();
}

static MeshPtr GetMesh(const EntityPtr& p, const size_t meshIndex) {
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
       //std::cout << "[OpenGL] " << message << std::endl;
    }
}

RendererBackend::RendererBackend(const uint32_t width, const uint32_t height, const std::string& appName) {
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
        Shader{"depth.vs", ShaderType::VERTEX}, 
        Shader{"depth.fs", ShaderType::FRAGMENT}}));
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

    for (int i = 0; i < 6; ++i) {
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
        Shader{"vpl_light_cull.cs", ShaderType::COMPUTE}}));
    state_.shaders.push_back(state_.vplCulling.get());

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
    if (haltonSequence.size() * sizeof(std::pair<float, float>) != haltonSequence.size() * sizeof(GpuHaltonEntry)) {
        throw std::runtime_error("Halton sequence size check failed");
    }
    haltonSequence_ = GpuBuffer((const void *)haltonSequence.data(), sizeof(GpuHaltonEntry) * haltonSequence.size(), GPU_DYNAMIC_DATA);
}

void RendererBackend::InitPointShadowMaps_() {
    // Create the normal point shadow map cache
    smapCache_ = CreateShadowMap3DCache_(state_.shadowCubeMapX, state_.shadowCubeMapY, state_.numRegularShadowMaps, false);

    // Initialize the point light buffers including shadow map texture buffer
    const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
    state_.nonShadowCastingPointLights = GpuBuffer(nullptr, sizeof(GpuPointLight) * state_.maxTotalRegularLightsPerFrame, flags);
    state_.shadowIndices = GpuBuffer(nullptr, sizeof(GpuAtlasEntry) * state_.maxShadowCastingLightsPerFrame, flags);
    state_.shadowCastingPointLights = GpuBuffer(nullptr, sizeof(GpuPointLight) * state_.maxShadowCastingLightsPerFrame, flags);

    STRATUS_LOG << "Size: " << smapCache_.buffers.size() << std::endl;

    // Create the virtual point light shadow map cache
    vplSmapCache_ = CreateShadowMap3DCache_(state_.vpls.vplShadowCubeMapX, state_.vpls.vplShadowCubeMapY, MAX_TOTAL_VPL_SHADOW_MAPS, true);
    state_.vpls.shadowDiffuseIndices = GpuBuffer(nullptr, sizeof(GpuAtlasEntry) * MAX_TOTAL_VPL_SHADOW_MAPS, flags);

    STRATUS_LOG << "Size: " << vplSmapCache_.buffers.size() << std::endl;
}

void RendererBackend::InitializeVplData_() {
    const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
    // +1 since we store the total size of the visibility array at the first index
    std::vector<int> visibleIndicesData(MAX_TOTAL_VPLS_BEFORE_CULLING + 1, 0);
    state_.vpls.vplVisibleIndices = GpuBuffer((const void *)visibleIndicesData.data(), sizeof(int) * visibleIndicesData.size(), flags);
    state_.vpls.vplData = GpuBuffer(nullptr, sizeof(GpuVplData) * MAX_TOTAL_VPLS_BEFORE_CULLING, flags);
    state_.vpls.vplUpdatedData = GpuBuffer(nullptr, sizeof(GpuVplData) * MAX_TOTAL_VPLS_PER_FRAME, flags);
    //state_.vpls.vplNumVisible = GpuBuffer(nullptr, sizeof(int), flags);
}

void RendererBackend::ValidateAllShaders_() {
    isValid_ = true;
    for (Pipeline * p : state_.shaders) {
        isValid_ = isValid_ && p->IsValid();
    }
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
    const uint32_t cascadeResolutionXY = frame_->csc.cascadeResolutionXY;
    const uint32_t numCascades = frame_->csc.cascades.size();
    if (frame_->csc.regenerateFbo || !frame_->csc.fbo.Valid()) {
        // Create the depth buffer
        // @see https://stackoverflow.com/questions/22419682/glsl-sampler2dshadow-and-shadow2d-clarificationssss
        Texture tex(TextureConfig{ TextureType::TEXTURE_2D_ARRAY, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, cascadeResolutionXY, cascadeResolutionXY, numCascades, false }, NoTextureData);
        tex.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        tex.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
        // We need to set this when using sampler2DShadow in the GLSL shader
        tex.SetTextureCompare(TextureCompareMode::COMPARE_REF_TO_TEXTURE, TextureCompareFunc::LEQUAL);

        // Create the frame buffer
        frame_->csc.fbo = FrameBuffer({ tex });
    }
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

    for (GBuffer* gbptr : buffers) {
        GBuffer& buffer = *gbptr;

        // Position buffer
        //buffer.position = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, NoTextureData);
        //buffer.position.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

        // Normal buffer
        buffer.normals = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.normals.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        buffer.normals.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create the color buffer - notice that is uses higher
        // than normal precision. This allows us to write color values
        // greater than 1.0 to support things like HDR.
        buffer.albedo = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.albedo.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.albedo.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Base reflectivity buffer
        buffer.baseReflectivity = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.baseReflectivity.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.baseReflectivity.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Roughness-Metallic-Ambient buffer
        buffer.roughnessMetallicAmbient = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.roughnessMetallicAmbient.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.roughnessMetallicAmbient.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create the Structure buffer which contains rgba where r=partial x-derivative of camera-space depth, g=partial y-derivative of camera-space depth, b=16 bits of depth, a=final 16 bits of depth (b+a=32 bits=depth)
        buffer.structure = Texture(TextureConfig{ TextureType::TEXTURE_RECTANGLE, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.structure.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.structure.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create velocity buffer
        // TODO: Determine best bit depth - apparently we tend to need higher precision since these values can be consistently super small
        buffer.velocity = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RG, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.velocity.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.velocity.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Holds mesh ids
        buffer.id = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RED, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.id.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.id.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create the depth buffer
        buffer.depth = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.depth.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.depth.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create the frame buffer with all its texture attachments
        //buffer.fbo = FrameBuffer({buffer.position, buffer.normals, buffer.albedo, buffer.baseReflectivity, buffer.roughnessMetallicAmbient, buffer.structure, buffer.depth});
        buffer.fbo = FrameBuffer({ buffer.normals, buffer.albedo, buffer.baseReflectivity, buffer.roughnessMetallicAmbient, buffer.structure, buffer.velocity, buffer.id, buffer.depth });
        if (!buffer.fbo.Valid()) {
            isValid_ = false;
            return;
        }
    }
}

void RendererBackend::UpdateWindowDimensions_() {
    if ( !frame_->viewportDirty ) return;
    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    // Set up VPL tile data
    const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
    const int totalTiles = (frame_->viewportWidth / state_.vpls.tileXDivisor) * (frame_->viewportHeight / state_.vpls.tileYDivisor);
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
    state_.lightingHighBrightnessBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.lightingHighBrightnessBuffer.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    state_.lightingHighBrightnessBuffer.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    // Create the depth buffer
    state_.lightingDepthBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.lightingDepthBuffer.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);

    // Attach the textures to the FBO
    state_.lightingFbo = FrameBuffer({state_.lightingColorBuffer, state_.lightingHighBrightnessBuffer, state_.lightingDepthBuffer});
    if (!state_.lightingFbo.Valid()) {
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
    uint32_t currWidth = frame_->viewportWidth;
    uint32_t currHeight = frame_->viewportHeight;
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
        for (int i = 0; i < 2; ++i) {
            FrameBuffer& blurFbo = dualBlurFbos[i].fbo;
            Texture tex = Texture(color.GetConfig(), NoTextureData);
            tex.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
            tex.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
            blurFbo = FrameBuffer({tex});
            state_.gaussianBuffers.push_back(dualBlurFbos[i]);
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> sizes;
    for (int i = state_.numDownsampleIterations - 2; i >= 0; --i) {
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
    atmosphericTexture.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
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
    gammaTonemap.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
    gammaTonemap.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.gammaTonemapFbo.fbo = FrameBuffer({ gammaTonemap });
    if (!state_.gammaTonemapFbo.fbo.Valid()) {
        isValid_ = false;
        STRATUS_ERROR << "Unable to initialize gamma tonemap buffer" << std::endl;
        return;
    }

    // Create the FXAA buffers
    Texture fxaa = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    fxaa.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
    fxaa.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.fxaaFbo1.fbo = FrameBuffer({fxaa});
    if (!state_.fxaaFbo1.fbo.Valid()) {
        isValid_ = false;
        STRATUS_ERROR << "Unable to initialize fxaa luminance buffer" << std::endl;
        return;
    }
    state_.postFxBuffers.push_back(state_.fxaaFbo1);

    fxaa = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    fxaa.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
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
    taa.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
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
        if (frame_->csc.fbo.Valid()) {
            frame_->csc.fbo.Clear(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            //const int index = Engine::Instance()->FrameCount() % 4;
            //_frame->csc.fbo.ClearDepthStencilLayer(index);
        }

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
    std::vector<float> ks(16);
    std::iota(ks.begin(), ks.end(), 0.0f);
    std::shuffle(ks.begin(), ks.end(), std::default_random_engine{});

    // Create the data for the 4x4 lookup table
    float table[16 * 3]; // RGB
    for (size_t i = 0; i < ks.size(); ++i) {
        const float k = ks[i];
        const Radians r(2.0f * float(STRATUS_PI) * k / 16.0f);
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
    std::uniform_real_distribution<float> real(0.0f, 1.0f);

    // Create the 64x64 noise texture
    const size_t size = 64 * 64;
    std::vector<float> table(size);
    for (size_t i = 0; i < size; ++i) {
        table[i] = real(re);
    }

    const void* ptr = (const void *)table.data();
    state_.atmosphericNoiseTexture = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, 64, 64, 0, false}, TextureArrayData{ptr});
    state_.atmosphericNoiseTexture.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
    state_.atmosphericNoiseTexture.SetCoordinateWrapping(TextureCoordinateWrapping::REPEAT);
}

void RendererBackend::ClearRemovedLightData_() {
    int lightsCleared = 0;
    for (auto ptr : frame_->lightsToRemove) {
        RemoveLightFromShadowMapCache_(ptr);
        ++lightsCleared;
    }

    if (lightsCleared > 0) STRATUS_LOG << "Cleared " << lightsCleared << " lights this frame" << std::endl;
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
    glDepthFunc(GL_LEQUAL);
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
static std::vector<glm::mat4> GenerateLightViewTransforms(const glm::vec3 & lightPos) {
    return std::vector<glm::mat4>{
        //          pos       pos + dir                                  up
        glm::lookAt(lightPos, lightPos + glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };
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

void RendererBackend::RenderBoundingBoxes_(GpuCommandBuffer2Ptr& buffer) {
    if (buffer->NumDrawCommands() == 0) return;

    SetCullState(RenderFaceCulling::CULLING_NONE);

    buffer->BindModelTransformBuffer(13);
    buffer->BindAabbBuffer(14);

    BindShader_(state_.aabbDraw.get());

    // _state.aabbDraw->setMat4("projection", _frame->projection);
    // _state.aabbDraw->setMat4("view", _frame->camera->getViewTransform());
    state_.aabbDraw->SetMat4("projectionView", frame_->projectionView);

    for (int i = 0; i < buffer->NumDrawCommands(); ++i) {
        state_.aabbDraw->SetInt("modelIndex", i);
        glDrawArrays(GL_LINES, 0, 24);
    }

    UnbindShader_();
}

void RendererBackend::RenderBoundingBoxes_(std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>& map) {
    for (auto& entry : map) {
        RenderBoundingBoxes_(entry.second);
    }
}

void RendererBackend::RenderImmediate_(const RenderFaceCulling cull, GpuCommandBuffer2Ptr& buffer, const CommandBufferSelectionFunction& select) {
    if (buffer->NumDrawCommands() == 0) return;

    frame_->materialInfo->GetMaterialBuffer().BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 30);
    buffer->BindMaterialIndicesBuffer(31);
    buffer->BindModelTransformBuffer(13);
    buffer->BindPrevFrameModelTransformBuffer(14);
    select(buffer).Bind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);

    SetCullState(cull);

    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, (const void *)0, (GLsizei)buffer->NumDrawCommands(), (GLsizei)0);

    select(buffer).Unbind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);
}

void RendererBackend::Render_(Pipeline& s, const RenderFaceCulling cull, GpuCommandBuffer2Ptr& buffer, const CommandBufferSelectionFunction& select, bool isLightInteracting, bool removeViewTranslation) {
    if (buffer->NumDrawCommands() == 0) return;

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

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

    s.SetMat4("projection", projection);
    s.SetMat4("view", view);
    s.SetMat4("projectionView", projectionView);
    s.SetMat4("prevProjectionView", frame_->prevProjectionView);

    s.SetInt("viewWidth", frame_->viewportWidth);
    s.SetInt("viewHeight", frame_->viewportHeight);
    //s->setMat4("projectionView", &projection[0][0]);
    //s->setMat4("view", &view[0][0]);

    RenderImmediate_(cull, buffer, select);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
}

void RendererBackend::Render_(Pipeline& s, std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>& map, const CommandBufferSelectionFunction& select, bool isLightInteracting, bool removeViewTranslation) {
    for (auto& entry : map) {
        Render_(s, entry.first, entry.second, select, isLightInteracting, removeViewTranslation);
    }
}

void RendererBackend::RenderImmediate_(std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>& map, const CommandBufferSelectionFunction& select, const bool reverseCullFace) {
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
        RenderImmediate_(cull, entry.second, select);
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

        GetMesh(state_.skyboxCube, 0)->Render(1, GpuArrayBuffer());
        //_state.skyboxCube->GetMeshContainer(0)->mesh->Render(1, GpuArrayBuffer());
    }

    glDepthMask(GL_TRUE);
}

void RendererBackend::RenderSkybox_() {
    const glm::mat4& projection = frame_->projection;
    const glm::mat4 view = glm::mat4(glm::mat3(frame_->camera->GetViewTransform()));
    const glm::mat4 projectionView = projection * view;

    BindShader_(state_.skybox.get());
    RenderSkybox_(state_.skybox.get(), projectionView);
    UnbindShader_();
}

void RendererBackend::RenderCSMDepth_() {
    if (frame_->csc.cascades.size() > state_.csmDepth.size()) {
        throw std::runtime_error("Max cascades exceeded (> 6)");
    }

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    // Allows GPU to perform angle-dependent depth offset to help reduce artifacts such as shadow acne
    // See https://blogs.igalia.com/itoral/2017/10/02/working-with-lights-and-shadows-part-iii-rendering-the-shadows/
    // See https://community.khronos.org/t/can-anyone-explain-glpolygonoffset/35382
    glEnable(GL_POLYGON_OFFSET_FILL);
    // See https://paroj.github.io/gltut/Positioning/Tut05%20Depth%20Clamping.html
    glEnable(GL_DEPTH_CLAMP);
    // First value is conditional on slope
    // Second value is a constant unconditional offset
    //glPolygonOffset(3.0f, 0.0f);
    glPolygonOffset(2.0f, 0.0f);
    //glPolygonOffset(5.0f, 0.0f);
    //glBlendFunc(GL_ONE, GL_ONE);
    // glDisable(GL_CULL_FACE);

    frame_->csc.fbo.Bind();
    const Texture * depth = frame_->csc.fbo.GetDepthStencilAttachment();
    if (!depth) {
        throw std::runtime_error("Critical error: depth attachment not present");
    }
    glViewport(0, 0, depth->Width(), depth->Height());

    for (size_t cascade = 0; cascade < frame_->csc.cascades.size(); ++cascade) {
        Pipeline * shader = frame_->csc.worldLight->GetAlphaTest() ?
            state_.csmDepthRunAlphaTest[cascade].get() :
            state_.csmDepth[cascade].get();

        BindShader_(shader);

        shader->SetVec3("lightDir", &frame_->csc.worldLightCamera->GetDirection()[0]);
        shader->SetFloat("nearClipPlane", frame_->znear);

        // Set up each individual view-projection matrix
        // for (int i = 0; i < _frame->csc.cascades.size(); ++i) {
        //     auto& csm = _frame->csc.cascades[i];
        //     _state.csmDepth->setMat4("shadowMatrices[" + std::to_string(i) + "]", &csm.projectionViewRender[0][0]);
        // }

        // Select face (one per frame)
        //const int face = Engine::Instance()->FrameCount() % 4;
        //_state.csmDepth->setInt("face", face);

        // Render everything
        // See https://www.gamedev.net/forums/topic/695063-is-there-a-quick-way-to-fix-peter-panning-shadows-detaching-from-objects/5370603/
        // for the tip about enabling reverse culling for directional shadow maps to reduce peter panning
        auto& csm = frame_->csc.cascades[cascade];
        shader->SetMat4("shadowMatrix", csm.projectionViewRender);
        // const size_t lod = cascade * 2 + 1;
        const size_t lod = frame_->drawCommands->NumLods() - 1;
        if (cascade < 2) {
            const CommandBufferSelectionFunction select = [](GpuCommandBuffer2Ptr& b) {
                return b->GetSelectedLodDrawCommandsBuffer();
            };
            RenderImmediate_(frame_->drawCommands->dynamicPbrMeshes, select, true);
            RenderImmediate_(frame_->drawCommands->staticPbrMeshes, select, true);
        }
        else {
            const CommandBufferSelectionFunction select = [lod](GpuCommandBuffer2Ptr& b) {
                return b->GetIndirectDrawCommandsBuffer(lod);
            };
            RenderImmediate_(frame_->drawCommands->dynamicPbrMeshes, select, true);
            RenderImmediate_(frame_->drawCommands->staticPbrMeshes, select, true);
        }

        // RenderImmediate_(csm.visibleDynamicPbrMeshes);
        // RenderImmediate_(csm.visibleStaticPbrMeshes);

        UnbindShader_();
    }
    
    frame_->csc.fbo.Unbind();

    glDisable(GL_POLYGON_OFFSET_FILL);
    glDisable(GL_DEPTH_CLAMP);
}

void RendererBackend::RenderSsaoOcclude_() {
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    // Aspect ratio
    const float ar        = float(frame_->viewportWidth) / float(frame_->viewportHeight);
    // Distance to the view projection plane
    const float g         = 1.0f / glm::tan(frame_->fovy.value() / 2.0f);
    const float w         = frame_->viewportWidth;
    // Gets fed into sigma value
    const float intensity = 5.0f;

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
    if (!frame_->csc.worldLight->GetEnabled()) return;

    constexpr float preventDivByZero = std::numeric_limits<float>::epsilon();

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    auto re = std::default_random_engine{};
    const float n = frame_->csc.worldLight->GetAtmosphericNumSamplesPerPixel();
    // On the range [0.0, 1/n)
    std::uniform_real_distribution<float> real(0.0f, 1.0f / n);
    const glm::vec2 noiseShift(real(re), real(re));
    const float dmin     = frame_->znear;
    const float dmax     = frame_->csc.cascades[frame_->csc.cascades.size() - 1].cascadeEnds;
    const float lambda   = frame_->csc.worldLight->GetAtmosphericParticleDensity();
    // cbrt = cube root
    const float cubeR    = std::cbrt(frame_->csc.worldLight->GetAtmosphericScatterControl());
    const float g        = (1.0f - cubeR) / (1.0f + cubeR + preventDivByZero);
    // aspect ratio
    const float ar       = float(frame_->viewportWidth) / float(frame_->viewportHeight);
    // g in frustum parameters
    const float projDist = 1.0f / glm::tan(frame_->fovy.value() / 2.0f);
    const glm::vec3 frustumParams(ar / projDist, 1.0f / projDist, dmin);
    const glm::mat4 shadowMatrix = frame_->csc.cascades[0].projectionViewSample * frame_->camera->GetWorldTransform();
    const glm::vec3 anisotropyConstants(1 - g, 1 + g * g, 2 * g);
    const glm::vec4 shadowSpaceCameraPos = frame_->csc.cascades[0].projectionViewSample * glm::vec4(frame_->camera->GetPosition(), 1.0f);
    const glm::vec3 normalizedCameraLightDirection = frame_->csc.worldLightDirectionCameraSpace;

    const auto timePoint = std::chrono::high_resolution_clock::now();
    const float milliseconds = float(std::chrono::time_point_cast<std::chrono::milliseconds>(timePoint).time_since_epoch().count());

    BindShader_(state_.atmospheric.get());
    state_.atmosphericFbo.Bind();
    state_.atmospheric->SetVec3("frustumParams", frustumParams);
    state_.atmospheric->SetMat4("shadowMatrix", shadowMatrix);
    state_.atmospheric->BindTexture("structureBuffer", state_.currentFrame.structure);
    state_.atmospheric->BindTexture("infiniteLightShadowMap", *frame_->csc.fbo.GetDepthStencilAttachment());
    state_.atmospheric->SetFloat("time", milliseconds);
    
    // Set up cascade data
    for (int i = 0; i < 4; ++i) {
        const auto& cascade = frame_->csc.cascades[i];
        const std::string si = "[" + std::to_string(i) + "]";
        state_.atmospheric->SetFloat("maxCascadeDepth" + si, cascade.cascadeEnds);
        if (i > 0) {
            const std::string sim1 = "[" + std::to_string(i - 1) + "]";
            state_.atmospheric->SetMat4("cascade0ToCascadeK" + sim1, cascade.sampleCascade0ToCurrent);
        }
    }

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
    state_.atmospheric->SetFloat("windowWidth", float(colorTex.Width()));
    state_.atmospheric->SetFloat("windowHeight", float(colorTex.Height()));

    glViewport(0, 0, colorTex.Width(), colorTex.Height());
    RenderQuad_();
    state_.atmosphericFbo.Unbind();
    UnbindShader_();

    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

void RendererBackend::InitVplFrameData_(const std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer) {
    std::vector<GpuVplData> vplData(perVPLDistToViewer.size());
    for (size_t i = 0; i < perVPLDistToViewer.size(); ++i) {
        VirtualPointLight * point = (VirtualPointLight *)perVPLDistToViewer[i].first.get();
        GpuVplData& data = vplData[i];
        data.position = GpuVec(glm::vec4(point->GetPosition(), 1.0f));
        data.farPlane = point->GetFarPlane();
        data.radius = point->GetRadius();
        data.intensity = point->GetIntensity();
    }
    state_.vpls.vplData.CopyDataToBuffer(0, sizeof(GpuVplData) * vplData.size(), (const void *)vplData.data());
}

void RendererBackend::UpdatePointLights_(std::vector<std::pair<LightPtr, double>>& perLightDistToViewer, 
                                         std::vector<std::pair<LightPtr, double>>& perLightShadowCastingDistToViewer,
                                         std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer,
                                         std::vector<int>& visibleVplIndices) {
    const Camera& c = *frame_->camera;

    const bool worldLightEnabled = frame_->csc.worldLight->GetEnabled();

    perLightDistToViewer.clear();
    perLightShadowCastingDistToViewer.clear();
    perVPLDistToViewer.clear();
    visibleVplIndices.clear();

    perLightDistToViewer.reserve(state_.maxTotalRegularLightsPerFrame);
    perLightShadowCastingDistToViewer.reserve(state_.maxShadowCastingLightsPerFrame);
    if (worldLightEnabled) {
        perVPLDistToViewer.reserve(MAX_TOTAL_VPLS_BEFORE_CULLING);
    }

    // Init per light instance data
    for (auto& light : frame_->lights) {
        const double distance = glm::distance(c.GetPosition(), light->GetPosition());
        if (light->IsVirtualLight()) {
            if (worldLightEnabled && distance <= MAX_VPL_DISTANCE_TO_VIEWER) {
                perVPLDistToViewer.push_back(std::make_pair(light, distance));
            }
        }
        else {
            perLightDistToViewer.push_back(std::make_pair(light, distance));
        }

        if ( !light->IsVirtualLight() && light->CastsShadows() ) {
            perLightShadowCastingDistToViewer.push_back(std::make_pair(light, distance));
        }
    }

    // Sort lights based on distance to viewer
    const auto comparison = [](const std::pair<LightPtr, double> & a, const std::pair<LightPtr, double> & b) {
        return a.second < b.second;
    };

    const bool regularLightsMaxExceeded = perLightDistToViewer.size() > state_.maxTotalRegularLightsPerFrame;
    const bool regularShadowLightsMaxExceeded = perLightShadowCastingDistToViewer.size() > state_.maxShadowCastingLightsPerFrame;
    
    if (regularLightsMaxExceeded || regularShadowLightsMaxExceeded) {
        std::sort(perLightDistToViewer.begin(), perLightDistToViewer.end(), comparison);
        std::sort(perLightShadowCastingDistToViewer.begin(), perLightShadowCastingDistToViewer.end(), comparison);
    }

    // Remove lights exceeding the absolute maximum
    if (regularLightsMaxExceeded) {
        //std::sort(perLightDistToViewer.begin(), perLightDistToViewer.end(), comparison);
        perLightDistToViewer.resize(state_.maxTotalRegularLightsPerFrame);
    }

    // Remove shadow-casting lights that exceed our max count
    if (regularShadowLightsMaxExceeded) {
        //std::sort(perLightShadowCastingDistToViewer.begin(), perLightShadowCastingDistToViewer.end(), comparison);
        perLightShadowCastingDistToViewer.resize(state_.maxShadowCastingLightsPerFrame);
    }

    // Remove vpls exceeding absolute maximum
    if (worldLightEnabled) {
        //std::sort(perVPLDistToViewer.begin(), perVPLDistToViewer.end(), comparison);
        if (perVPLDistToViewer.size() > MAX_TOTAL_VPLS_BEFORE_CULLING) {
            std::sort(perVPLDistToViewer.begin(), perVPLDistToViewer.end(), comparison);
            perVPLDistToViewer.resize(MAX_TOTAL_VPLS_BEFORE_CULLING);
        }

        InitVplFrameData_(perVPLDistToViewer);
        PerformVirtualPointLightCullingStage1_(perVPLDistToViewer, visibleVplIndices);
    }

    // Check if any need to have a new shadow map pulled from the cache
    for (const auto&[light, _] : perLightShadowCastingDistToViewer) {
        if (!ShadowMapExistsForLight_(light)) {
            frame_->lightsToUpdate.PushBack(light);
        }
    }

    for (size_t i = 0; i < visibleVplIndices.size(); ++i) {
        const int index = visibleVplIndices[i];
        auto light = perVPLDistToViewer[index].first;
        if (!ShadowMapExistsForLight_(light)) {
            frame_->lightsToUpdate.PushBack(light);
        }
    }

    // Set blend func just for shadow pass
    // glBlendFunc(GL_ONE, GL_ONE);
    glEnable(GL_DEPTH_TEST);
    // Perform the shadow volume pre-pass
    for (int shadowUpdates = 0; shadowUpdates < state_.maxShadowUpdatesPerFrame && frame_->lightsToUpdate.Size() > 0; ++shadowUpdates) {
        auto light = frame_->lightsToUpdate.PopFront();
        // Ideally this won't be needed but just in case
        if ( !light->CastsShadows() ) continue;
        //const double distance = perLightShadowCastingDistToViewer.find(light)->second;
    
        // TODO: Make this work with spotlights
        //PointLightPtr point = (PointLightPtr)light;
        PointLight * point = (PointLight *)light.get();
        auto& cache = GetSmapCacheForLight_(light);
        GpuAtlasEntry smap = GetOrAllocateShadowMapForLight_(light);

        const auto cubeMapWidth = cache.buffers[smap.index].GetDepthStencilAttachment()->Width();
        const auto cubeMapHeight = cache.buffers[smap.index].GetDepthStencilAttachment()->Height();
        const glm::mat4 lightPerspective = glm::perspective<float>(glm::radians(90.0f), float(cubeMapWidth) / float(cubeMapHeight), point->GetNearPlane(), point->GetFarPlane());

        // glBindFramebuffer(GL_FRAMEBUFFER, smap.frameBuffer);
        if (cache.buffers[smap.index].GetColorAttachments().size() > 0) {
            cache.buffers[smap.index].GetColorAttachments()[0].ClearLayer(0, smap.layer, nullptr);
        }
        float depthClear = 1.0f;
        cache.buffers[smap.index].GetDepthStencilAttachment()->ClearLayer(0, smap.layer, &depthClear);

        cache.buffers[smap.index].Bind();
        glViewport(0, 0, cubeMapWidth, cubeMapHeight);
        // Current pass only cares about depth buffer
        // glClear(GL_DEPTH_BUFFER_BIT);

        Pipeline * shader = light->IsVirtualLight() ? state_.vplShadows.get() : state_.shadows.get();
        auto transforms = GenerateLightViewTransforms(point->GetPosition());
        for (size_t i = 0; i < transforms.size(); ++i) {
            const glm::mat4 projectionView = lightPerspective * transforms[i];

            BindShader_(shader);
            // * 6 since each cube map is accessed by a layer-face which is divisible by 6
            shader->SetInt("layer", int(smap.layer * 6 + i));
            shader->SetMat4("shadowMatrix", projectionView);
            shader->SetVec3("lightPos", light->GetPosition());
            shader->SetFloat("farPlane", point->GetFarPlane());

            if (point->IsVirtualLight()) {
                // Use lower LOD
                const size_t lod = frame_->drawCommands->NumLods() - 1;
                const CommandBufferSelectionFunction select = [lod](GpuCommandBuffer2Ptr& b) {
                    return b->GetIndirectDrawCommandsBuffer(lod);
                };
                RenderImmediate_(frame_->drawCommands->staticPbrMeshes, select, false);
                //RenderImmediate_(frame_->instancedDynamicPbrMeshes[frame_->instancedDynamicPbrMeshes.size() - 1]);

                const glm::mat4 projectionViewNoTranslate = lightPerspective * glm::mat4(glm::mat3(transforms[i]));

                BindShader_(state_.skyboxLayered.get());
                state_.skyboxLayered->SetInt("layer", int(smap.layer * 6 + i));

                auto tmp = frame_->settings.GetSkyboxIntensity();
                if (tmp > 1.0f) {
                    frame_->settings.SetSkyboxIntensity(1.0f);
                }
                
                RenderSkybox_(state_.skyboxLayered.get(), projectionViewNoTranslate);

                if (tmp > 1.0f) {
                    frame_->settings.SetSkyboxIntensity(tmp);
                }
            }
            else {
                const CommandBufferSelectionFunction select = [](GpuCommandBuffer2Ptr& b) {
                    return b->GetIndirectDrawCommandsBuffer(0);
                };
                RenderImmediate_(frame_->drawCommands->staticPbrMeshes, select, false);
                if ( !point->IsStaticLight() ) RenderImmediate_(frame_->drawCommands->dynamicPbrMeshes, select, false);
            }
            UnbindShader_();
        }

        // Unbind
        cache.buffers[smap.index].Unbind();
    }
}

void RendererBackend::PerformVirtualPointLightCullingStage1_(
    std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer,
    std::vector<int>& visibleVplIndices) {

    if (perVPLDistToViewer.size() == 0) return;

    state_.vplCulling->Bind();

    const Camera & lightCam = *frame_->csc.worldLightCamera;
    // glm::mat4 lightView = lightCam.getViewTransform();
    const glm::vec3 direction = lightCam.GetDirection();

    state_.vplCulling->SetVec3("infiniteLightDirection", direction);
    state_.vplCulling->SetInt("totalNumLights", perVPLDistToViewer.size());

    // Set up # visible atomic counter
    int numVisible = 0;
    //state_.vpls.vplNumVisible.CopyDataToBuffer(0, sizeof(int), (const void *)&numVisible);
    //state_.vpls.vplNumVisible.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);

    // Bind light data and visibility indices
    state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 3);
    state_.vpls.vplData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);
    state_.vpls.vplUpdatedData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 4);

    InitCoreCSMData_(state_.vplCulling.get());
    state_.vplCulling->DispatchCompute(1, 1, 1);
    state_.vplCulling->SynchronizeCompute();

    state_.vplCulling->Unbind();

    // int totalVisible = *(int *)state_.vpls.vplNumVisible.MapMemory();
    // state_.vpls.vplNumVisible.UnmapMemory();

    // if (totalVisible > 0) {
    //     // These should still be sorted since the GPU compute shader doesn't reorder them
    //     int* indices = (int*)state_.vpls.vplVisibleIndices.MapMemory();
    //     visibleVplIndices.resize(totalVisible);

    //     for (int i = 0; i < totalVisible; ++i) {
    //         visibleVplIndices[i] = indices[i];
    //     }

    //     state_.vpls.vplVisibleIndices.UnmapMemory();
    // }
    // else {
    //     perVPLDistToViewer.clear();
    //     visibleVplIndices.clear();
    // }

    //STRATUS_LOG << totalVisible << std::endl;

    //if (totalVisible > MAX_TOTAL_VPLS_PER_FRAME) {
    //    visibleVplIndices.resize(MAX_TOTAL_VPLS_PER_FRAME);
    //    totalVisible = MAX_TOTAL_VPLS_PER_FRAME;
    //    // visibleVplIndices.clear();

    //    // // We want at least 64 lights close to the viewer
    //    // for (int i = 0; i < 64; ++i) {
    //    //     visibleVplIndices.push_back(indices[i]);
    //    // }

    //    // const int rest = totalVisible - 64;
    //    // const int step = std::max<int>(rest / (MAX_TOTAL_VPLS_PER_FRAME - 64), 1);
    //    // for (int i = 64; i < totalVisible; i += step) {
    //    //     visibleVplIndices.push_back(indices[i]);
    //    // }

    //    // totalVisible = int(visibleVplIndices.size());
    //    // // Make sure we didn't go over because of step size
    //    // if (visibleVplIndices.size() > MAX_TOTAL_VPLS_PER_FRAME) {
    //    //     visibleVplIndices.resize(MAX_TOTAL_VPLS_PER_FRAME);
    //    //     totalVisible = MAX_TOTAL_VPLS_PER_FRAME;
    //    // }

    //    state_.vpls.vplNumVisible.CopyDataToBuffer(0, sizeof(int), (const void *)&totalVisible);
    //    state_.vpls.vplVisibleIndices.CopyDataToBuffer(0, sizeof(int) * totalVisible, (const void *)visibleVplIndices.data());
    //}
}

// void RendererBackend::PerformVirtualPointLightCullingStage2_(
//     const std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer,
//     const std::vector<int>& visibleVplIndices) {
void RendererBackend::PerformVirtualPointLightCullingStage2_(
    const std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer) {

    // int totalVisible = *(int *)state_.vpls.vplNumVisible.MapMemory();
    // state_.vpls.vplNumVisible.UnmapMemory();

    //if (perVPLDistToViewer.size() == 0 || visibleVplIndices.size() == 0) return;
    if (perVPLDistToViewer.size() == 0) return;

    int* visibleVplIndices = (int*)state_.vpls.vplVisibleIndices.MapMemory(GPU_MAP_READ);
    const int totalVisible = visibleVplIndices[0];
    visibleVplIndices += 1;
    
    if (totalVisible == 0) {
        state_.vpls.vplVisibleIndices.UnmapMemory();
        return;
    }

    // Pack data into system memory
    std::vector<GpuTextureHandle> diffuseHandles;
    diffuseHandles.reserve(totalVisible);
    std::vector<GpuAtlasEntry> shadowDiffuseIndices;
    shadowDiffuseIndices.reserve(totalVisible);
    for (size_t i = 0; i < totalVisible; ++i) {
        const int index = visibleVplIndices[i];
        VirtualPointLight * point = (VirtualPointLight *)perVPLDistToViewer[index].first.get();
        auto smap = GetOrAllocateShadowMapForLight_(perVPLDistToViewer[index].first);
        shadowDiffuseIndices.push_back(smap);
    }

    state_.vpls.vplVisibleIndices.UnmapMemory();
    visibleVplIndices = nullptr;

    // Move data to GPU memory
    state_.vpls.shadowDiffuseIndices.CopyDataToBuffer(0, sizeof(GpuAtlasEntry) * shadowDiffuseIndices.size(), (const void *)shadowDiffuseIndices.data());

    const Camera & lightCam = *frame_->csc.worldLightCamera;
    // glm::mat4 lightView = lightCam.getViewTransform();
    const glm::vec3 direction = lightCam.GetDirection();

    state_.vplColoring->Bind();

    // Bind inputs
    auto& cache = vplSmapCache_;
    state_.vplColoring->SetVec3("infiniteLightDirection", direction);
    state_.vplColoring->SetVec3("infiniteLightColor", frame_->csc.worldLight->GetLuminance());
    for (size_t i = 0; i < cache.buffers.size(); ++i) {
        state_.vplColoring->BindTexture("diffuseCubeMaps[" + std::to_string(i) + "]", cache.buffers[i].GetColorAttachments()[0]);
    }

    state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);
    //state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 3);
    state_.vpls.shadowDiffuseIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 4);

    // Bind outputs
    state_.vpls.vplUpdatedData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);

    // Dispatch and synchronize
    state_.vplColoring->DispatchCompute(1, 1, 1);
    state_.vplColoring->SynchronizeCompute();

    state_.vplColoring->Unbind();

    // Now perform culling per tile since we now know which lights are active
    // state_.vplTileDeferredCullingStage1->Bind();

    // // Bind inputs
    // //_state.vplTileDeferredCullingStage1->bindTexture("gPosition", _state.buffer.position);
    // state_.vplTileDeferredCullingStage1->SetMat4("invProjectionView", frame_->invProjectionView);
    // state_.vplTileDeferredCullingStage1->BindTexture("gDepth", state_.currentFrame.depth);
    // state_.vplTileDeferredCullingStage1->BindTexture("gNormal", state_.currentFrame.normals);
    // // _state.vplTileDeferredCulling->setInt("viewportWidth", _frame->viewportWidth);
    // // _state.vplTileDeferredCulling->setInt("viewportHeight", _frame->viewportHeight);

    // state_.vpls.vplData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);
    // state_.vpls.vplShadowMaps.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 11);

    // // Bind outputs
    // state_.vpls.vplStage1Results.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);

    // // Dispatch and synchronize
    // state_.vplTileDeferredCullingStage1->DispatchCompute(
    //     (unsigned int)frame_->viewportWidth  / state_.vpls.tileXDivisor,
    //     (unsigned int)frame_->viewportHeight / state_.vpls.tileYDivisor,
    //     1
    // );
    // state_.vplTileDeferredCullingStage1->SynchronizeCompute();

    // state_.vplTileDeferredCullingStage1->Unbind();

    // // Perform stage 2 of the tiled deferred culling
    // state_.vplTileDeferredCullingStage2->Bind();

    // // Bind inputs
    // state_.vplTileDeferredCullingStage2->SetVec3("viewPosition", frame_->camera->GetPosition());

    // state_.vpls.vplData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);
    // state_.vpls.vplStage1Results.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);
    // state_.vpls.vplNumVisible.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 2);
    // state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 3);
    // state_.vpls.vplShadowMaps.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 11);

    // // Bind outputs
    // state_.vpls.vplVisiblePerTile.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 4);
    
    // // Dispatch and synchronize
    // state_.vplTileDeferredCullingStage2->DispatchCompute(
    //     (unsigned int)frame_->viewportWidth  / (state_.vpls.tileXDivisor * 32),
    //     (unsigned int)frame_->viewportHeight / (state_.vpls.tileYDivisor * 2),
    //     1
    // );
    // state_.vplTileDeferredCullingStage2->SynchronizeCompute();

    // state_.vplTileDeferredCullingStage2->Unbind();

    // int * tv = (int *)_state.vpls.vplNumVisible.MapMemory();
    // GpuVplStage2PerTileOutputs * tiles = (GpuVplStage2PerTileOutputs *)_state.vpls.vplVisiblePerTile.MapMemory();
    // GpuVplData * vpld = (GpuVplData *)_state.vpls.vplData.MapMemory();
    // int m = 0;
    // int mi = std::numeric_limits<int>::max();
    // std::cout << "Total Visible: " << *tv << std::endl;

    // for (int i = 0; i < *tv; ++i) {
    //     auto& position = vpld[i].position;
    //     auto& color = vpld[i].color;
    //     std::cout << position.v[0] << ", " << position.v[1] << ", " << position.v[2] << std::endl;
    //     std::cout << color.v[0] << ", " << color.v[1] << ", " << color.v[2] << std::endl;
    //     std::cout << vpld[i].farPlane << std::endl;
    //     std::cout << vpld[i].intensity << std::endl;
    //     std::cout << vpld[i].radius << std::endl;
    // }

    // int numNonZero = 0;
    // for (int i = 0; i < 1920 * 1080; ++i) {
    //     m = std::max(m, tiles[i].numVisible);
    //     mi = std::min(mi, tiles[i].numVisible);
    //     if (tiles[i].numVisible > 0) ++numNonZero;
    // }
    // std::cout << "MAX VPL: " << m << std::endl;
    // std::cout << "MIN VPL: " << mi << std::endl;
    // std::cout << "NNZ: " << numNonZero << std::endl;
    // _state.vpls.vplData.UnmapMemory();
    // _state.vpls.vplVisiblePerTile.UnmapMemory();
    // _state.vpls.vplNumVisible.UnmapMemory();
}

void RendererBackend::ComputeVirtualPointLightGlobalIllumination_(const std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer, const double deltaSeconds) {
    if (perVPLDistToViewer.size() == 0) return;

    // auto space = LogSpace<float>(1, 512, 30);
    // for (const auto& s : space) std::cout << s << " ";
    // std::cout << std::endl;

    const auto timePoint = std::chrono::high_resolution_clock::now();
    const float milliseconds = float(std::chrono::time_point_cast<std::chrono::milliseconds>(timePoint).time_since_epoch().count());

    glDisable(GL_DEPTH_TEST);
    BindShader_(state_.vplGlobalIllumination.get());
    state_.vpls.vplGIFbo.Bind();
    glViewport(0, 0, state_.vpls.vplGIFbo.GetColorAttachments()[0].Width(), state_.vpls.vplGIFbo.GetColorAttachments()[0].Height());

    // Set up infinite light color
    auto& cache = vplSmapCache_;
    const glm::vec3 lightColor = frame_->csc.worldLight->GetLuminance();
    state_.vplGlobalIllumination->SetVec3("infiniteLightColor", lightColor);

    state_.vplGlobalIllumination->SetInt("numTilesX", frame_->viewportWidth  / state_.vpls.tileXDivisor);
    state_.vplGlobalIllumination->SetInt("numTilesY", frame_->viewportHeight / state_.vpls.tileYDivisor);

    // All relevant rendering data is moved to the GPU during the light cull phase
    state_.vpls.vplUpdatedData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);
    state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);
    //state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 2);
    state_.vpls.shadowDiffuseIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 3);
    haltonSequence_.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 4);
    state_.vplGlobalIllumination->SetInt("haltonSize", int(haltonSequence.size()));

    state_.vplGlobalIllumination->SetMat4("invProjectionView", frame_->invProjectionView);
    for (size_t i = 0; i < cache.buffers.size(); ++i) {
        state_.vplGlobalIllumination->BindTexture("shadowCubeMaps[" + std::to_string(i) + "]", *cache.buffers[i].GetDepthStencilAttachment());
    }

    state_.vplGlobalIllumination->SetFloat("minRoughness", frame_->settings.GetMinRoughness());
    state_.vplGlobalIllumination->SetBool("usePerceptualRoughness", frame_->settings.usePerceptualRoughness);
    state_.vplGlobalIllumination->BindTexture("screen", state_.lightingColorBuffer);
    state_.vplGlobalIllumination->BindTexture("gDepth", state_.currentFrame.depth);
    state_.vplGlobalIllumination->BindTexture("gNormal", state_.currentFrame.normals);
    state_.vplGlobalIllumination->BindTexture("gAlbedo", state_.currentFrame.albedo);
    state_.vplGlobalIllumination->BindTexture("gBaseReflectivity", state_.currentFrame.baseReflectivity);
    state_.vplGlobalIllumination->BindTexture("gRoughnessMetallicAmbient", state_.currentFrame.roughnessMetallicAmbient);
    state_.vplGlobalIllumination->BindTexture("ssao", state_.ssaoOcclusionBlurredTexture);
    state_.vplGlobalIllumination->BindTexture("historyDepth", state_.vpls.vplGIDenoisedPrevFrameFbo.GetColorAttachments()[3]);
    state_.vplGlobalIllumination->SetFloat("time", milliseconds);
    state_.vplGlobalIllumination->SetInt("frameCount", int(INSTANCE(Engine)->FrameCount()));

    state_.vplGlobalIllumination->SetVec3("fogColor", frame_->settings.GetFogColor());
    state_.vplGlobalIllumination->SetFloat("fogDensity", frame_->settings.GetFogDensity());

    const Camera& camera = *frame_->camera;
    state_.vplGlobalIllumination->SetVec3("viewPosition", camera.GetPosition());
    state_.vplGlobalIllumination->SetInt("viewportWidth", frame_->viewportWidth);
    state_.vplGlobalIllumination->SetInt("viewportHeight", frame_->viewportHeight);

    RenderQuad_();
    
    UnbindShader_();
    state_.vpls.vplGIFbo.Unbind();

    std::vector<FrameBuffer *> buffers = {
        &state_.vpls.vplGIDenoisedFbo1,
        &state_.vpls.vplGIDenoisedFbo2
    };

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
    state_.vplGlobalIlluminationDenoising->SetFloat("framesPerSecond", float(1.0 / deltaSeconds));

    size_t bufferIndex = 0;
    const int maxIterations = 3;
    for (; bufferIndex < maxIterations; ++bufferIndex) {

        // The first iteration is used for reservoir merging so we don't
        // start increasing the multiplier until after the 2nd pass
        const int i = bufferIndex; //bufferIndex == 0 ? 0 : bufferIndex - 1;
        const int multiplier = std::pow(2, i) - 1;
        FrameBuffer * buffer = buffers[bufferIndex % buffers.size()];

        buffer->Bind();
        state_.vplGlobalIlluminationDenoising->BindTexture("indirectIllumination", indirectIllum);
        state_.vplGlobalIlluminationDenoising->BindTexture("indirectShadows", indirectShadows);
        state_.vplGlobalIlluminationDenoising->SetInt("multiplier", multiplier);
        state_.vplGlobalIlluminationDenoising->SetInt("passNumber", i);
        state_.vplGlobalIlluminationDenoising->SetBool("mergeReservoirs", bufferIndex == 0);

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

void RendererBackend::RenderScene(const double deltaSeconds) {
    CHECK_IS_APPLICATION_THREAD();

    const Camera& c = *frame_->camera;

    // Bind buffers
    GpuMeshAllocator::BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 32);
    GpuMeshAllocator::BindElementArrayBuffer();

    // Perform world light depth pass if enabled - needed by a lot of the rest of the frame so
    // do this first
    if (frame_->csc.worldLight->GetEnabled()) {
        RenderCSMDepth_();
    }

    std::vector<std::pair<LightPtr, double>>& perLightDistToViewer = perLightDistToViewer_;
    // // This one is just for shadow-casting lights
    std::vector<std::pair<LightPtr, double>>& perLightShadowCastingDistToViewer = perLightShadowCastingDistToViewer_;
    std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer = perVPLDistToViewer_;
    std::vector<int>& visibleVplIndices = visibleVplIndices_;

    // Perform point light pass
    UpdatePointLights_(perLightDistToViewer, perLightShadowCastingDistToViewer, perVPLDistToViewer, visibleVplIndices);

    // TEMP: Set up the light source
    //glm::vec3 lightPos(0.0f, 0.0f, 0.0f);
    //glm::vec3 lightColor(10.0f); 

    // Make sure some of our global GL states are set properly for primary rendering below
    glBlendFunc(state_.blendSFactor, state_.blendDFactor);
    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    RenderForwardPassPbr_();

    //glEnable(GL_BLEND);

    // Begin first SSAO pass (occlusion)
    RenderSsaoOcclude_();

    // Begin second SSAO pass (blurring)
    RenderSsaoBlur_();

    // Begin atmospheric pass
    RenderAtmosphericShadowing_();

    // Begin deferred lighting pass
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    state_.lightingFbo.Bind();

    //_unbindAllTextures();
    Pipeline* lighting = state_.lighting.get();
    if (frame_->csc.worldLight->GetEnabled()) {
        lighting = state_.lightingWithInfiniteLight.get();
    }

    BindShader_(lighting);
    InitLights_(lighting, perLightDistToViewer, state_.maxShadowCastingLightsPerFrame);
    lighting->BindTexture("atmosphereBuffer", state_.atmosphericTexture);
    lighting->SetMat4("invProjectionView", frame_->invProjectionView);
    lighting->BindTexture("gDepth", state_.currentFrame.depth);
    lighting->BindTexture("gNormal", state_.currentFrame.normals);
    lighting->BindTexture("gAlbedo", state_.currentFrame.albedo);
    lighting->BindTexture("gBaseReflectivity", state_.currentFrame.baseReflectivity);
    lighting->BindTexture("gRoughnessMetallicAmbient", state_.currentFrame.roughnessMetallicAmbient);
    lighting->BindTexture("ssao", state_.ssaoOcclusionBlurredTexture);
    lighting->SetFloat("windowWidth", frame_->viewportWidth);
    lighting->SetFloat("windowHeight", frame_->viewportHeight);
    lighting->SetVec3("fogColor", frame_->settings.GetFogColor());
    lighting->SetFloat("fogDensity", frame_->settings.GetFogDensity());
    RenderQuad_();
    state_.lightingFbo.Unbind();
    UnbindShader_();
    state_.finalScreenBuffer = state_.lightingFbo; // state_.lightingColorBuffer;

    // If world light is enabled perform VPL Global Illumination pass
    if (frame_->csc.worldLight->GetEnabled() && frame_->settings.globalIlluminationEnabled) {
        // Handle VPLs for global illumination (can't do this earlier due to needing position data from GBuffer)
        PerformVirtualPointLightCullingStage2_(perVPLDistToViewer);
        ComputeVirtualPointLightGlobalIllumination_(perVPLDistToViewer, deltaSeconds);
    }

    // Forward pass for all objects that don't interact with light (may also be used for transparency later as well)
    state_.lightingFbo.CopyFrom(state_.currentFrame.fbo, BufferBounds{0, 0, frame_->viewportWidth, frame_->viewportHeight}, BufferBounds{0, 0, frame_->viewportWidth, frame_->viewportHeight}, BufferBit::DEPTH_BIT, BufferFilter::NEAREST);
    // Blit to default framebuffer - not that the framebuffer you are writing to has to match the internal format
    // of the framebuffer you are reading to!
    glEnable(GL_DEPTH_TEST);
    state_.lightingFbo.Bind();
    
    // Skybox is one that does not interact with light at all
    RenderSkybox_();

    // No light interaction
    // TODO: Allow to cast shadows? May be useful in scenes that want to use
    // purely diffuse non-pbr objects which still cast shadows.
    RenderForwardPassFlat_();

    // Render bounding boxes
    //RenderBoundingBoxes_(frame_->visibleInstancedFlatMeshes);
    //RenderBoundingBoxes_(frame_->visibleInstancedDynamicPbrMeshes);
    //RenderBoundingBoxes_(frame_->visibleInstancedStaticPbrMeshes);

    state_.lightingFbo.Unbind();
    state_.finalScreenBuffer = state_.lightingFbo;// state_.lightingColorBuffer;
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Enable post-FX effects such as bloom
    PerformPostFxProcessing_();

    // Perform final drawing to screen + gamma correction
    FinalizeFrame_();

    // Unbind element array buffer
    GpuMeshAllocator::UnbindElementArrayBuffer();
}

void RendererBackend::RenderForwardPassPbr_() {
    // Make sure to bind our own frame buffer for rendering
    state_.currentFrame.fbo.Bind();

    // Perform depth prepass
    BindShader_(state_.depthPrepass.get());

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    // Begin geometry pass
    BindShader_(state_.geometry.get());

    state_.geometry->SetMat4("jitterProjectionView", frame_->jitterProjectionView);

    //glDepthFunc(GL_LEQUAL);

    const CommandBufferSelectionFunction select = [](GpuCommandBuffer2Ptr& b) {
        return b->GetVisibleDrawCommandsBuffer();
    };

    Render_(*state_.geometry.get(), frame_->drawCommands->dynamicPbrMeshes, select, true);
    Render_(*state_.geometry.get(), frame_->drawCommands->staticPbrMeshes, select, true);

    state_.currentFrame.fbo.Unbind();

    UnbindShader_();

    //glDepthMask(GL_TRUE);
}

void RendererBackend::RenderForwardPassFlat_() {
    BindShader_(state_.forward.get());

    // TODO: Enable TAA for flat forward geometry
    state_.forward->SetMat4("jitterProjectionView", frame_->projectionView);

    glEnable(GL_DEPTH_TEST);

    const CommandBufferSelectionFunction select = [](GpuCommandBuffer2Ptr& b) {
        return b->GetVisibleDrawCommandsBuffer();
    };

    Render_(*state_.forward.get(), frame_->drawCommands->flatMeshes, select, false);

    UnbindShader_();
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
    std::vector<PostFXBuffer> finalizedPostFxFrames(state_.numDownsampleIterations + state_.numUpsampleIterations);
   
    Pipeline* bloom = state_.bloom.get();
    BindShader_(bloom);

    Texture lightingColorBuffer = state_.finalScreenBuffer.GetColorAttachments()[0];

    // Downsample stage
    bloom->SetBool("downsamplingStage", true);
    bloom->SetBool("upsamplingStage", false);
    bloom->SetBool("finalStage", false);
    bloom->SetBool("gaussianStage", false);
    for (int i = 0, gaussian = 0; i < state_.numDownsampleIterations; ++i, gaussian += 2) {
        PostFXBuffer& buffer = state_.postFxBuffers[i];
        Texture colorTex = buffer.fbo.GetColorAttachments()[0];
        auto width = colorTex.Width();
        auto height = colorTex.Height();
        bloom->SetFloat("viewportX", float(width));
        bloom->SetFloat("viewportY", float(height));
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
        for (int i = 0; i < 2; ++i) {
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
    int postFXIndex = state_.numDownsampleIterations;
    for (int i = state_.numDownsampleIterations - 1; i >= 0; --i, ++postFXIndex) {
        PostFXBuffer& buffer = state_.postFxBuffers[postFXIndex];
        auto width = buffer.fbo.GetColorAttachments()[0].Width();
        auto height = buffer.fbo.GetColorAttachments()[0].Height();
        bloom->SetFloat("viewportX", float(width));
        bloom->SetFloat("viewportY", float(height));
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
    const glm::vec3& normalizedLightDirCamSpace = frame_->csc.worldLightDirectionCameraSpace;
    const Texture& colorTex = state_.atmosphericTexture;
    const float w = colorTex.Width();
    const float h = colorTex.Height();
    const float xlight = w * ((projection[0][0] * normalizedLightDirCamSpace.x + 
                               projection[0][1] * normalizedLightDirCamSpace.y + 
                               projection[0][2] * normalizedLightDirCamSpace.z) / (2.0f * normalizedLightDirCamSpace.z) + 0.5f);
    const float ylight = h * ((projection[1][0] * normalizedLightDirCamSpace.x + 
                               projection[1][1] * normalizedLightDirCamSpace.y + 
                               projection[1][2] * normalizedLightDirCamSpace.z) / (2.0f * normalizedLightDirCamSpace.z) + 0.5f);
    
    return 2.0f * normalizedLightDirCamSpace.z * glm::vec3(xlight, ylight, 1.0f);
}

void RendererBackend::PerformAtmosphericPostFx_() {
    if (!frame_->csc.worldLight->GetEnabled()) return;

    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    const glm::vec3 lightPosition = CalculateAtmosphericLightPosition_();
    //const float sinX = stratus::sine(_frame->csc.worldLight->getRotation().x).value();
    //const float cosX = stratus::cosine(_frame->csc.worldLight->getRotation().x).value();
    const glm::vec3 lightColor = frame_->csc.worldLight->GetAtmosphereColor();// * glm::vec3(cosX, cosX, sinX);

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

    // Now render the screen
    BindShader_(state_.fullscreen.get());
    state_.fullscreen->BindTexture("screen", state_.finalScreenBuffer.GetColorAttachments()[0]);
    RenderQuad_();
    UnbindShader_();
}

void RendererBackend::End() {
    CHECK_IS_APPLICATION_THREAD();

    GraphicsDriver::SwapBuffers(frame_->settings.vsyncEnabled);

    frame_.reset();
}

void RendererBackend::RenderQuad_() {
    GetMesh(state_.screenQuad, 0)->Render(1, GpuArrayBuffer());
    //_state.screenQuad->GetMeshContainer(0)->mesh->Render(1, GpuArrayBuffer());
}

RendererBackend::ShadowMapCache RendererBackend::CreateShadowMap3DCache_(uint32_t resolutionX, uint32_t resolutionY, uint32_t count, bool vpl) {
    ShadowMapCache cache;
    
    int remaining = int(count);
    while (remaining > 0 && cache.buffers.size() < MAX_TOTAL_SHADOW_ATLASES) {
        // Determine how many entries will be present in this atlas
        int tmp = remaining - MAX_TOTAL_SHADOWS_PER_ATLAS;
        int entries = MAX_TOTAL_SHADOWS_PER_ATLAS;
        if (tmp < 0) {
            entries = remaining;
        }
        remaining = tmp;

        const uint32_t numLayers = entries;

        std::vector<Texture> attachments;
        Texture texture = Texture(TextureConfig{ TextureType::TEXTURE_CUBE_MAP_ARRAY, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, resolutionX, resolutionY, numLayers, false }, NoTextureData);
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
    for (int index = 0; index < cache.buffers.size(); ++index) {
        const int depth = int(cache.buffers[index].GetDepthStencilAttachment()->Depth());
        for (int layer = 0; layer < depth; ++layer) {
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
    const Camera & lightCam = *frame_->csc.worldLightCamera;
    // glm::mat4 lightView = lightCam.getViewTransform();
    const glm::vec3 direction = lightCam.GetDirection();

    s->SetVec3("infiniteLightDirection", direction);    
    s->BindTexture("infiniteLightShadowMap", *frame_->csc.fbo.GetDepthStencilAttachment());
    for (int i = 0; i < frame_->csc.cascades.size(); ++i) {
        //s->bindTexture("infiniteLightShadowMaps[" + std::to_string(i) + "]", *_state.csms[i].fbo.getDepthStencilAttachment());
        s->SetMat4("cascadeProjViews[" + std::to_string(i) + "]", frame_->csc.cascades[i].projectionViewSample);
        // s->setFloat("cascadeSplits[" + std::to_string(i) + "]", _state.cascadeSplits[i]);
    }

    for (int i = 0; i < 2; ++i) {
        s->SetVec4("shadowOffset[" + std::to_string(i) + "]", frame_->csc.cascadeShadowOffsets[i]);
    }

    for (int i = 0; i < frame_->csc.cascades.size() - 1; ++i) {
        // s->setVec3("cascadeScale[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeScale[0]);
        // s->setVec3("cascadeOffset[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeOffset[0]);
        s->SetVec4("cascadePlanes[" + std::to_string(i) + "]", frame_->csc.cascades[i + 1].cascadePlane);
    }
}

void RendererBackend::InitLights_(Pipeline * s, const std::vector<std::pair<LightPtr, double>> & lights, const size_t maxShadowLights) {
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
    //int lightIndex = 0;
    //int shadowLightIndex = 0;
    //for (int i = 0; i < lights.size(); ++i) {
    //    LightPtr light = lights[i].first;
    //    PointLight * point = (PointLight *)light.get();
    //    const double distance = lights[i].second; //glm::distance(c.getPosition(), light->position);
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

    std::vector<GpuPointLight> gpuLights;
    std::vector<GpuAtlasEntry> gpuShadowCubeMaps;
    std::vector<GpuPointLight> gpuShadowLights;
    gpuLights.reserve(lights.size());
    gpuShadowCubeMaps.reserve(maxShadowLights);
    gpuShadowLights.reserve(maxShadowLights);
    for (int i = 0; i < lights.size(); ++i) {
        LightPtr light = lights[i].first;
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

    state_.nonShadowCastingPointLights.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);
    state_.shadowIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);
    state_.shadowCastingPointLights.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 2);

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
    s->SetInt("numLights", int(gpuLights.size()));
    s->SetInt("numShadowLights", int(gpuShadowLights.size()));
    s->SetVec3("viewPosition", c.GetPosition());
    s->SetFloat("emissionStrength", frame_->settings.GetEmissionStrength());
    s->SetFloat("minRoughness", frame_->settings.GetMinRoughness());
    s->SetBool("usePerceptualRoughness", frame_->settings.usePerceptualRoughness);
    for (size_t i = 0; i < cache.buffers.size(); ++i) {
        s->BindTexture("shadowCubeMaps[" + std::to_string(i) + "]", *cache.buffers[i].GetDepthStencilAttachment());
    }
    const glm::vec3 lightPosition = CalculateAtmosphericLightPosition_();
    s->SetVec3("atmosphericLightPos", lightPosition);

    // Set up world light if enabled
    //glm::mat4 lightView = constructViewMatrix(_state.worldLight.getRotation(), _state.worldLight.getPosition());
    //glm::mat4 lightView = constructViewMatrix(_state.worldLight.getRotation(), glm::vec3(0.0f));
    // Camera lightCam(false);
    // lightCam.setAngle(_state.worldLight.getRotation());
    const Camera & lightCam = *frame_->csc.worldLightCamera;
    glm::mat4 lightWorld = lightCam.GetWorldTransform();
    // glm::mat4 lightView = lightCam.getViewTransform();
    glm::vec3 direction = lightCam.GetDirection(); //glm::vec3(-lightWorld[2].x, -lightWorld[2].y, -lightWorld[2].z);
    // STRATUS_LOG << "Light direction: " << direction << std::endl;
    lightColor = frame_->csc.worldLight->GetLuminance();
    s->SetVec3("infiniteLightColor", lightColor);
    s->SetFloat("infiniteLightZnear", frame_->csc.znear);
    s->SetFloat("infiniteLightZfar", frame_->csc.zfar);
    s->SetFloat("infiniteLightDepthBias", frame_->csc.worldLight->GetDepthBias());
    s->SetFloat("worldLightAmbientIntensity", frame_->csc.worldLight->GetAmbientIntensity());

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