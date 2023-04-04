
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
    if (severity == GL_DEBUG_SEVERITY_MEDIUM || severity == GL_DEBUG_SEVERITY_HIGH) {
       //std::cout << "[OpenGL] " << message << std::endl;
    }
}

// These are the first 16 values of the Halton sequence. For more information see:
//     https://en.wikipedia.org/wiki/Halton_sequence
//     https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler

static const std::vector<std::pair<float, float>> haltonSequence = {
    { 1.0f /  2.0f,  1.0f /  3.0f},
    { 1.0f /  4.0f,  2.0f /  3.0f},
    { 3.0f /  4.0f,  1.0f /  9.0f},
    { 1.0f /  8.0f,  4.0f /  9.0f},
    { 5.0f /  8.0f,  7.0f /  9.0f},
    { 3.0f /  8.0f,  2.0f /  9.0f},
    { 7.0f /  8.0f,  5.0f /  9.0f},
    { 1.0f / 16.0f,  8.0f /  9.0f},
    { 9.0f / 16.0f,  1.0f / 27.0f},
    { 5.0f / 16.0f, 10.0f / 27.0f},
    {13.0f / 16.0f, 19.0f / 27.0f},
    { 3.0f / 16.0f,  4.0f / 27.0f},
    {11.0f / 16.0f, 13.0f / 27.0f},
    { 7.0f / 16.0f, 22.0f / 27.0f},
    {15.0f / 16.0f,  7.0f / 27.0f},
    { 1.0f / 32.0f, 16.0f / 27.0f},
};

RendererBackend::RendererBackend(const uint32_t width, const uint32_t height, const std::string& appName) {
    static_assert(sizeof(GpuVec) == 16, "Memory alignment must match up with GLSL");

    isValid_ = true;

    // Set up OpenGL debug logging
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(OpenGLDebugCallback, nullptr);

    const std::filesystem::path shaderRoot("../Source/Shaders");
    const ShaderApiVersion version{GraphicsDriver::GetConfig().majorVersion, GraphicsDriver::GetConfig().minorVersion};

    // Initialize the pipelines
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

    // Set up the hdr/gamma postprocessing shader

    state_.gammaTonemap = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"gammaTonemap.vs", ShaderType::VERTEX},
        Shader{"gammaTonemap.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.gammaTonemap.get());

    // Set up the shadow preprocessing shaders
    for (int i = 0; i < 6; ++i) {
        state_.shadows.push_back(std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
            Shader{"shadow.vs", ShaderType::VERTEX},
            //Shader{"shadow.gs", ShaderType::GEOMETRY},
            Shader{"shadow.fs", ShaderType::FRAGMENT}},
            {{"DEPTH_LAYER", std::to_string(i)}}))
        );
        state_.shaders.push_back(state_.shadows[i].get());

        state_.vplShadows.push_back(std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
            Shader{"shadow.vs", ShaderType::VERTEX},
            //Shader{"shadow.gs", ShaderType::GEOMETRY},
            Shader{"shadowVpl.fs", ShaderType::FRAGMENT}},
            {{"DEPTH_LAYER", std::to_string(i)}}))
        );
        state_.shaders.push_back(state_.vplShadows[i].get());
    }

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

    state_.vplGlobalIlluminationBlurring = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
        Shader{"vpl_pbr_gi.vs", ShaderType::VERTEX},
        Shader{"vpl_pbr_gi_blur.fs", ShaderType::FRAGMENT}}));
    state_.shaders.push_back(state_.vplGlobalIlluminationBlurring.get());

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

    state_.dummyCubeMap = CreateShadowMap3D_(state_.shadowCubeMapX, state_.shadowCubeMapY, false);

    // Init constant SSAO data
    InitSSAO_();

    // Init constant atmospheric data
    InitAtmosphericShadowing_();

    // Create a pool of shadow maps for point lights to use
    InitPointShadowMaps_();

    // Virtual point lights
    InitializeVplData_();
}

void RendererBackend::InitPointShadowMaps_() {
    // Create the normal point shadow map cache
    for (int i = 0; i < state_.numRegularShadowMaps; ++i) {
        CreateShadowMap3D_(state_.shadowCubeMapX, state_.shadowCubeMapY, false);
    }

    // Initialize the point light buffers including shadow map texture buffer
    const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
    state_.nonShadowCastingPointLights = GpuBuffer(nullptr, sizeof(GpuPointLight) * state_.maxTotalRegularLightsPerFrame, flags);
    state_.shadowCastingPointLights = GpuBuffer(nullptr, sizeof(GpuPointLight) * state_.maxShadowCastingLightsPerFrame, flags);
    state_.shadowCubeMaps = GpuBuffer(nullptr, sizeof(GpuTextureHandle) * state_.maxShadowCastingLightsPerFrame, flags);

    // Create the virtual point light shadow map cache
    for (int i = 0; i < MAX_TOTAL_VPL_SHADOW_MAPS; ++i) {
        CreateShadowMap3D_(state_.vpls.vplShadowCubeMapX, state_.vpls.vplShadowCubeMapY, true);
    }
}

void RendererBackend::InitializeVplData_() {
    const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
    std::vector<int> visibleIndicesData(MAX_TOTAL_VPLS_BEFORE_CULLING, 0);
    state_.vpls.vplDiffuseMaps = GpuBuffer(nullptr, sizeof(GpuTextureHandle) * MAX_TOTAL_VPLS_BEFORE_CULLING, flags);
    state_.vpls.vplShadowMaps = GpuBuffer(nullptr, sizeof(GpuTextureHandle) * MAX_TOTAL_VPLS_BEFORE_CULLING, flags);
    state_.vpls.vplVisibleIndices = GpuBuffer((const void *)visibleIndicesData.data(), sizeof(int) * visibleIndicesData.size(), flags);
    state_.vpls.vplData = GpuBuffer(nullptr, sizeof(GpuVplData) * MAX_TOTAL_VPLS_BEFORE_CULLING, flags);
    state_.vpls.vplNumVisible = GpuBuffer(nullptr, sizeof(int), flags);
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
        &state_.currentFrame
    };

    for (GBuffer* gbptr : buffers) {
        GBuffer& buffer = *gbptr;

        // Position buffer
        //buffer.position = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, NoTextureData);
        //buffer.position.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

        // Normal buffer
        buffer.normals = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.normals.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

        // Create the color buffer - notice that is uses higher
        // than normal precision. This allows us to write color values
        // greater than 1.0 to support things like HDR.
        buffer.albedo = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.albedo.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

        // Base reflectivity buffer
        buffer.baseReflectivity = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.baseReflectivity.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

        // Roughness-Metallic-Ambient buffer
        buffer.roughnessMetallicAmbient = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.roughnessMetallicAmbient.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

        // Create the Structure buffer which contains rgba where r=partial x-derivative of camera-space depth, g=partial y-derivative of camera-space depth, b=16 bits of depth, a=final 16 bits of depth (b+a=32 bits=depth)
        buffer.structure = Texture(TextureConfig{ TextureType::TEXTURE_RECTANGLE, TextureComponentFormat::RGBA, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.structure.SetMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        buffer.structure.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        // Create velocity buffer
        buffer.velocity = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RG, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.velocity.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

        // Create the depth buffer
        buffer.depth = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
        buffer.depth.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

        // Create the frame buffer with all its texture attachments
        //buffer.fbo = FrameBuffer({buffer.position, buffer.normals, buffer.albedo, buffer.baseReflectivity, buffer.roughnessMetallicAmbient, buffer.structure, buffer.depth});
        buffer.fbo = FrameBuffer({ buffer.normals, buffer.albedo, buffer.baseReflectivity, buffer.roughnessMetallicAmbient, buffer.structure, buffer.velocity, buffer.depth });
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
    state_.lightingDepthBuffer.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Attach the textures to the FBO
    state_.lightingFbo = FrameBuffer({state_.lightingColorBuffer, state_.lightingHighBrightnessBuffer, state_.lightingDepthBuffer});
    if (!state_.lightingFbo.Valid()) {
        isValid_ = false;
        return;
    }

    // Code to create the SSAO fbo
    state_.ssaoOcclusionTexture = Texture(TextureConfig{TextureType::TEXTURE_RECTANGLE, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.ssaoOcclusionTexture.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    state_.ssaoOcclusionTexture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.ssaoOcclusionBuffer = FrameBuffer({state_.ssaoOcclusionTexture});
    if (!state_.ssaoOcclusionBuffer.Valid()) {
        isValid_ = false;
        return;
    }

    // Code to create the SSAO blurred fbo
    state_.ssaoOcclusionBlurredTexture = Texture(TextureConfig{TextureType::TEXTURE_RECTANGLE, TextureComponentFormat::RED, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.ssaoOcclusionBlurredTexture.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    state_.ssaoOcclusionBlurredTexture.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.ssaoOcclusionBlurredBuffer = FrameBuffer({state_.ssaoOcclusionBlurredTexture});
    if (!state_.ssaoOcclusionBlurredBuffer.Valid()) {
        isValid_ = false;
        return;
    }

    // Code to create the Virtual Point Light Global Illumination fbo
    state_.vpls.vplGIColorBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.vpls.vplGIColorBuffer.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    state_.vpls.vplGIColorBuffer.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.vpls.vplGIFbo = FrameBuffer({state_.vpls.vplGIColorBuffer});
    if (!state_.vpls.vplGIFbo.Valid()) {
        isValid_ = false;
        return;
    }

    state_.vpls.vplGIBlurredBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false}, NoTextureData);
    state_.vpls.vplGIBlurredBuffer.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    state_.vpls.vplGIBlurredBuffer.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    state_.vpls.vplGIBlurredFbo = FrameBuffer({state_.vpls.vplGIBlurredBuffer});
    if (!state_.vpls.vplGIBlurredBuffer.Valid()) {
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
    Texture taa = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGBA, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, frame_->viewportWidth, frame_->viewportHeight, 0, false }, NoTextureData);
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
        state_.lightingFbo.Clear(color);
        state_.vpls.vplGIFbo.Clear(color);
        state_.vpls.vplGIBlurredFbo.Clear(color);

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

    // Increment halton index
    currentHaltonIndex_ = (currentHaltonIndex_ + 1) % haltonSequence.size();

    // Make sure we set our context as the active one
    GraphicsDriver::MakeContextCurrent();

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
static std::vector<glm::mat4> GenerateLightViewTransforms(const glm::mat4 & projection, const glm::vec3 & lightPos) {
    return std::vector<glm::mat4>{
        //                       pos       pos + dir                                  up
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
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

void RendererBackend::RenderBoundingBoxes_(GpuCommandBufferPtr& buffer) {
    if (buffer->NumDrawCommands() == 0 || buffer->aabbs.size() == 0) return;

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

void RendererBackend::RenderBoundingBoxes_(std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>& map) {
    for (auto& entry : map) {
        RenderBoundingBoxes_(entry.second);
    }
}

void RendererBackend::RenderImmediate_(const RenderFaceCulling cull, GpuCommandBufferPtr& buffer) {
    if (buffer->NumDrawCommands() == 0) return;

    frame_->materialInfo.materialsBuffer.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 30);
    buffer->BindMaterialIndicesBuffer(31);
    buffer->BindModelTransformBuffer(13);
    buffer->BindPrevFrameModelTransformBuffer(14);
    buffer->BindIndirectDrawCommands();

    SetCullState(cull);

    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, (const void *)0, (GLsizei)buffer->NumDrawCommands(), (GLsizei)0);

    buffer->UnbindIndirectDrawCommands();
}

void RendererBackend::Render_(Pipeline& s, const RenderFaceCulling cull, GpuCommandBufferPtr& buffer, bool isLightInteracting, bool removeViewTranslation) {
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

    s.SetMat4("projectionView", projectionView);
    s.SetMat4("prevProjectionView", frame_->prevProjectionView);

    s.SetInt("viewWidth", frame_->viewportWidth);
    s.SetInt("viewHeight", frame_->viewportHeight);
    //s->setMat4("projectionView", &projection[0][0]);
    //s->setMat4("view", &view[0][0]);

    RenderImmediate_(cull, buffer);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
}

void RendererBackend::Render_(Pipeline& s, std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>& map, bool isLightInteracting, bool removeViewTranslation) {
    for (auto& entry : map) {
        Render_(s, entry.first, entry.second, isLightInteracting, removeViewTranslation);
    }
}

void RendererBackend::RenderImmediate_(std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>& map) {
    for (auto& entry : map) {
        RenderImmediate_(entry.first, entry.second);
    }
}

void RendererBackend::RenderSkybox_() {
    BindShader_(state_.skybox.get());
    glDepthMask(GL_FALSE);

    TextureLoadingStatus status;
    Texture sky = INSTANCE(ResourceManager)->LookupTexture(frame_->skybox, status);
    if (ValidateTexture(sky, status)) {
        const glm::mat4& projection = frame_->projection;
        const glm::mat4 view = glm::mat4(glm::mat3(frame_->camera->GetViewTransform()));
        const glm::mat4 projectionView = projection * view;

        // _state.skybox->setMat4("projection", projection);
        // _state.skybox->setMat4("view", view);
        state_.skybox->SetMat4("projectionView", projectionView);

        state_.skybox->SetVec3("colorMask", frame_->skyboxColorMask);
        state_.skybox->SetFloat("intensity", frame_->skyboxIntensity);
        state_.skybox->BindTexture("skybox", sky);

        GetMesh(state_.skyboxCube, 0)->Render(1, GpuArrayBuffer());
        //_state.skyboxCube->GetMeshContainer(0)->mesh->Render(1, GpuArrayBuffer());
    }

    UnbindShader_();
    glDepthMask(GL_TRUE);
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
    glPolygonOffset(3.0f, 0.0f);
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
        auto& csm = frame_->csc.cascades[cascade];
        shader->SetMat4("shadowMatrix", csm.projectionViewRender);
        const size_t lod = cascade * 2;
        //RenderImmediate_(frame_->instancedStaticPbrMeshes[lod]);
        //RenderImmediate_(frame_->instancedDynamicPbrMeshes[lod]);
        RenderImmediate_(frame_->selectedLodsDynamicPbrMeshes);
        RenderImmediate_(frame_->selectedLodsStaticPbrMeshes);

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

    BindShader_(state_.atmospheric.get());
    state_.atmosphericFbo.Bind();
    state_.atmospheric->SetVec3("frustumParams", frustumParams);
    state_.atmospheric->SetMat4("shadowMatrix", shadowMatrix);
    state_.atmospheric->BindTexture("structureBuffer", state_.currentFrame.structure);
    state_.atmospheric->BindTexture("infiniteLightShadowMap", *frame_->csc.fbo.GetDepthStencilAttachment());
    
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
    std::sort(perLightDistToViewer.begin(), perLightDistToViewer.end(), comparison);
    std::sort(perLightShadowCastingDistToViewer.begin(), perLightShadowCastingDistToViewer.end(), comparison);

    // Remove lights exceeding the absolute maximum
    if (perLightDistToViewer.size() > state_.maxTotalRegularLightsPerFrame) {
        perLightDistToViewer.resize(state_.maxTotalRegularLightsPerFrame);
    }

    // Remove shadow-casting lights that exceed our max count
    if (perLightShadowCastingDistToViewer.size() > state_.maxShadowCastingLightsPerFrame) {
        perLightShadowCastingDistToViewer.resize(state_.maxShadowCastingLightsPerFrame);
    }

    // Remove vpls exceeding absolute maximum
    if (worldLightEnabled) {
        std::sort(perVPLDistToViewer.begin(), perVPLDistToViewer.end(), comparison);
        if (perVPLDistToViewer.size() > MAX_TOTAL_VPLS_BEFORE_CULLING) {
            perVPLDistToViewer.resize(MAX_TOTAL_VPLS_BEFORE_CULLING);
        }

        InitVplFrameData_(perVPLDistToViewer);
        PerformVirtualPointLightCullingStage1_(perVPLDistToViewer, visibleVplIndices);
    }

    // Check if any need to have a new shadow map pulled from the cache
    for (const auto&[light, _] : perLightShadowCastingDistToViewer) {
        if (!ShadowMapExistsForLight_(light)) {
            frame_->lightsToUpate.PushBack(light);
        }
    }

    for (size_t i = 0; i < visibleVplIndices.size(); ++i) {
        const int index = visibleVplIndices[i];
        auto light = perVPLDistToViewer[index].first;
        if (!ShadowMapExistsForLight_(light)) {
            frame_->lightsToUpate.PushBack(light);
        }
    }

    // Set blend func just for shadow pass
    // glBlendFunc(GL_ONE, GL_ONE);
    glEnable(GL_DEPTH_TEST);
    // Perform the shadow volume pre-pass
    for (int shadowUpdates = 0; shadowUpdates < state_.maxShadowUpdatesPerFrame && frame_->lightsToUpate.Size() > 0; ++shadowUpdates) {
        auto light = frame_->lightsToUpate.PopFront();
        // Ideally this won't be needed but just in case
        if ( !light->CastsShadows() ) continue;
        //const double distance = perLightShadowCastingDistToViewer.find(light)->second;
    
        // TODO: Make this work with spotlights
        //PointLightPtr point = (PointLightPtr)light;
        PointLight * point = (PointLight *)light.get();
        ShadowMap3D smap = GetOrAllocateShadowMapForLight_(light);

        const glm::mat4 lightPerspective = glm::perspective<float>(glm::radians(90.0f), float(smap.shadowCubeMap.Width()) / smap.shadowCubeMap.Height(), point->GetNearPlane(), point->GetFarPlane());

        // glBindFramebuffer(GL_FRAMEBUFFER, smap.frameBuffer);
        smap.frameBuffer.Clear(glm::vec4(1.0f));
        smap.frameBuffer.Bind();
        glViewport(0, 0, smap.shadowCubeMap.Width(), smap.shadowCubeMap.Height());
        // Current pass only cares about depth buffer
        // glClear(GL_DEPTH_BUFFER_BIT);

        auto transforms = GenerateLightViewTransforms(lightPerspective, point->GetPosition());
        for (size_t i = 0; i < transforms.size(); ++i) {
            Pipeline * shader = light->IsVirtualLight() ? state_.vplShadows[i].get() : state_.shadows[i].get();
            BindShader_(shader);

            shader->SetMat4("shadowMatrix", transforms[i]);
            shader->SetVec3("lightPos", light->GetPosition());
            shader->SetFloat("farPlane", point->GetFarPlane());

            RenderImmediate_(frame_->instancedStaticPbrMeshes[0]);
            if ( !point->IsStaticLight() ) RenderImmediate_(frame_->instancedDynamicPbrMeshes[0]);

            UnbindShader_();
        }

        // Unbind
        smap.frameBuffer.Unbind();
    }
}

void RendererBackend::PerformVirtualPointLightCullingStage1_(
    const std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer,
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
    state_.vpls.vplNumVisible.CopyDataToBuffer(0, sizeof(int), (const void *)&numVisible);
    state_.vpls.vplNumVisible.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);

    // Bind light data and visibility indices
    state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 3);
    state_.vpls.vplData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);

    InitCoreCSMData_(state_.vplCulling.get());
    state_.vplCulling->DispatchCompute(1, 1, 1);
    state_.vplCulling->SynchronizeCompute();

    state_.vplCulling->Unbind();

    int totalVisible = *(int *)state_.vpls.vplNumVisible.MapMemory();
    state_.vpls.vplNumVisible.UnmapMemory();

    // These should still be sorted since the GPU compute shader doesn't reorder them
    int * indices = (int *)state_.vpls.vplVisibleIndices.MapMemory();
    visibleVplIndices.resize(totalVisible);

    for (int i = 0; i < totalVisible; ++i) {
        visibleVplIndices[i] = indices[i];
    }

    state_.vpls.vplVisibleIndices.UnmapMemory();

    //STRATUS_LOG << totalVisible << std::endl;

    if (totalVisible > MAX_TOTAL_VPLS_PER_FRAME) {
        visibleVplIndices.resize(MAX_TOTAL_VPLS_PER_FRAME);
        totalVisible = MAX_TOTAL_VPLS_PER_FRAME;
        // visibleVplIndices.clear();

        // // We want at least 64 lights close to the viewer
        // for (int i = 0; i < 64; ++i) {
        //     visibleVplIndices.push_back(indices[i]);
        // }

        // const int rest = totalVisible - 64;
        // const int step = std::max<int>(rest / (MAX_TOTAL_VPLS_PER_FRAME - 64), 1);
        // for (int i = 64; i < totalVisible; i += step) {
        //     visibleVplIndices.push_back(indices[i]);
        // }

        // totalVisible = int(visibleVplIndices.size());
        // // Make sure we didn't go over because of step size
        // if (visibleVplIndices.size() > MAX_TOTAL_VPLS_PER_FRAME) {
        //     visibleVplIndices.resize(MAX_TOTAL_VPLS_PER_FRAME);
        //     totalVisible = MAX_TOTAL_VPLS_PER_FRAME;
        // }

        state_.vpls.vplNumVisible.CopyDataToBuffer(0, sizeof(int), (const void *)&totalVisible);
        state_.vpls.vplVisibleIndices.CopyDataToBuffer(0, sizeof(int) * totalVisible, (const void *)visibleVplIndices.data());
    }
}

void RendererBackend::PerformVirtualPointLightCullingStage2_(
    const std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer,
    const std::vector<int>& visibleVplIndices) {

    if (perVPLDistToViewer.size() == 0) return;

    // Pack data into system memory
    std::vector<GpuTextureHandle> diffuseHandles(perVPLDistToViewer.size());
    std::vector<GpuTextureHandle> smapHandles(perVPLDistToViewer.size());
    for (size_t i = 0; i < visibleVplIndices.size(); ++i) {
        const int index = visibleVplIndices[i];
        VirtualPointLight * point = (VirtualPointLight *)perVPLDistToViewer[index].first.get();
        auto smap = GetOrAllocateShadowMapForLight_(perVPLDistToViewer[index].first);
        diffuseHandles[index] = smap.diffuseCubeMap.GpuHandle();
        smapHandles[index] = smap.shadowCubeMap.GpuHandle();
    }

    // Move data to GPU memory
    state_.vpls.vplDiffuseMaps.CopyDataToBuffer(0, sizeof(GpuTextureHandle) * diffuseHandles.size(), (const void *)diffuseHandles.data());
    state_.vpls.vplShadowMaps.CopyDataToBuffer(0, sizeof(GpuTextureHandle) * smapHandles.size(), (const void *)smapHandles.data());

    const Camera & lightCam = *frame_->csc.worldLightCamera;
    // glm::mat4 lightView = lightCam.getViewTransform();
    const glm::vec3 direction = lightCam.GetDirection();

    state_.vplColoring->Bind();

    // Bind inputs
    state_.vplColoring->SetVec3("infiniteLightDirection", direction);
    state_.vplColoring->SetVec3("infiniteLightColor", frame_->csc.worldLight->GetLuminance());

    state_.vpls.vplNumVisible.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);
    state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 3);
    state_.vpls.vplDiffuseMaps.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 5);

    // Bind outputs
    state_.vpls.vplData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);

    // Dispatch and synchronize
    state_.vplColoring->DispatchCompute(1, 1, 1);
    state_.vplColoring->SynchronizeCompute();

    state_.vplColoring->Unbind();

    // Now perform culling per tile since we now know which lights are active
    state_.vplTileDeferredCullingStage1->Bind();

    // Bind inputs
    //_state.vplTileDeferredCullingStage1->bindTexture("gPosition", _state.buffer.position);
    state_.vplTileDeferredCullingStage1->SetMat4("invProjectionView", frame_->invProjectionView);
    state_.vplTileDeferredCullingStage1->BindTexture("gDepth", state_.currentFrame.depth);
    state_.vplTileDeferredCullingStage1->BindTexture("gNormal", state_.currentFrame.normals);
    // _state.vplTileDeferredCulling->setInt("viewportWidth", _frame->viewportWidth);
    // _state.vplTileDeferredCulling->setInt("viewportHeight", _frame->viewportHeight);

    state_.vpls.vplData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);
    state_.vpls.vplShadowMaps.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 11);

    // Bind outputs
    state_.vpls.vplStage1Results.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);

    // Dispatch and synchronize
    state_.vplTileDeferredCullingStage1->DispatchCompute(
        (unsigned int)frame_->viewportWidth  / state_.vpls.tileXDivisor,
        (unsigned int)frame_->viewportHeight / state_.vpls.tileYDivisor,
        1
    );
    state_.vplTileDeferredCullingStage1->SynchronizeCompute();

    state_.vplTileDeferredCullingStage1->Unbind();

    // Perform stage 2 of the tiled deferred culling
    state_.vplTileDeferredCullingStage2->Bind();

    // Bind inputs
    state_.vplTileDeferredCullingStage2->SetVec3("viewPosition", frame_->camera->GetPosition());

    state_.vpls.vplData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);
    state_.vpls.vplStage1Results.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);
    state_.vpls.vplNumVisible.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 2);
    state_.vpls.vplVisibleIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 3);
    state_.vpls.vplShadowMaps.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 11);

    // Bind outputs
    state_.vpls.vplVisiblePerTile.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 4);
    
    // Dispatch and synchronize
    state_.vplTileDeferredCullingStage2->DispatchCompute(
        (unsigned int)frame_->viewportWidth  / (state_.vpls.tileXDivisor * 32),
        (unsigned int)frame_->viewportHeight / (state_.vpls.tileYDivisor * 2),
        1
    );
    state_.vplTileDeferredCullingStage2->SynchronizeCompute();

    state_.vplTileDeferredCullingStage2->Unbind();

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

void RendererBackend::ComputeVirtualPointLightGlobalIllumination_(const std::vector<std::pair<LightPtr, double>>& perVPLDistToViewer) {
    if (perVPLDistToViewer.size() == 0) return;

    glDisable(GL_DEPTH_TEST);
    BindShader_(state_.vplGlobalIllumination.get());
    state_.vpls.vplGIFbo.Bind();

    // Set up infinite light color
    const glm::vec3 lightColor = frame_->csc.worldLight->GetLuminance();
    state_.vplGlobalIllumination->SetVec3("infiniteLightColor", lightColor);

    state_.vplGlobalIllumination->SetInt("numTilesX", frame_->viewportWidth  / state_.vpls.tileXDivisor);
    state_.vplGlobalIllumination->SetInt("numTilesY", frame_->viewportHeight / state_.vpls.tileYDivisor);

    // All relevant rendering data is moved to the GPU during the light cull phase
    state_.vpls.vplData.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);
    state_.vpls.vplVisiblePerTile.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);
    state_.vpls.vplShadowMaps.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 11);

    state_.vplGlobalIllumination->SetMat4("invProjectionView", frame_->invProjectionView);
    state_.vplGlobalIllumination->BindTexture("screen", state_.lightingColorBuffer);
    state_.vplGlobalIllumination->BindTexture("gDepth", state_.currentFrame.depth);
    state_.vplGlobalIllumination->BindTexture("gNormal", state_.currentFrame.normals);
    state_.vplGlobalIllumination->BindTexture("gAlbedo", state_.currentFrame.albedo);
    state_.vplGlobalIllumination->BindTexture("gBaseReflectivity", state_.currentFrame.baseReflectivity);
    state_.vplGlobalIllumination->BindTexture("gRoughnessMetallicAmbient", state_.currentFrame.roughnessMetallicAmbient);
    state_.vplGlobalIllumination->BindTexture("ssao", state_.ssaoOcclusionBlurredTexture);

    state_.vplGlobalIllumination->SetVec3("fogColor", frame_->fogColor);
    state_.vplGlobalIllumination->SetFloat("fogDensity", frame_->fogDensity);

    const Camera& camera = frame_->camera.get();
    state_.vplGlobalIllumination->SetVec3("viewPosition", camera.GetPosition());
    state_.vplGlobalIllumination->SetInt("viewportWidth", frame_->viewportWidth);
    state_.vplGlobalIllumination->SetInt("viewportHeight", frame_->viewportHeight);

    RenderQuad_();
    
    UnbindShader_();
    state_.vpls.vplGIFbo.Unbind();

    BindShader_(state_.vplGlobalIlluminationBlurring.get());
    state_.vpls.vplGIBlurredFbo.Bind();
    state_.vplGlobalIlluminationBlurring->BindTexture("screen", state_.lightingColorBuffer);
    state_.vplGlobalIlluminationBlurring->BindTexture("indirectIllumination", state_.vpls.vplGIColorBuffer);

    RenderQuad_();

    UnbindShader_();
    state_.vpls.vplGIBlurredFbo.Unbind();

    state_.lightingFbo.CopyFrom(state_.vpls.vplGIBlurredFbo, BufferBounds{0, 0, frame_->viewportWidth, frame_->viewportHeight}, BufferBounds{0, 0, frame_->viewportWidth, frame_->viewportHeight}, BufferBit::COLOR_BIT, BufferFilter::NEAREST);
}

void RendererBackend::RenderScene() {
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

    std::vector<std::pair<LightPtr, double>> perLightDistToViewer;
    // This one is just for shadow-casting lights
    std::vector<std::pair<LightPtr, double>> perLightShadowCastingDistToViewer;
    std::vector<std::pair<LightPtr, double>> perVPLDistToViewer;
    std::vector<int> visibleVplIndices;

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
    lighting->SetVec3("fogColor", frame_->fogColor);
    lighting->SetFloat("fogDensity", frame_->fogDensity);
    RenderQuad_();
    state_.lightingFbo.Unbind();
    UnbindShader_();
    state_.finalScreenBuffer = state_.lightingFbo; // state_.lightingColorBuffer;

    // If world light is enabled perform VPL Global Illumination pass
    if (frame_->csc.worldLight->GetEnabled() && frame_->globalIlluminationEnabled) {
        // Handle VPLs for global illumination (can't do this earlier due to needing position data from GBuffer)
        PerformVirtualPointLightCullingStage2_(perVPLDistToViewer, visibleVplIndices);
        ComputeVirtualPointLightGlobalIllumination_(perVPLDistToViewer);
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

static glm::vec2 GetJitterForIndex(const size_t index, const float width, const float height) {
    glm::vec2 jitter(haltonSequence[index].first, haltonSequence[index].second);
    // Halton numbers are from [0, 1] so we convert this to an appropriate +/- subpixel offset
    jitter = ((jitter - glm::vec2(0.5f)) / glm::vec2(width, height)) * 2.0f;

    return jitter;
}

void RendererBackend::RenderForwardPassPbr_() {
    // Make sure to bind our own frame buffer for rendering
    state_.currentFrame.fbo.Bind();

    BindShader_(state_.geometry.get());

    glm::vec2 jitter(0.0f);

    if (frame_->taaEnabled) {
        jitter = GetJitterForIndex(currentHaltonIndex_, float(frame_->viewportWidth), float(frame_->viewportHeight));
    }

    state_.geometry->SetVec2("jitter", jitter);

    // Begin geometry pass
    glEnable(GL_DEPTH_TEST);

    Render_(*state_.geometry.get(), frame_->visibleInstancedDynamicPbrMeshes, true);
    Render_(*state_.geometry.get(), frame_->visibleInstancedStaticPbrMeshes, true);

    state_.currentFrame.fbo.Unbind();

    UnbindShader_();
}

void RendererBackend::RenderForwardPassFlat_() {
    BindShader_(state_.forward.get());

    glm::vec2 jitter(0.0f);

    // if (frame_->taaEnabled) {
    //     jitter = GetJitterForIndex(currentHaltonIndex_, float(frame_->viewportWidth), float(frame_->viewportHeight));
    // }

    state_.forward->SetVec2("jitter", jitter);

    glEnable(GL_DEPTH_TEST);
    Render_(*state_.forward.get(), frame_->visibleInstancedFlatMeshes, false);

    UnbindShader_();
}

void RendererBackend::PerformPostFxProcessing_() {
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    //glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    PerformBloomPostFx_();

    PerformAtmosphericPostFx_();

    PerformGammaTonemapPostFx_();

    // Needs to come after gamma correction + tonemapping
    PerformFxaaPostFx_();

    PerformTaaPostFx_();

    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void RendererBackend::PerformBloomPostFx_() {
    // We use this so that we can avoid a final copy between the downsample and blurring stages
    std::vector<PostFXBuffer> finalizedPostFxFrames(state_.numDownsampleIterations + state_.numUpsampleIterations);
   
    Pipeline* bloom = state_.bloom.get();
    BindShader_(bloom);

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
            bloom->BindTexture("mainTexture", state_.finalScreenBuffer.GetColorAttachments()[0]);
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
            bloom->BindTexture("bloomTexture", state_.lightingColorBuffer);
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
    if (!frame_->fxaaEnabled) return;

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
    if (!frame_->taaEnabled) return;

    glViewport(0, 0, frame_->viewportWidth, frame_->viewportHeight);

    BindShader_(state_.taa.get());

    state_.taaFbo.fbo.Bind();

    state_.taa->BindTexture("screen", state_.finalScreenBuffer.GetColorAttachments()[0]);
    state_.taa->BindTexture("prevScreen", state_.previousFrameBuffer.GetColorAttachments()[0]);
    state_.taa->BindTexture("velocity", state_.currentFrame.velocity);

    RenderQuad_();

    state_.taaFbo.fbo.Unbind();

    UnbindShader_();

    state_.finalScreenBuffer = state_.taaFbo.fbo;
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

    state_.previousFrameBuffer.CopyFrom(state_.finalScreenBuffer, BufferBounds{ 0, 0, frame_->viewportWidth, frame_->viewportHeight }, BufferBounds{ 0, 0, frame_->viewportWidth, frame_->viewportHeight }, BufferBit::COLOR_BIT, BufferFilter::NEAREST);

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

    GraphicsDriver::SwapBuffers(frame_->vsyncEnabled);

    frame_.reset();
}

void RendererBackend::RenderQuad_() {
    GetMesh(state_.screenQuad, 0)->Render(1, GpuArrayBuffer());
    //_state.screenQuad->GetMeshContainer(0)->mesh->Render(1, GpuArrayBuffer());
}

TextureHandle RendererBackend::CreateShadowMap3D_(uint32_t resolutionX, uint32_t resolutionY, bool vpl) {
    ShadowMap3D smap;
    smap.shadowCubeMap = Texture(TextureConfig{TextureType::TEXTURE_3D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, resolutionX, resolutionY, 0, false}, NoTextureData);
    smap.shadowCubeMap.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    smap.shadowCubeMap.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    // We need to set this when using sampler2DShadow in the GLSL shader
    //smap.shadowCubeMap.setTextureCompare(TextureCompareMode::COMPARE_REF_TO_TEXTURE, TextureCompareFunc::LEQUAL);

    if (vpl) {
        smap.diffuseCubeMap = Texture(TextureConfig{TextureType::TEXTURE_3D, TextureComponentFormat::RGB, TextureComponentSize::BITS_8, TextureComponentType::FLOAT, resolutionX, resolutionY, 0, false}, NoTextureData);
        smap.diffuseCubeMap.SetMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        smap.diffuseCubeMap.SetCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

        smap.frameBuffer = FrameBuffer({smap.diffuseCubeMap, smap.shadowCubeMap});
    }
    else {
        smap.frameBuffer = FrameBuffer({smap.shadowCubeMap});
    }
    
    if (!smap.frameBuffer.Valid()) {
        isValid_ = false;
        return TextureHandle::Null();
    }

    auto& cache = vpl ? vplSmapCache_ : smapCache_;

    TextureHandle handle = TextureHandle::NextHandle();
    cache.shadowMap3DHandles.insert(std::make_pair(handle, smap));

    // These will be resident in GPU memory for the entire life cycle of the renderer
    Texture::MakeResident(smap.shadowCubeMap);
    if (vpl) Texture::MakeResident(smap.diffuseCubeMap);

    return handle;
}

Texture RendererBackend::LookupShadowmapTexture_(TextureHandle handle) const {
    if (handle == TextureHandle::Null()) return Texture();

    // See if it's in the regular cache
    auto it = smapCache_.shadowMap3DHandles.find(handle);
    if (it != smapCache_.shadowMap3DHandles.end()) {
        return it->second.shadowCubeMap;
    }

    // See if it's in the VPL cache
    auto vit = vplSmapCache_.shadowMap3DHandles.find(handle);
    if (vit != vplSmapCache_.shadowMap3DHandles.end()) {
        return vit->second.shadowCubeMap;
    }

    return Texture();
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
    std::vector<GpuTextureHandle> gpuShadowCubeMaps;
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
            gpuShadowCubeMaps.push_back(smap.shadowCubeMap.GpuHandle());
        }
        else {
            gpuLights.push_back(std::move(gpuLight)); 
        }
    }

    state_.nonShadowCastingPointLights.CopyDataToBuffer(0, sizeof(GpuPointLight) * gpuLights.size(), (const void*)gpuLights.data());
    state_.shadowCubeMaps.CopyDataToBuffer(0, sizeof(GpuTextureHandle) * gpuShadowCubeMaps.size(), (const void*)gpuShadowCubeMaps.data());
    state_.shadowCastingPointLights.CopyDataToBuffer(0, sizeof(GpuPointLight) * gpuShadowLights.size(), (const void*)gpuShadowLights.data());

    state_.nonShadowCastingPointLights.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 0);
    state_.shadowCubeMaps.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);
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

    s->SetInt("numLights", int(gpuLights.size()));
    s->SetInt("numShadowLights", int(gpuShadowLights.size()));
    s->SetVec3("viewPosition", c.GetPosition());
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
    s->SetFloat("worldLightAmbientIntensity", frame_->csc.worldLight->GetAmbientIntensity());

    InitCoreCSMData_(s);

    // s->setMat4("cascade0ProjView", &_state.csms[0].projectionView[0][0]);
}

TextureHandle RendererBackend::GetOrAllocateShadowMapHandleForLight_(LightPtr light) {
    auto& cache = GetSmapCacheForLight_(light);
    assert(cache.shadowMap3DHandles.size() > 0);

    auto it = cache.lightsToShadowMap.find(light);
    // If not found, look for an existing shadow map
    if (it == cache.lightsToShadowMap.end()) {
        TextureHandle handle;
        for (const auto & entry : cache.shadowMap3DHandles) {
            if (cache.usedShadowMaps.find(entry.first) == cache.usedShadowMaps.end()) {
                handle = entry.first;
                break;
            }
        }

        if (handle == TextureHandle::Null()) {
            // Evict oldest since we could not find an available handle
            LightPtr oldest = cache.lruLightCache.front();
            cache.lruLightCache.pop_front();
            handle = cache.lightsToShadowMap.find(oldest)->second;
            EvictLightFromShadowMapCache_(oldest);
        }

        SetLightShadowMapHandle_(light, handle);
        AddLightToShadowMapCache_(light);
        return handle;
    }

    // Update the LRU cache
    AddLightToShadowMapCache_(light);
    return it->second;
}

RendererBackend::ShadowMap3D RendererBackend::GetOrAllocateShadowMapForLight_(LightPtr light) {
    auto& cache = GetSmapCacheForLight_(light);
    return cache.shadowMap3DHandles.find(GetOrAllocateShadowMapHandleForLight_(light))->second;
}

void RendererBackend::SetLightShadowMapHandle_(LightPtr light, TextureHandle handle) {
    auto& cache = GetSmapCacheForLight_(light);
    cache.lightsToShadowMap.insert(std::make_pair(light, handle));
    cache.usedShadowMaps.insert(handle);
}

void RendererBackend::EvictLightFromShadowMapCache_(LightPtr light) {
    auto& cache = GetSmapCacheForLight_(light);
    for (auto it = cache.lruLightCache.begin(); it != cache.lruLightCache.end(); ++it) {
        if (*it == light) {
            cache.lruLightCache.erase(it);
            return;
        }
    }
}

bool RendererBackend::ShadowMapExistsForLight_(LightPtr light) {
    auto& cache = GetSmapCacheForLight_(light);
    return cache.lightsToShadowMap.find(light) != cache.lightsToShadowMap.end();
}

void RendererBackend::AddLightToShadowMapCache_(LightPtr light) {
    auto& cache = GetSmapCacheForLight_(light);
    // First remove the existing light entry if it's already there
    EvictLightFromShadowMapCache_(light);
    // Push to back so that it is seen as most recently used
    cache.lruLightCache.push_back(light);
}

void RendererBackend::RemoveLightFromShadowMapCache_(LightPtr light) {
    if ( !ShadowMapExistsForLight_(light) ) return;

    auto& cache = GetSmapCacheForLight_(light);

    // Deallocate its map
    TextureHandle handle = cache.lightsToShadowMap.find(light)->second;
    cache.lightsToShadowMap.erase(light);
    cache.usedShadowMaps.erase(handle);

    // Remove from LRU cache
    EvictLightFromShadowMapCache_(light);
}

RendererBackend::ShadowMapCache& RendererBackend::GetSmapCacheForLight_(LightPtr light) {
    return light->IsVirtualLight() ? vplSmapCache_ : smapCache_;
}
}