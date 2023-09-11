[Info] Thread::(Renderer) stratus::Engine::Initialize:155 -> Engine initializing
[Info] Thread::(Renderer) stratus::EngineModuleInit::InitializeEngineModule:130 -> Initializing InputManager
[Info] Thread::(Renderer) stratus::EngineModuleInit::InitializeEngineModule:130 -> Initializing EntityManager
[Info] Thread::(Renderer) stratus::EngineModuleInit::InitializeEngineModule:130 -> Initializing TaskSystem
[Info] Thread::(Renderer) stratus::TaskSystem::Initialize:25 -> Started TaskSystem with 12 threads
[Info] Thread::(Renderer) stratus::EngineModuleInit::InitializeEngineModule:130 -> Initializing MaterialManager
[Info] Thread::(Renderer) stratus::EngineModuleInit::InitializeEngineModule:130 -> Initializing ResourceManager
[Info] Thread::(Renderer) stratus::EngineModuleInit::InitializeEngineModule:130 -> Initializing Window
[Info] Thread::(Renderer) stratus::Window::Initialize:69 -> Initializing SDL video
[Info] Thread::(Renderer) stratus::Window::Initialize:76 -> Initializing SDL window
[Info] Thread::(Renderer) stratus::GraphicsDriver::Initialize:93 -> Creating OpenGL context
[Info] Thread::(Renderer) stratus::PrintGLInfo:21 ->
==================== OpenGL Information ====================
        Renderer: NVIDIA GeForce GTX 1060 6GB/PCIe/SSE2
        Version: 4.6.0 NVIDIA 536.23
        Major, minor Version: 4, 6
        Max anisotropy: 16
        Max draw buffers: 8
        Max combined textures: 192
        Max cube map texture size: 32768
        Max fragment uniform vectors: 1024
        Max fragment uniform components: 4096
        Max varying floats: 124
        Max render buffer size: 32768
        Max texture image units: 32
        Max texture size 1D: 32768
        Max texture size 2D: 32768x32768
        Max texture size 3D: 16384x16384x16384
        Max texture size cube map: 32768x32768
        Max vertex attribs: 16
        Max vertex uniform vectors: 1024
        Max vertex uniform components: 4096
        Max viewport dims: (32768, 32768)

        ==> Compute Information
        Max compute shader storage blocks: 16
        Max compute uniform blocks: 14
        Max compute uniform texture image units: 32
        Max compute uniform components: 2048
        Max compute atomic counters: 16384
        Max compute atomic counter buffers: 8
        Max compute work group invocations: 1536
        Max compute work group count: 2147483647x65535x65535
        Max compute work group size: 1536x1024x64

        ==> Virtual/Sparse Texture Information
        GL_RGBA8
                Supports sparse (virtual) textures 2D: true
                Num sparse (virtual) page sizes 2D: 1
                Preferred page size X 2D: 128
                Preferred page size Y 2D: 128
                Supports sparse (virtual) textures 3D: true
                Num sparse (virtual) page sizes 3D: 1
                Preferred page size X 3D: 32
                Preferred page size Y 3D: 32
                Preferred page size Z 3D: 16
        GL_RGBA16
                Supports sparse (virtual) textures 2D: true
                Num sparse (virtual) page sizes 2D: 1
                Preferred page size X 2D: 128
                Preferred page size Y 2D: 64
                Supports sparse (virtual) textures 3D: true
                Num sparse (virtual) page sizes 3D: 1
                Preferred page size X 3D: 32
                Preferred page size Y 3D: 16
                Preferred page size Z 3D: 16
        GL_RGBA32F
                Supports sparse (virtual) textures 2D: true
                Num sparse (virtual) page sizes 2D: 1
                Preferred page size X 2D: 64
                Preferred page size Y 2D: 64
                Supports sparse (virtual) textures 3D: true
                Num sparse (virtual) page sizes 3D: 1
                Preferred page size X 3D: 16
                Preferred page size Y 3D: 16
                Preferred page size Z 3D: 16
        GL_DEPTH_COMPONENT
                Supports sparse (virtual) textures 2D: true
                Num sparse (virtual) page sizes 2D: 1
                Preferred page size X 2D: 128
                Preferred page size Y 2D: 128
                Supports sparse (virtual) textures 3D: true
                Num sparse (virtual) page sizes 3D: 1
                Preferred page size X 3D: 32
                Preferred page size Y 3D: 32
                Preferred page size Z 3D: 16
        GL_DEPTH_STENCIL
                Supports sparse (virtual) textures 2D: true
                Num sparse (virtual) page sizes 2D: 1
                Preferred page size X 2D: 128
                Preferred page size Y 2D: 128
                Supports sparse (virtual) textures 3D: true
                Num sparse (virtual) page sizes 3D: 1
                Preferred page size X 3D: 32
                Preferred page size Y 3D: 32
                Preferred page size Z 3D: 16
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 587202560
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 41943040
[Info] Thread::(Renderer) stratus::EngineModuleInit::InitializeEngineModule:130 -> Initializing RendererFrontend
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/depth.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/depth.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/pbr_geometry_pass.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/pbr_geometry_pass.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/flat_forward_pass.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/flat_forward_pass.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/skybox.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/skybox.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/skybox.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/skybox.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/gammaTonemap.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/gammaTonemap.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/shadow.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/shadow.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/shadow.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/shadowVpl.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/pbr.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/pbr.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/pbr.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/pbr.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/bloom.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/bloom.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/csm.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/ssao.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/ssao.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/ssao.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/ssao_blur.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/atmospheric.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/atmospheric.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/atmospheric_postfx.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/atmospheric_postfx.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/viscull_vpls.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/viscull_vsm.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vsm_clear.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vpl_light_color.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vpl_tiled_deferred_culling_stage1.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vpl_tiled_deferred_culling_stage2.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vpl_pbr_gi.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vpl_pbr_gi.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vpl_pbr_gi.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vpl_pbr_gi_denoise.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fxaa.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fxaa_luminance.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fxaa.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fxaa_smoothing.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/taa.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/taa.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/aabb_draw.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/aabb_draw.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fullscreen.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fullscreen.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fullscreen.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fullscreen_pages.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fullscreen.vs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/fullscreen_page_groups.fs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/viscull_point_lights.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vsm_analyze_depth.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/vsm_mark_pages.cs
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 4, 4, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RED, BITS_16, FLOAT, 64, 64, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_CUBE_MAP_ARRAY, DEPTH, BITS_16, FLOAT, 256, 256, 200, 0, 0}
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/viscull_lods.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/viscull_lods.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/viscull_csms.cs
[Info] Thread::(Renderer) stratus::Pipeline::Compile_:226 -> Loading shader: ../Source/Shaders/update_model_transforms.cs
[Info] Thread::(Renderer) stratus::EngineModuleInit::InitializeEngineModule:130 -> Initializing SanMiguel
[Info] Thread::(Renderer) SanMiguel::Initialize:47 -> Initializing SanMiguel
[Info] Thread::(Renderer) stratus::Engine::Initialize:169 -> Initialization complete
[Info] Thread::(TaskThread#1) stratus::ResourceManager::LoadModel_:782 -> Attempting to load model: ../Resources/San_Miguel/san-miguel-low-poly.glb
[Info] Thread::(Renderer) stratus::RendererBackend::RecalculateCascadeData_:375 -> Regenerating Cascade Data
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/Skyboxes/learnopengl/sbox_right.jpg (handle = Handle{21})
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D_ARRAY, RED, BITS_32, FLOAT, 8192, 8192, 1, 0, 1}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RED, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RG, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_RECTANGLE, RGBA, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RG, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RED, BITS_16, UINT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, DEPTH, BITS_DEFAULT, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RED, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/Skyboxes/learnopengl/sbox_left.jpg (handle = Handle{21})
TextureConfig{TEXTURE_2D, RG, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_RECTANGLE, RGBA, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RG, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RED, BITS_16, UINT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, DEPTH, BITS_DEFAULT, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, DEPTH, BITS_DEFAULT, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_RECTANGLE, RED, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_RECTANGLE, RED, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/Skyboxes/learnopengl/sbox_top.jpg (handle = Handle{21})
TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RED, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RED, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGB, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RED, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_RECTANGLE, RED, BITS_16, FLOAT, 800, 450, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/Skyboxes/learnopengl/sbox_bottom.jpg (handle = Handle{21})
TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 800, 450, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 800, 450, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 800, 450, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 400, 225, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 400, 225, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 400, 225, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 200, 112, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 200, 112, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 200, 112, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 100, 56, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 100, 56, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 100, 56, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 50, 28, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 50, 28, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 50, 28, 0, 0, 0}
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/Skyboxes/learnopengl/sbox_front.jpg (handle = Handle{21})
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 25, 14, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 25, 14, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 25, 14, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 50, 28, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 100, 56, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 200, 112, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 400, 225, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/Skyboxes/learnopengl/sbox_back.jpg (handle = Handle{21})
TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 800, 450, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_8, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_2D, RGBA, BITS_16, FLOAT, 1600, 900, 0, 0, 0}
[Info] Thread::(TaskThread#4) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(TaskThread#6) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group[Info] Thread::(TaskThread#7) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(TaskThread#3) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(TaskThread#9) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(TaskThread#12) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(TaskThread#11) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group

[Info] Thread::(TaskThread#5) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(TaskThread#8) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(TaskThread#10) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed:dik,u eik, olz 12582912, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TextureConfig{TEXTURE_CUBE_MAP, RGB, BITS_DEFAULT, UINT_NORM, 2048, 2048, 0, 1, 0}
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:354.095 (2.8241 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:174.59 (5.7277 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:437.044 (2.2881 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:366.099 (2.7315 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:386.429 (2.5878 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:415.369 (2.4075 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:407.432 (2.4544 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:348.529 (2.8692 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:389.454 (2.5677 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:372.121 (2.6873 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:465.463 (2.1484 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:366.676 (2.7272 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:432.32 (2.3131 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:379.824 (2.6328 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:407.017 (2.4569 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:481.696 (2.076 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:375.009 (2.6666 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:443.754 (2.2535 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:389.029 (2.5705 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:418.445 (2.3898 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:325.119 (3.0758 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:361.925 (2.763 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:474.091 (2.1093 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:359.183 (2.7841 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:456.246 (2.1918 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:430.367 (2.3236 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:435.123 (2.2982 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:374.434 (2.6707 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:440.199 (2.2717 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:340.53 (2.9366 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:322.581 (3.1 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:292.535 (3.4184 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:339.121 (2.9488 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:479.916 (2.0837 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:480.769 (2.08 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:357.961 (2.7936 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:364.312 (2.7449 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:481.974 (2.0748 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:474.518 (2.1074 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:361.468 (2.7665 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:468.384 (2.135 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:477.327 (2.095 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:471.209 (2.1222 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:454.071 (2.2023 ms)
[Info] Thread::(TaskThread#1) stratus::ResourceManager::LoadModel_:823 -> IMPORTED: 3134
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#0
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#1
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#2
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#3
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#4
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#5
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#6
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#7
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#8
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#9
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#10
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#11
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#12
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#13
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#14
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#15
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#16
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#17
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#18
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#19
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#20
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#21
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#22
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#23
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#24
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#25
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#26
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#27
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#28
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#29
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#30
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#31
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#32
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#33
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#34
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#35
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#36
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#37
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#38
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#39
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#40
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#41
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#42
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#43
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#44
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#45
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#46
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#47
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#48
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#49
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#50
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#51
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#52
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#53
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#54
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#55
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#56
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#57
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#58
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#59
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#60
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#61
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#62
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#63
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#64
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#65
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#66
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#67
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#68
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#69
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#70
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:426.021 (2.3473 ms)
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#71
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#72
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#73
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#74
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#75
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#76
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#77
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#78
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#79
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#80
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#81
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#82
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#83
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#84
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#85
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#86
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#87
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#88
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#89
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#90
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#91
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#92
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#93
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#94
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#95
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#96
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#97
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#98
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#99
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#100
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#101
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#102
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#103
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#104
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#105
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#106
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#107
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#108
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#109
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#110
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#111
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#112
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#113
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#114
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#115
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#116
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#117
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#118
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#119
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#120
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#121
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#122
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#123
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#124
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#125
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#126
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#127
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#128
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#129
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#130
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#131
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#132
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#133
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#134
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#135
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#136
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#137
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#138
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#139
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#140
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#141
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#142
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#143
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#144
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#145
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#146
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#147
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#148
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#149
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#150
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#151
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#152
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#153
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#154
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#155
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#156
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#157
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#158
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#159
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#160
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#161
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#162
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#163
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#164
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#165
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#166
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#167
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#168
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#169
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#170
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#171
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#172
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#173
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#174
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#175
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#176
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#177
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#178
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#179
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#180
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#181
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#182
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#183
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#184
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#185
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#186
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#187
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#188
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#189
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#190
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#191
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#192
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#193
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#194
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#195
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#196
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#197
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#198
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#199
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#200
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#201
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#202
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#203
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#204
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#205
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#206
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#207
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#208
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#209
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#210
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#211
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#212
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#213
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#214
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#215
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#216
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#217
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#218
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#219
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#220
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#221
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#222
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#223
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#224
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#225
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#226
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#227
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#228
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#229
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#230
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#231
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#232
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#233
[Info] Thread::(TaskThread#1) stratus::MaterialManager::CreateMaterial:183 -> Attempting to create material: ../Resources/San_Miguel/san-miguel-low-poly.glb#234
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#0]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:410.425 (2.4365 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*0 (handle = Handle{90})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*1 (handle = Handle{91})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*2 (handle = Handle{92})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::([Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*14 (handle = Handle{93})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*15 (handle = Handle{94})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#232]
Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 540000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#232]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#231]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#230]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#230]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#230]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 1000, 180, 0, 1, 0}
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 3452964, 4
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*262 (handle = Handle{96})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#230]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*261 (handle = Handle{97})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*263 (handle = Handle{95})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 700, 525, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1200, 216, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#1]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#2]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#2]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 512, 512, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#232]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#232]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#232]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 512, 512, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> [Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*3 (handle = Handle{98})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 270000, 1
Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#232]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#2]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 536400, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#2]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#229]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 298, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*260 (handle = Handle{100})
[Info][Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*11 (handle = Handle{99})
 Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#228]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#228]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#2]
[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*258 (handle = Handle{101})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#2]
[Info] Thread::(TaskThread#7) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*259 (handle = Handle{102})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 9712800, 3
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#228]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#228]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#228]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#228]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#2]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#2]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#228]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#228]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1200, 1199, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#228]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#228]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1200, 1199, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#2]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#3]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#3]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 600, 600, 0, 1, 0}[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2652864, 3
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*4 (handle = Handle{103})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*5 (handle = Handle{104})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#227]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#227]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#3]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 512, 512, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#227]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 512, 512, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1034400, 2
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*6 (handle = Handle{105})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 431, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 400, 431, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#226]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 222000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#226]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 185, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#226]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#226]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#225]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#226]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#226]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#226]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*236 (handle = [Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*235 (handle = Handle{106})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [Handle{107})
../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 432000, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 240, 0, 1, 0}../Resources/San_Miguel/san-miguel-low-poly.glb#207]

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 300, 240, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#219]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#219]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*249 (handle = Handle{108})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#219]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#219]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*250 (handle = Handle{109})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#224]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#224]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#224][Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*256 (handle = Handle{111})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*257 (handle = Handle{110})

[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1458000, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#224]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#221]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#221]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#224]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#4]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 500, 486, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#221]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 500, 486, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*252 (handle = Handle{112})
../Resources/San_Miguel/san-miguel-low-poly.glb#221]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1398000, 2
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*253 (handle = Handle{113})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#221]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#221]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 500, 466, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#222]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#223]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 500, 466, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*254 (handle = Handle{114})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1518000, 2
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*255 (handle = Handle{115})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#219]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#220]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#218]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#217]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 500, 506, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#4]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 500, 506, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1458000, 2
[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*247 (handle = Handle{118})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*251 (handle = Handle{116})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#7) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*246 (handle = Handle{119})
[Info] Thread::([Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 500, 506, 0, 1, 0}
Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#216]
TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*248 (handle = Handle{117})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#213]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#215]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 500, 466, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#214]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*245 (handle = Handle{120})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1861500, 3
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*244 (handle = Handle{122})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*242 (handle = Handle{121})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 500, 457, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#7]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#7]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 500, 457, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 327, 500, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*8 (handle = Handle{125})
[Info] Thread::(TaskThread#7) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*9 (handle = Handle{126})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*7 (handle = Handle{124})
[Info] Thread::(TaskThread#8) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*10 (handle = Handle{127})
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> [Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1191000, 2
Attempting to load texture from file: ../Resources/San_Miguel/*243 (handle = Handle{123})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#211]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#210]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 500, 457, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 500, 337, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*239 (handle = Handle{129})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*240 (handle = Handle{128})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 6856968, 6
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#208]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 256, 256, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 703, 1000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 665, 1000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 760, 507, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 400, 267, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 5925000, 3
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*237 (handle = Handle{130})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#209]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 645, 1000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1000, 665, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 665, 1000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1995000, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*238 (handle = Handle{131})
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1000, 665, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2175000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1000, 725, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#206]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*234 (handle = Handle{132})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 321600, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 268, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*232 (handle = Handle{133})
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*233 (handle = Handle{134})[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [
../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1787400, 2
../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#7]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#7]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 496, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 497, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*18 (handle = Handle{135})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 320400, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 267, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#204]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:318.492 (3.1398 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info][Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*230 (handle = Handle{136})
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*231 (handle = Handle{137})
 Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1002000, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 500, 334, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 500, 334, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*24 (handle = Handle{138})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 661500, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#7]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 700, 315, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#198]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> [Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*221 (handle = Handle{140})
Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*220 (handle = Handle{139})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 477600, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 400, 398, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 477600, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#202]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 398, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*229 (handle = Handle{142})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*228 (handle = Handle{141})
../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 480000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#196]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#201]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*227 (handle = Handle{145})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*226 (handle = Handle{144})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 480000, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*38 (handle = Handle{143})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1248000, 3
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#200]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 48, 2000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*224 (handle = Handle{146})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*225 (handle = Handle{147})
../Resources/San_Miguel/san-miguel-low-poly.glb#195]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#195]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*217 (handle = Handle{148})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#195]
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*218 (handle = Handle{149})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 960000, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#195]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#195]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#195]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#195]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#195]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> [Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#195]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#195]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1957500, 2
../Resources/San_Miguel/san-miguel-low-poly.glb#199]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 800, 360, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 900, 405, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*223 (handle = Handle{151})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*222 (handle = Handle{150})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 480000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#10]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 480000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#193]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#212]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*241 (handle = Handle{153})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*215 (handle = Handle{152})
../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#192]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#192]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*214 (handle = Handle{155580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#192]
})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*213 (handle = Handle{154})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1069200, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#192]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#192]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 594, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2745000, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#176]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 665, 1000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 500, 500, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1078200, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*191 (handle = Handle{156})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 599, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*192 (handle = Handle{157})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1740000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#177]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#189]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:306.72 (3.2603 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#205]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 800, 725, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*208 (handle = Handle{158})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#191]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*211 (handle = Handle{159})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*212 (handle = Handle{160})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info][Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*186 (handle = Handle{161})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
 Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1440000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 800, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2408400, 3
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 594, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 600, 594, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#45]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*53 (handle = Handle{162})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#190]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*210 (handle = Handle{164})
[Info] Thread::(TaskThread#1) [Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1137600, 1
stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*209 (handle = Handle{163})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#203]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 800, 474, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1080000, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 600, 600, 0, 1, 0}../Resources/San_Miguel/san-miguel-low-poly.glb#9]

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1080000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#203]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*188 (handle = Handle{165})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#173]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#175]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#175]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#175]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*190 (handle = Handle{166})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#175]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#175]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#34]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 320000, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*37 (handle = Handle{168})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*36 (handle = Handle{167})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1080000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 600, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1080000, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*187 (handle = Handle{169})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 270000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#130]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*145 (handle = Handle{170})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 390000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 130, 1000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#233]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#233]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*264 (handle = Handle{171})
../Resources/San_Miguel/san-miguel-low-poly.glb#197]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*219 (handle = Handle{172})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1080000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#50]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1056000, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*62 (handle = Handle{174})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*142 (handle = Handle{173})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*63 (handle = Handle{175})
../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#23]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1000, 352, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1363700, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#23]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 700, 497, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1043700, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#23]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 700, 497, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:452.284 (2.211 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#188]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#194]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#194]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*216 (handle = Handle{178})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*207 (handle = Handle{177})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*206 (handle = Handle{176})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#187]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#6]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2250000, 2
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*205 (handle = Handle{179})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 1000, 270, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 800, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 810000, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#170]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#170]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#170]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#170]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#170]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#170]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#170]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1000, 270, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#186]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 4902000, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*185 (handle = Handle{180})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#186]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#186]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#168]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#168]
[Info] Thread::([Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#168]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 817, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#168]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#168]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*204 (handle = Handle{181})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*183 (handle = Handle{182})
../Resources/San_Miguel/san-miguel-low-poly.glb#168]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#168]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#168]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*184 (handle = Handle{183})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#186]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#186]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#186]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#186]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 3145728, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#186]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#168]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#185]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1024, 1024, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*203 (handle = Handle{184})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#160]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#160]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#184]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#164]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#164]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*175 (handle = Handle{185})
[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*179 (handle = Handle{188})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*201 (handle = Handle{186})
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*202 (handle = Handle{[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#164]
187})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 196608, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8][Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 ->
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 256, 256, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 4084400, 4
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#159]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#159]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#159]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#159]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#183]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#161]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 453, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#161]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 158, 1300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 121, 1000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#182]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 242, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#167]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*200 (handle = Handle{191})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#7) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*182 (handle = Handle{194})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*174 (handle = Handle{189})
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*199 (handle = Handle{190})
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*176 (handle = Handle{192})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 91800, 1
[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*198 (handle = Handle{193})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 102, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2427318, 4
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#181]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#165]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#165]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 303, 302, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 207, 800, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 207, 800, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 483, 600, 0, 1, 0}
[Info][Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*180 (handle = Handle{197})
 Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#158]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*197 (handle = Handle{196})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 135000, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*196 (handle = Handle{195})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*173 (handle = Handle{198})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#180]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 150, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#180]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 4158504, 1
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*195 (handle = Handle{199})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#180]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#162]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#162]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1338, 1036, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2339700, 3
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*177 (handle = Handle{200})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#179]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#166]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 100, 3676, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 100, 3676, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#155]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 149, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#178]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*194 (handle = Handle{201})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 109032, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*181 (handle = Handle{202})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#178]
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*170 (handle = Handle{203})
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#178]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 88, 413, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#159][Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 49152, 1

[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*193 (handle = Handle{204})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#159]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#159]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 128, 128, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1953792, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#154]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#154]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#154]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 465, 1024, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#154]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#154]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 128, 128, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#154]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 6516000, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*169 (handle = Handle{205})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#157]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info][Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 ->  Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 1086, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*172 (handle = Handle{207})
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*151 (handle = Handle{206})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 6150675, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 975, 1983, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 219, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 689152, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 128, 128, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#156]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#156]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#156]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*171 (handle = Handle{208})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#156]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#156]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#156]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#156]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1048576, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#153]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 512, 512, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#153]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#153]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*168 (handle = Handle{209})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#153]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#153]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 67500, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#153]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#153]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#153]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#153]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 75, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#165]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#165]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#153]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#164]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#164]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#164]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#164]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#9]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#175]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#175]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:231.734 (4.3153 ms)
../Resources/San_Miguel/san-miguel-low-poly.glb#175]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#163]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#163]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*178 (handle = Handle{210})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 454400, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#150]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#150]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#150]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 284, 400, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*165 (handle = Handle{211})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 320000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#171]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#162]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#8]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#152]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#152]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*167 (handle = Handle{212})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info][Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
 Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#169]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#169]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#161]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#161]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*12 (handle = Handle{213})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*13 (handle = Handle{214})
../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 270000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 270000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 300, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#11]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#13]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*17 (handle = Handle{216})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*16 (handle = Handle{215})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 196608, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#151]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 256, 256, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#149]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 196608, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*166 (handle = Handle{217})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 256, 256, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#148]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*164 (handle = Handle{218})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#147]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*163 (handle = Handle{219})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*162 (handle = Handle{220})
../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#146]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*161 (handle = Handle{221})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#145]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#144]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*160 (handle = Handle{222})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#144]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#144]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#144]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#144]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#144]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#144]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 6861532, 2
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*159 (handle = Handle{223})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#144]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#144]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 998, 2078, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 48768, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*158 (handle = Handle{224})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 127, 128, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 360000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 150, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#143]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#142]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*157 (handle = Handle{225})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#141]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 47250, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 63, 250, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*156 (handle = Handle{226})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#140]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*155 (handle = Handle{227})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#139]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 135000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#138]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 150, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12][Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*154 (handle = Handle{228})

[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1589496, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*153 (handle = Handle{229})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#234]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 618, 643, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#137]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*265 (handle = Handle{230})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 320000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#12]TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}

[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1835008, 2
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*152 (handle = Handle{231})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 256, 1024, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 512, 512, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#127]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*74 (handle = Handle{232})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:385.46 (2.5943 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#136]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*141 (handle = Handle{233})
../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 480000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#12]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*19 (handle = Handle{234})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#16]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#16]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#17]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 810000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 450, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#16]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#16]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*147 (handle = Handle{235})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#16]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 270000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#16]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*140 (handle = Handle{236})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:321.854 (3.107 ms)
../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 67500, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 75, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#15]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#16]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#16]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#14]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#18]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*20 (handle = Handle{237})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*21 (handle = Handle{238})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#128]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#128]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#129]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*143 (handle = Handle{239})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#129]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 235200, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*144 (handle = Handle{240})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 196, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*189 (handle = Handle{241})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1093632, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#174]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 512, 512, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 192, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2266112, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 397, 1024, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#174]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#134]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*149 (handle = Handle{242})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#128]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#128]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2602800, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#125]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1800, 482, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#133]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*148 (handle = Handle{243})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 5016000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#128]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#128]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#128]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 836, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#131]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*146 (handle = Handle{244})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#130]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#130]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#129]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 3739575, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#129]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#132]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#127]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2095, 595, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:427.058 (2.3416 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#126]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#135]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
Attempting to load texture from file: ../Resources/San_Miguel/*150 (handle = Handle{245})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#125]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#113]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#124]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*128 (handle = Handle{246})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*139 (handle = Handle{247})
../Resources/San_Miguel/san-miguel-low-poly.glb#120]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#121]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#123]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#119]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 320000, 1
[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*136 (handle = Handle{249})
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*135 (handle = Handle{248})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#118]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
Attempting to load texture from file: ../Resources/San_Miguel/*138 (handle = Handle{250})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 347250, 3
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*132 (handle = Handle{253})
[Info] Thread::(TaskThread#7) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*134 (handle = Handle{251})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 50, 200, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*133 (handle = Handle{252})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#117]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 63, 250, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 4997100, 4
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#117]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 664, 2000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117][Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 ->
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 59, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 7980000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 1330, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#117]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*131 (handle = Handle{254})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116][Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 580800, 1

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 363, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#116]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*130 (handle = Handle{255})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 561600, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 351, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#115]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#114]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#114]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#114]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*129 (handle = Handle{256})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#114]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*126 (handle = Handle{257})
../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 580800, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 363, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 270000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*127 (handle = Handle{258})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112][Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 612800, 1

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#122]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 383, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*121 (handle = Handle{260})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*137 (handle = Handle{259})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#111]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 4356000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 726, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#112]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:354.56 (2.8204 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#111]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*125 (handle = Handle{261})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 411200, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 257, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#110]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#108]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#108]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#108]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#108]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#108]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*123 (handle = Handle{262})[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 49152, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*124 (handle = Handle{263})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 128, 128, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_../Resources/San_Miguel/san-miguel-low-poly.glb#109]
:79 -> Texture data bytes processed: 96000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 120, 200, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#109]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#108]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#108]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#108]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#108]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#103]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#107]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*118 (handle = Handle{264})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*122 (handle = Handle{265})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 47250, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#106]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#105]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 63, 250, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#104]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#104]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*120 (handle = Handle{266})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*119 (handle = Handle{267})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#104]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#103]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#103]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#103]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#103]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 524288, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#103]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#103]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#102]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 256, 512, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#102]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*117 (handle = Handle{268})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#19]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#101]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#101][Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 ->
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*116 (handle = Handle{269})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#99]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*71 (handle = Handle{270})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 8632000, 2
[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*114 (handle = Handle{272})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*113 (handle = Handle{271})
../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#98]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 1332, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98][Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 ->
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*112 (handle = Handle{274})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*111 (handle = Handle{273})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2484000, 3
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 503, 700, 0, 1, 0}[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 363, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 735000, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#100]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 350, 350, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 350, 350, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> [Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*115 (handle = Handle{275})
Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#21]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#22]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*22 (handle = Handle{276})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#22]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*23 (handle = Handle{277})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2278800, 2
../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 422, 900, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#86]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 422, 900, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*99 (handle = Handle{278})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 7992000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#88]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#90]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#99]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#87]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#98]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 1332, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 360000, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*101 (handle = Handle{279})
580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*103 (handle = Handle{280})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*100 (handle = Handle{281})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 150, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#98]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#93]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#84]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*97 (handle = Handle{283})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*106 (handle = Handle{282})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*108 (handle = Handle{284})
../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 995200, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 222, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#22]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#22]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*110 (handle = Handle{285})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 98304, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 128, 256, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#85]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 10278000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#83]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 1713, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#81]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*96[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*94 (handle = Handle{288})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*98 (handle = Handle{286})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [ (handle = Handle{287})
../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 631296, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#97]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#79]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#79]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 76, 1024, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#82]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 270000, 1
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*92 (handle = Handle{289})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#96]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#80]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 300, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#80]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 311296, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*95 (handle = Handle{290})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*109 (handle = Handle{291})
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*93 (handle = Handle{292})
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:114.494 (8.7341 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> [Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#80]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 76, 1024, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 480000, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#80]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#80]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#80]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#80]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#80]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#80]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2892000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#95]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 482, 2000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#95]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#22]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#22]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#81]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#81]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#94]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#94]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#94]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*107 (handle = Handle{293})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#94]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#94]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#94]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#94]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#80]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1454080, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#80]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#92]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#79]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#73]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 355, 1024, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 7992000, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*85 (handle = Handle{295})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*88 (handle = Handle{296})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*105 (handle = Handle{294})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 1332, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1139800, 3
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 250, 500, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 200, 208, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*84 (handle = Handle{297})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#81]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 47250, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#81]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#75]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#75]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#70]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 63, 250, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#70]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*82 (handle = Handle{299})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*87 (handle = Handle{298})
../Resources/San_Miguel/san-miguel-low-poly.glb#91]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#22]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#89]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 131072, 1
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*104 (handle = Handle{300})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 ->  -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 128, 256, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*102 (handle = Handle{301})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 960000, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2502027, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1397, 597, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#76]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#24]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#25]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#26]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*25 (handle = Handle{302})
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*26 (handle = Handle{303})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 36300, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 110, 110, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 786432, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*27 (handle = Handle{304})
../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#27]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 512, 512, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:359.221 (2.7838 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1080000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#27]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 600, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#27]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#75]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#70]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#71]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#74]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*83 (handle = Handle{305})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*86 (handle = Handle{306})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#77]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 240000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 200, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*89 (handle = Handle{307})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1190400, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 800, 496, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#69]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*81 (handle = Handle{308})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#68]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*80 (handle = Handle{309})
../Resources/San_Miguel/san-miguel-low-poly.glb#67]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 452800, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#69]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#69]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#68]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 283, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#68]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*79 (handle = Handle{310})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 580800, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#66]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#66]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#66]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#66]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 363, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#66]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*78 (handle = Handle{311})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 8295600, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#66]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#66]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#66]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#66]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#65]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#65]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 1332, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#64]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 253, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 303600, 1
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*76 (handle = Handle{313})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*77 (handle = Handle{312})
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 253, 400, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 369152, 2
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*72 (handle = Handle{314})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 128, 128, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 92400, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 88, 350, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#63]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#63]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#63]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#63]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#63]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*75 (handle = Handle{315})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#63]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#62]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#61]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#61]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#61]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#61]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*73 (handle = Handle{316})[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#61]

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#61]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*69 (handle = Handle{317})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 640000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#60]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 284800, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 178, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#60]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#58]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:277.054 (3.6094 ms)
../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#56]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#57]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#57]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*70 (handle = Handle{318})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#57]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 49152, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#59]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#59]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#59]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 128, 128, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#59]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#59]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#59]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#59]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#59]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#54]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#55]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*67 (handle = Handle{319})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#55]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*68 (handle = Handle{320})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#54]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#54]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#53]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#78]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*66 (handle = Handle{321})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1345344, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 429, 784, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*91 (handle = Handle{323})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1588896, 2
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*90 (handle = Handle{322})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*65 (handle = Handle{324})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 412, 799, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 146, 466, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 5492700, 3
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#52]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#51]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2000, 453, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#51]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 2000, 453, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 63, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*64 (handle = Handle{325})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#173]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1800000, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#49]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#49]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#49]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#49]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#49]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#48]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 300, 1500, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#48]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#48][Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*59 (handle = Handle{329})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*58 (handle = Handle{328})
[Info] Thread::([Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*60 (handle = Handle{326})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 128700, 1

TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*61 (handle = Handle{327})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#48]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 143, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 214600, 3
../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 99, 200, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 99, 200, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 95, 200, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*57 (handle = Handle{331})
[Info] Thread::([Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 57000, 1
TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*56 (handle = Handle{330})
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 95, 200, 0, 1, 0}
../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 76000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#47]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 95, 200, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#47]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*55 (handle = Handle{333})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*54 (handle = Handle{332})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 90000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> 580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 60, 500, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 90000, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info][Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
 Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 60, 500, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#46]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*50 (handle = Handle{335})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*51 (handle = Handle{334})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 89250, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 119, 250, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 336800, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#43]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 421, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*52 (handle = Handle{336})[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:327.15 (3.0567 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 336800, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 421, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*49 (handle = Handle{337})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 336800, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 421, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#172]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#44]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#43]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#42]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*47 (handle = Handle{338})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*48 (handle = Handle{339})
../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 332100, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#41]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 246, 450, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 261600, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 218, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:413.07 (2.4209 ms)
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#41]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#40]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#72]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*46 (handle = Handle{340})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#39]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#38]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*44 (handle = Handle{341})
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*45 (handle = Handle{342})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#37]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#36][Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*43 (handle = Handle{344})
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*42 (handle = Handle{343})

[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1481400, 2
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#35]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#34]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#34]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#34]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1000, 301, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#34]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#34]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 800, 241, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 2025000, 1
../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*41 (handle = Handle{346})
[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*40 (handle = Handle{345})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*39 (handle = Handle{347})
../Resources/San_Miguel/san-miguel-low-poly.glb#33]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#33]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 1500, 450, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#32]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 11126736, 4
[Info] Thread::(TaskThread#7) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*35 (handle = Handle{348})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#32]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#31]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#31]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#23]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 2048, 1269, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#23]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#30]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#29]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#28]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 1500, 450, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 565, 600, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 48, 2000, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(TaskThread#3) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*33 (handle = Handle{350})
../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#2) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*34 (handle = Handle{349})
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1587600, 2
[Info] Thread::(TaskThread#6) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*30 (handle = Handle{353})
[Info] Thread::(TaskThread#8) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*29 (handle = Handle{355})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#5) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*32 (handle = Handle{352})
[Info] Thread::(TaskThread#7) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*28 (handle = Handle{354})
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#4) stratus::ResourceManager::LoadTexture_:906 -> Attempting to load texture from file: ../Resources/San_Miguel/*31 (handle = Handle{351})
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 660, 700, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 168, 300, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 3223600, 6
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> ../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 421, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, SRGB_ALPHA, BITS_DEFAULT, UINT_NORM, 200, 421, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::([Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 400, 400, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 400, 271, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [[Info] Thread::(Renderer../Resources/San_Miguel/san-miguel-low-poly.glb#5]
) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, RGB, BITS_DEFAULT, UINT_NORM, 800, 527, 0, 1, 0}[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]

[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncTextureData_:79 -> Texture data bytes processed: 1264800, 1
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) stratus::TextureImpl::TextureImpl:209 -> [Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
TextureConfig{TEXTURE_2D, SRGB, BITS_DEFAULT, UINT_NORM, 800, 527, 0, 1, 0}
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(Renderer) WorldLightController::HandleInput:66 -> World Lighting Toggled
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#5]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#21]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#22]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#20]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#21]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#22]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(TaskThread#1) stratus::ProcessMaterial:580 -> Loading Mesh Material [../Resources/San_Miguel/san-miguel-low-poly.glb#207]
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:277.755 (3.6003 ms)
[Info] Thread::(TaskThread#1) stratus::ResourceManager::LoadModel_:876 -> Model loaded [../Resources/San_Miguel/san-miguel-low-poly.glb] with [3129] meshes
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:747.328 (1.3381 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:622.123 (1.6074 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:751.597 (1.3305 ms)
[Info] Thread::(TaskThread#1) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 2 as a task group
[Info] Thread::(Renderer) SanMiguel::Update:176 -> SPAWNED 0 VPLS
[Info] Thread::(TaskThread#1) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#12) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#4) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#9) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#2) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#5) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#8) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#6) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#10) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#7) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(TaskThread#3) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2352 bytes of mesh data: 2 meshes
[Info] Thread::(TaskThread#11) stratus::ResourceManager::ClearAsyncModelData_::<lambda_6addbf40b7e80e0b51a4b84505850e1c>::operator ():158 -> Processing 3129 as a task group
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:220.24 (4.5405 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:293.72 (3.4046 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:202.274 (4.9438 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:246.871 (4.0507 ms)
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2230424 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2275728 bytes of mesh data: 18 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2124640 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2138024 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2103864 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2147488 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2254392 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2106384 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2167032 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2123184 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2148664 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2110864 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2129400 bytes of mesh data: 10 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2124920 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2245488 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 83886080
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2112096 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2099888 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2175992 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2108344 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2186128 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2204048 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2098992 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2248848 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2154208 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2372944 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2123016 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2207184 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2246048 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2224768 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2098824 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2115344 bytes of mesh data: 19 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 125829120
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2172744 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2098320 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2249016 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2135952 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2234624 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2169888 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2101008 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2251816 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2171232 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2105208 bytes of mesh data: 20 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2101792 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2263912 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:64.8799 (15.4131 ms)
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2114840 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2120216 bytes of mesh data: 18 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2177000 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 167772160
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2351944 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2172912 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2172632 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2105320 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2274552 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2357096 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2180640 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2155720 bytes of mesh data: 18 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2196320 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2097592 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2166864 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2166528 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2133992 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2245432 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 209715200
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2128056 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2111200 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2162272 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2143624 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2202480 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2185176 bytes of mesh data: 18 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2129400 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2215864 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2113832 bytes of mesh data: 19 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2277072 bytes of mesh data: 18 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2230816 bytes of mesh data: 18 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2114784 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2098712 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2148664 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 251658240
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2200016 bytes of mesh data: 18 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2216312 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2161376 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2140936 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2128952 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2223200 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2157120 bytes of mesh data: 19 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2104928 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2114616 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2124696 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2100336 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2139592 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2099048 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2175544 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2229024 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2142392 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 293601280
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2181536 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2170056 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2231992 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2197552 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2155104 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2238768 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2098488 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2209200 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2283288 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2209144 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2397192 bytes of mesh data: 9 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2264472 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2233336 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2118032 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2151464 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2199512 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2223704 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 335544320
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2111704 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2106496 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2173416 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2196824 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2166360 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2242184 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2215808 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2270800 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2114448 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2274272 bytes of mesh data: 18 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2193016 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2127216 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2182992 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2208752 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2179968 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 377487360
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2168992 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2103920 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2189656 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2288104 bytes of mesh data: 22 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2118984 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2149056 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2225048 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2133712 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2184504 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2164344 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2181536 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2192568 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2098320 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2150008 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2279928 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2230704 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2176216 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 419430400
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2142504 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2282616 bytes of mesh data: 19 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2243248 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2160928 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:68.7739 (14.5404 ms)
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2121616 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2191112 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2120776 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2195648 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2187360 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2173024 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2230872 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2141944 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2214464 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2173528 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2269120 bytes of mesh data: 17 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2282448 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2324560 bytes of mesh data: 9 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 461373440
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2117080 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2192624 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2209088 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2217712 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2156392 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2237592 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2276064 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2189208 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2202816 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2189320 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2172968 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2130352 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2164400 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2240504 bytes of mesh data: 10 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2242072 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2256464 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2219168 bytes of mesh data: 18 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2296560 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 503316480
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2135616 bytes of mesh data: 9 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2190272 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2143624 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2337384 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2227064 bytes of mesh data: 7 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2193408 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2125368 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2127440 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2273152 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2145864 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2156056 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2358944 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2178176 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2255288 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2390696 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2225888 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2313808 bytes of mesh data: 9 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2360848 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2179016 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2235464 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2225720 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2161376 bytes of mesh data: 16 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 545259520
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2184672 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2109072 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2218496 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2177448 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2175432 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2143736 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2322208 bytes of mesh data: 15 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2205168 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2186184 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2329936 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2291688 bytes of mesh data: 9 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2197776 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2211608 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2135000 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2267048 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2119768 bytes of mesh data: 11 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2143624 bytes of mesh data: 9 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2314592 bytes of mesh data: 10 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2327864 bytes of mesh data: 9 meshes
[Info] Thread::(Renderer) stratus::GpuMeshAllocator::Resize_:566 -> Resizing: 587202560
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2260720 bytes of mesh data: 10 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2324448 bytes of mesh data: 10 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2169328 bytes of mesh data: 10 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2218440 bytes of mesh data: 10 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2207072 bytes of mesh data: 9 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2130240 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2406264 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2153200 bytes of mesh data: 7 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2314480 bytes of mesh data: 8 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2207744 bytes of mesh data: 7 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2130016 bytes of mesh data: 7 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2117528 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2142504 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2123408 bytes of mesh data: 14 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 2232048 bytes of mesh data: 12 meshes
[Info] Thread::(Renderer) stratus::ResourceManager::ClearAsyncModelData_:113 -> Processed 1775872 bytes of mesh data: 13 meshes
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:136.634 (7.3188 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:163.824 (6.1041 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:176.438 (5.6677 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:192.086 (5.206 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:256.377 (3.9005 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:243.659 (4.1041 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:237.79 (4.2054 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:199.025 (5.0245 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:126.613 (7.8981 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:183.429 (5.4517 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:175.085 (5.7115 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:231.198 (4.3253 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:232.202 (4.3066 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:225.347 (4.4376 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:159.365 (6.2749 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:125.957 (7.9392 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:115.062 (8.691 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:135.788 (7.3644 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:128.388 (7.7889 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:127.812 (7.824 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:141.527 (7.0658 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:148.736 (6.7233 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:132.961 (7.521 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:162.33 (6.1603 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:136.153 (7.3447 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:107.327 (9.3173 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:103.922 (9.6226 ms)
[Info] Thread::(Renderer) stratus::RendererBackend::ClearRemovedLightData_:881 -> Cleared 1 lights this frame
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:100.342 (9.9659 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:109.833 (9.1047 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:137.291 (7.2838 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:111.572 (8.9628 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:126.509 (7.9046 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:116.349 (8.5948 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:136.31 (7.3362 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:145.756 (6.8608 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:142.485 (7.0183 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:134.551 (7.4321 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:119.61 (8.3605 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:155.521 (6.43 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:160.741 (6.2212 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:162.138 (6.1676 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:159.396 (6.2737 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:163.578 (6.1133 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:229.737 (4.3528 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:286.402 (3.4916 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:219.351 (4.5589 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:233.193 (4.2883 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:129.266 (7.736 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:124.377 (8.0401 ms)
[Info] Thread::(Renderer) stratus::RendererBackend::ClearRemovedLightData_:881 -> Cleared 1 lights this frame
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:129.525 (7.7205 ms)
[Info] Thread::(Renderer) WorldLightController::HandleInput:66 -> World Lighting Toggled
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:128.171 (7.8021 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:119.584 (8.3623 ms)
[Info] Thread::(Renderer) WorldLightController::HandleInput:66 -> World Lighting Toggled
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:142.286 (7.0281 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:153.534 (6.5132 ms)
[Info] Thread::(Renderer) WorldLightController::HandleInput:66 -> World Lighting Toggled
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:114.57 (8.7283 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:102.057 (9.7984 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:163.457 (6.1178 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:133.912 (7.4676 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:181.064 (5.5229 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:150.611 (6.6396 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:165.728 (6.034 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:123.364 (8.1061 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:152.339 (6.5643 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:211.109 (4.7369 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:139.897 (7.1481 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:187.01 (5.3473 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:108.893 (9.1833 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:110.476 (9.0517 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:99.7546 (10.0246 ms)
[Info] Thread::(Renderer) WorldLightController::HandleInput:66 -> World Lighting Toggled
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:138.839 (7.2026 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:150.161 (6.6595 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:198.539 (5.0368 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:173.72 (5.7564 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:168.967 (5.9183 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:169.684 (5.8933 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:185.505 (5.3907 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:152.488 (6.5579 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:169.823 (5.8885 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:167.574 (5.9675 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:170.681 (5.8589 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:195.267 (5.1212 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:198.255 (5.044 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:233.013 (4.2916 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:221.749 (4.5096 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:215.703 (4.636 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:227.144 (4.4025 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:207.555 (4.818 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:154.012 (6.493 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:167.898 (5.956 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:246.828 (4.0514 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:270.307 (3.6995 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:218.141 (4.5842 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:240.784 (4.1531 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:259.794 (3.8492 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:243.368 (4.109 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:190.88 (5.2389 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:182.772 (5.4713 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:204.578 (4.8881 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:183.11 (5.4612 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:211.104 (4.737 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:266.071 (3.7584 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:206.629 (4.8396 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:199.422 (5.0145 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:179.382 (5.5747 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:203.857 (4.9054 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:208.433 (4.7977 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:217.476 (4.5982 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:196.897 (5.0788 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:224.694 (4.4505 ms)
[Info] Thread::(Renderer) SanMiguel::Update:98 -> FPS:227.568 (4.3943 ms)
[Info] Thread::(Renderer) stratus::Engine::ShutDown:231 -> Engine shutting down
[Info] Thread::(Renderer) stratus::TaskSystem::Shutdown:62 -> [1] Waiting on task threads to shutdown ...

C:\Users\ktste\StratusGFX\Bin\Ex04_SanMiguel.exe (process 18664) exited with code 0.
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .