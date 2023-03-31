# StratusGFX / Stratus Engine

I worked on this project in my spare time and I would consider its current state to still be pre-release/beta-release. Expect bugs and instability. Current versions have been set to the MPL 2.0 license.

-> [Graphics Feature Reel](https://ktstephano.github.io/rendering/stratusgfx/feature_reel)

-> [High Level Architecture Overview](https://ktstephano.github.io/rendering/stratusgfx/architecture)

-> [How A Frame Is Rendered](https://ktstephano.github.io/rendering/stratusgfx/frame_analysis)

-> [Video Feature Reel](https://www.youtube.com/watch?v=s5aIsgzwNPE)

(The video feature reel will be redone with latest features including better image smoothing in hopefully the next release)

![Sponza](https://user-images.githubusercontent.com/8399235/229018578-a9ae9609-5378-43de-a909-2ca2661ca2f5.png)
(Intel Sponza rendered with StratusGFX)

# Purpose

This project was created as a hobby research project for learning low level engine development and trying to emulate modern graphics techniques.

# Use Cases

There are two use cases that might apply to this project:

1) Taking it and using it as a learning resource.

2) Integrating it into other more general purpose engines or modeling tools. This is because Stratus is primarily a rendering engine so it should be possible to fold it into other feature complete tools. It could also be used as the rendering backend for new engine projects.

Due to the MPL 2.0 license, any extensions to the rendering code that are made public will enable other projects to benefit from the changes!

# Current Supported Features

### Graphics

* Physically based metallic-roughness pipeline
* Realtime global illumination
* Raymarched volumetric lighting and shadowing
* Cascaded shadow mapping
* Deferred lighting
* Mesh LOD generation and selection
* GPU Frustum Culling
* Screen Space Ambient Occlusion (SSAO)
* Filmic tonemapping
* Fog
* Bloom
* Fast Approximate Anti-Aliasing (FXAA)

### Engine

* Pool allocators
* Multi threaded utilities
* Concurrent hash map
* Entity-Component System (ECS)
* Logging

### Modern graphics API features used

* Compute shaders
* Direct state access
* Programmable vertex pulling
* Multi draw elements indirect
* Shader storage buffers

# Building

This code base will not work on MacOS. Linux and Windows should both be fine so long as the graphics driver supports OpenGL 4.6 and the compiler supports C++17.

### Windows

First install SDL from [https://www.libsdl.org](https://www.libsdl.org)

Somewhere on your hard drive create a folder where you will install dependencies. Set that as an environment variable called SYSROOT.

Next set up the repo

    git clone https://github.com/KTStephano/StratusGFX.git
    git submodule init
    git submodule update

Build Catch2, assimp, and meshoptimizer using cmake. Install their files to ${SYSROOT} under

    bin/
    cmake/
    include/
    lib/
    share/

Now generate the GL3W headers with extensions (--ext)

    cd gl3w
    python3 ./gl3w_gen.py --ext
    cd ../

Now build the source

    mkdir build; cd build
    cmake ..
    cmake --build . --config RelWithDebInfo

All executables will be put into StratusGFX/Bin. Good ones to run to see if it worked are 

    StratusGFX/Bin/Ex00_StartupShutdown.exe (runs through initialize, run one frame, shutdown sequence)
    StratusGFX/Bin/Ex01_StratusGFX.exe (you should see a forest of red cubes since textures aren't bundled with source)
    StratusGFX/Bin/StratusEngineUnitTests.exe
    StratusGFX/Bin/StratusEngineIntegrationTests.exe

### Linux

This should be roughly the same setup as with Windows except you can skip the SYSROOT step and either build from source + install or install the components with a package manager.

# First Places to Look

You can check [High Level Architecture Overview](https://ktstephano.github.io/rendering/stratusgfx/architecture), or you can start by looking through the code under Examples/ExampleEnv00 and Examples/ExampleEnv01. They both depend on code that is inside of Examples/Common which is another good place to look around.

None of the test scenes are bundled with this source so the rest of the environments will be completely blank when running.

# Running Example Environments 2-6

1) A zip file containing Sponza, Interrogation Room, San Miguel, Bistro and Bathroom can be found here: [https://drive.google.com/file/d/1zuxILmOs9fX-w37EB65yOtOZA8em6nnP/view?usp=sharing](https://drive.google.com/file/d/1zuxILmOs9fX-w37EB65yOtOZA8em6nnP/view?usp=sharing)

2) Extract the Resources.zip folder into the root of StratusGFX. It will then be at the same level as Bin/, Examples/, Source/, Tests/. 

3) Change directory into Bin/ and run the example environments.

Example environment 01 will still be a forest of red cubes since its textures and models aren't part of the bundle.

Example environment 07 Warehouse is unavailable since I need to redownload and repackage it due to some data corruption with my copy.

Credits for the 3D assets are as follows:

Sponza: https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html

Bistro: https://developer.nvidia.com/orca/amazon-lumberyard-bistro

San Miguel: https://casual-effects.com/data/

Bathroom: https://sketchfab.com/3d-models/the-bathroom-free-d5e5035dda434b8d9beaa7271f1c85fc
(for the version in this bundle I made a few changes to the model that were not present in the original)

Interrogation Room: https://sketchfab.com/3d-models/interogation-room-6e9151ec29494469a74081ddc054d569

# Future of StratusGFX

### Short Term Goals

-> Addition of either TAA or TSSAA to help with image stability while in motion
-> Performance improvements

### Medium Term Goals

-> Order independent transparency
-> Better handling of LOD generation and transition
-> Screen Space Reflections (SSR)

### Long Term Goals

-> Switch backend to Vulkan which will enable the engine to run on MacOS and not just Windows/Linux
-> Addition of baked lighting features so that weaker hardware has a fallback option
-> Addition of other modern global illumination techniques to either replace or complement what is already there