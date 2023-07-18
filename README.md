# ![logo](https://user-images.githubusercontent.com/8399235/254134808-86724ef5-73ed-4a1f-999d-babb74f7942a.jpg)

Release State: **Pre-Release/Beta**

Hobby research rendering engine which was open sourced once its core features were in a presentable state. Expect bugs and instability as it is still under development. Licensed under MPL-2.0.

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

At its core Stratus is a rendering engine first with minimal features found in general purpose engines. Its focus is on modern 3D graphics capabilities. Because of this it has two main use cases:

1) People taking it and using it as a learning resource.

2) Integration into other general purpose engines (new or existing) and modeling tools.

Because of the MPL license, any community changes made to the rendering code will continue to help others in new and existing projects that use it.

# Current Supported Features

### Graphics

* Physically based metallic-roughness pipeline
* Realtime global illumination and indirect shadowing
* Raymarched volumetric lighting
* Cascaded shadow mapping
* Deferred lighting and soft shadowing
* Mesh LOD generation and selection
* GPU frustum culling
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

# Minimum Hardware Requirements

| Type | Minimum |
| --- | --- |
| CPU | Ryzen 3 1200 (quad core) |
| RAM | 8 GB |
| GPU | Nvidia GTX 1050 Ti |
| *Storage (Deploy Binaries) | 700 MB |
| *Storage (Binaries + Source + Dependencies) | 7 GB |
| *Storage (Binaries + Source + Demo Data + Dependencies) | 16 GB |

\* Uncompressed

# Building For Windows & Linux

This code base will currently not work on MacOS. Linux and Windows should both be fine so long as the graphics driver supports OpenGL 4.6 and the compiler supports C++17.

First set up the repo

    git clone --recursive https://github.com/KTStephano/StratusGFX.git
    cd StratusGFX

Build 3rd party dependencies -> should only need to do this once per clone

    python3 ./dependency_build.py

Now build the StratusGFX source

    mkdir build; cd build
    cmake ..
    cmake --build . -j 8 --config RelWithDebInfo

All executables will be put into StratusGFX/Bin. Make sure you run them while inside Bin/. Good ones to run to see if it worked are 

    Ex00_StartupShutdown.exe (runs through initialize, run one frame, shutdown sequence)
    Ex01_StratusGFX.exe (you should see a forest of red cubes since textures aren't bundled with source)
    StratusEngineUnitTests.exe
    StratusEngineIntegrationTests.exe

# First Places to Look

-> [Documentation](https://github.com/KTStephano/StratusGFX/wiki)

You can check [High Level Architecture Overview](https://ktstephano.github.io/rendering/stratusgfx/architecture), or you can start by looking through the code under Examples/ExampleEnv00 and Examples/ExampleEnv01. They both depend on code that is inside of Examples/Common which is another good place to look around.

None of the test scenes are bundled with this source so the rest of the environments will be completely blank when running.

# Running Example Environments 2-6

-> More in depth explanation here: [Examples](https://github.com/KTStephano/StratusGFX/wiki/Examples)

1) A zip file containing Sponza, Interrogation Room, San Miguel, Bistro and Bathroom can be found here: [https://drive.google.com/file/d/1zuxILmOs9fX-w37EB65yOtOZA8em6nnP/view?usp=sharing](https://drive.google.com/file/d/1zuxILmOs9fX-w37EB65yOtOZA8em6nnP/view?usp=sharing)

2) Extract the Resources.zip folder into the root of StratusGFX. It will then be at the same level as Bin/, Examples/, Source/, Tests/. Make sure that the folder structure looks like StratusGFX/Resources/* where * will be folders such as Sponza, Bistro, etc.

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

# Controls For Example Environments

WASD to move

Left mouse to fly up, right mouse to fly down

F toggles camera light

E toggles directional light

G toggles global illumination

R recompiles all shaders

# Future of StratusGFX

### Short Term

-> Improving build system to make it easier for people to get up and running with the code

-> Addition of either TAA or TSSAA to help with image stability while in motion

-> Animation

-> Performance improvements

### Medium Term

-> Order independent transparency

-> Better handling of LOD generation and transition

-> Screen Space Reflections (SSR)

### Long Term

-> Switch backend to Vulkan which will enable the engine to run on MacOS and not just Windows/Linux

-> Addition of baked lighting features so that weaker hardware has a fallback option

-> Addition of other modern global illumination techniques to either replace or complement what is already there
