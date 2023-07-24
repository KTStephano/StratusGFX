# ![logo](https://user-images.githubusercontent.com/8399235/254135954-84b9cb9a-b01e-43e0-8a3c-16217af4432e.jpg)

Release State: **Pre-Release/Beta**

Engine Version: 0.10

Realtime 3D rendering engine. Expect bugs and instability as it is still under development. The engine is free and open source (licensed under MPL-2.0). Please feel free to contact me about any questions or issues you have!

![sponza](https://ktstephano.github.io/assets/v0.10/SponzaGI_Front.png)

(3D Model: Intel Sponza)

![sanmiguel](https://ktstephano.github.io/assets/v0.10/FinalAfterPostProcessing.png)

(3D Model: San Miguel)

![cornell_front](https://ktstephano.github.io/assets/v0.10/Cornell_Front.png)

![cornell_back](https://ktstephano.github.io/assets/v0.10/Cornell_Back.png)

(3D Model: Cornell Box)

-> [Video Feature Reel](https://www.youtube.com/watch?v=dj0wVxwd1ng)

-> [Graphics Image Feature Reel](https://ktstephano.github.io/rendering/stratusgfx/feature_reel)

-> [High Level Architecture Overview](https://ktstephano.github.io/rendering/stratusgfx/architecture)

-> [How A Frame Is Rendered](https://ktstephano.github.io/rendering/stratusgfx/frame_analysis_v0_10)

# Purpose

This project was created as a hobby research project for learning low level engine development and implementing modern graphics techniques.

# Use Cases

At its core Stratus is a rendering engine first with minimal features found in general purpose engines. Its focus is on modern 3D graphics capabilities. Because of this it has two main use cases:

1) People taking it and using it as a learning resource.

2) Integration into other general purpose engines (new or existing) and modeling tools.

Because of the MPL license, any community changes made to the rendering code will continue to help others in new and existing projects that use it.

# Current Supported Features

### Graphics

* Physically based metallic-roughness pipeline
* Realtime global illumination
* Spatiotemporal image denoising
* Raymarched volumetric lighting and shadowing
* Cascaded shadow mapping
* Deferred lighting
* Mesh LOD generation and selection
* GPU Frustum Culling
* Screen Space Ambient Occlusion (SSAO)
* Reinhard or ACES Tonemapping
* Fog
* Bloom
* Fast Approximate Anti-Aliasing (FXAA)
* Temporal Anti-Aliasing (TAA)

### Engine

* Pool allocators
* GPU memory allocators/managers
* Multi threaded utilities
* Concurrent hash map
* Entity-Component System (ECS)
* Logging

### Modern graphics API features used

* Compute shaders
* Direct state access
* Programmable vertex pulling
* Multi draw elements indirect
* Shader storage buffer

# Minimum Hardware Requirements

| Type | Minimum |
| --- | --- |
| CPU | Ryzen 3 1200 (quad core) |
| RAM | 8 GB |
| GPU | Nvidia GTX 1050 Ti |

# Building For Windows & Linux

This code base will currently not work on MacOS. Linux and Windows should both be fine so long as the graphics driver supports OpenGL 4.6 and the compiler supports C++17.

First set up the repo

    git clone --recursive https://github.com/KTStephano/StratusGFX.git
    cd StratusGFX
    git submodule update

Build 3rd party dependencies -> should only need to do this once per clone

    python3 ./dependency_build.py

Now build the StratusGFX source

    cmake -Bbuild -S. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=OFF
    cmake --build build/ -j 8

All executables will be put into StratusGFX/Bin. Make sure you run them while inside Bin/. Good ones to run to see if it worked are 

    Ex00_StartupShutdown.exe (runs through initialize, run one frame, shutdown sequence)
    Ex01_StratusGFX.exe (you should see a forest of red cubes since textures aren't bundled with source)
    StratusEngineUnitTests.exe
    StratusEngineIntegrationTests.exe

# First Places to Look

You can check [High Level Architecture Overview](https://ktstephano.github.io/rendering/stratusgfx/architecture), or you can start by looking through the code under Examples/ExampleEnv00 and Examples/ExampleEnv01. They both depend on code that is inside of Examples/Common which is another good place to look around.

None of the test scenes are bundled with this source so the rest of the environments will be completely blank when running.

# Running Example Environments 2-7

When starting the examples will take a bit to load/process. You can see the status in the console. After starting you may experience some temporary frame drops but this will stabilize quickly.

-> More in depth explanation here: [Examples](https://github.com/KTStephano/StratusGFX/wiki/Examples)

1) A zip file containing Sponza, Interrogation Room, San Miguel, Bistro and Bathroom can be found here: [https://drive.google.com/file/d/1m56T8cWMwAOHTAIbxAn-891trehgJpzH/view?usp=drive_link](https://drive.google.com/file/d/1m56T8cWMwAOHTAIbxAn-891trehgJpzH/view?usp=drive_link)

2) Extract the Resources.zip folder into the root of StratusGFX. It will then be at the same level as Bin/, Examples/, Source/, Tests/. Make sure that the folder structure looks like StratusGFX/Resources/* where * will be folders such as Sponza, Bistro, etc.

3) Change directory into Bin/ and run the example environments.

Example environment 01 will still be a forest of red cubes since its textures and models aren't part of the bundle.

Credits for the 3D assets used in the examples/demos are as follows:

[Crytek Sponza](https://sketchfab.com/3d-models/abandoned-warehouse-1e40d433ed6f48fb880a0d2172aff7ca)

[Intel Sponza](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html)

[San Miguel](https://casual-effects.com/data/)

[Bistro](https://developer.nvidia.com/orca/amazon-lumberyard-bistro)

[Junk Shop](https://cloud.blender.org/p/gallery/5dd6d7044441651fa3decb56)

[The Bathroom](https://sketchfab.com/3d-models/the-bathroom-free-d5e5035dda434b8d9beaa7271f1c85fc)

[Interrogation Room](https://sketchfab.com/3d-models/interogation-room-6e9151ec29494469a74081ddc054d569)

[Abandoned Warehouse](https://sketchfab.com/3d-models/abandoned-warehouse-1e40d433ed6f48fb880a0d2172aff7ca)

# Controls For Example Environments

WASD to move

Left mouse to fly up, right mouse to fly down

U unlocks look up/look down for camera

F toggles camera light

E toggles directional light

G toggles global illumination

R recompiles all shaders
