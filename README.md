# StratusGFX / Stratus Engine

![sponza](Sponza.png)
(Intel Sponza rendered in StratusGFX)

I worked on this project in my spare time and I would consider its current state to still be pre-release/beta-release. Expect bugs and instability. Current versions have been set to the MPL 2.0 license.

-> [Feature Reel](https://ktstephano.github.io/rendering/stratusgfx/feature_reel)

-> [High Level Architecture Overview](https://ktstephano.github.io/rendering/stratusgfx/architecture)

-> [How A Frame Is Rendered](https://ktstephano.github.io/rendering/stratusgfx/frame_analysis)

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