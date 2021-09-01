//
// Created by stephano on 2/17/19.
//

#ifndef STRATUSGFX_ENGINE_H
#define STRATUSGFX_ENGINE_H

#include "StratusCommon.h"
//#include "Renderer.h"

namespace stratus {
// Contains everything the engine needs to perform first-time init
struct EngineInitParams {
    int           numCmdArgs;
    const char ** cmdArgs;
};

struct EngineStatistics {
    // Each frame update increments this by 1
    uint64_t currentFrame = 0;
    // Records the time the last frame took to complete - 16.0/1000.0 = 60 fps for example
    double lastFrameTimeSeconds;
    // Average over all previous frames since engine init
    double totalAverageFrameTimeSeconds;
};

// Engine class which handles initializing all core engine subsystems and helps 
// keep everything in sync during frame updates
class Engine {
    // Performs first-time engine and subsystem init
    Engine(const EngineInitParams &);

public:
    // Performs first-time start up and then begins the main loop
    friend void EngineMain(const int numArgs, const char ** args);

    // Global engine instance
    static Engine * Instance() { return _instance; }

    // Checks if the engine has completed its init phase
    bool IsInitializing() const;
    // True if the engine is performing final shutdown sequence
    bool IsShuttingDown() const;
    // Returns how many frames the engine has processed since first start
    uint64_t FrameCount() const;
    // Useful functions for checking current and average frame delta seconds
    double LastFrameTimeSeconds() const;
    double TotalAverageFrameTimeSeconds() const;

    // Begins shutdown sequence for engine and all core subsystems
    void ShutDown();
    // Processes the next full system frame, including rendering. Returns false only
    // if the main engine loop should stop.
    bool Frame();

private:
    // Global engine instance - should only be set by EngineMain function
    static Engine * _instance;
    EngineStatistics _stats;
};

}

#endif //STRATUSGFX_ENGINE_H
