#include "StratusTaskSystem.h"
#include "StratusLog.h"
#include <string>

namespace stratus {
    TaskSystem * TaskSystem::_instance = nullptr;

    TaskSystem::TaskSystem() {}
            
    bool TaskSystem::Initialize() {
        _taskThreads.clear();
        unsigned int concurrency = 1;
        if (std::thread::hardware_concurrency() > 2) {
            concurrency = std::thread::hardware_concurrency() - 1;
        }

        for (unsigned int i = 0; i < concurrency; ++i) {
            Thread * ptr = new Thread("TaskThread#" + std::to_string(i + 1), true);
            _taskThreads.push_back(std::unique_ptr<Thread>(std::move(ptr)));
        }

        _nextTaskThread = 0;

        STRATUS_LOG << "Started " << Name() << " with " << concurrency << " threads" << std::endl;
    }

    SystemStatus TaskSystem::Update(const double) {
        return SystemStatus::SYSTEM_CONTINUE;
    }

    void TaskSystem::Shutdown() {
        _taskThreads.clear();
    }
}