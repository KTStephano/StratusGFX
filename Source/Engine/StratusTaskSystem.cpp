#include "StratusTaskSystem.h"
#include "StratusLog.h"
#include <string>

namespace stratus {
    TaskSystem::TaskSystem() {}
            
    bool TaskSystem::Initialize() {
        _taskThreads.clear();
        unsigned int concurrency = 1;
        if (std::thread::hardware_concurrency() > 2) {
            concurrency = std::thread::hardware_concurrency();
        }

        for (unsigned int i = 0; i < concurrency; ++i) {
            Thread * ptr = new Thread("TaskThread#" + std::to_string(i + 1), true);
            _threadsWorking.push_back(0);
            _threadToIndexMap.insert(std::make_pair(ptr->Id(), _threadsWorking.size() - 1));
            _taskThreads.push_back(ThreadPtr(std::move(ptr)));
        }

        _nextTaskThread = 0;

        STRATUS_LOG << "Started " << Name() << " with " << concurrency << " threads" << std::endl;

        return true;
    }

    SystemStatus TaskSystem::Update(const double) {
        for (ThreadPtr& thread : _taskThreads) {
            if (thread->Idle()) {
                thread->Dispatch();
            }
        }

        auto ul = std::unique_lock<std::mutex>(_m);
        if (_waiting.size() == 0) return SystemStatus::SYSTEM_CONTINUE;

        std::vector<__TaskWait *> waiting;
        for (__TaskWait* wait : _waiting) {
            if (wait->CheckForCompletion()) {
                delete wait;
            }
            else {
                waiting.push_back(wait);
            }
        }

        _waiting = std::move(waiting);
        
        return SystemStatus::SYSTEM_CONTINUE;
    }

    void TaskSystem::Shutdown() {
        _taskThreads.clear();
    }
}