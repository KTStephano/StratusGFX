#include "StratusTaskSystem.h"
#include "StratusLog.h"
#include <string>

namespace stratus {
    TaskSystem::TaskSystem() {}
            
    bool TaskSystem::Initialize() {
        taskThreads_.clear();
        // Important that this is > 1
        unsigned int concurrency = 2;
        if (std::thread::hardware_concurrency() > concurrency) {
            concurrency = std::thread::hardware_concurrency();
        }

        for (unsigned int i = 0; i < concurrency; ++i) {
            Thread * ptr = new Thread("TaskThread#" + std::to_string(i + 1), true);
            threadsWorking_.push_back(std::unique_ptr<std::atomic<size_t>>(new std::atomic<size_t>(0)));
            threadToIndexMap_.insert(std::make_pair(ptr->Id(), threadsWorking_.size() - 1));
            taskThreads_.push_back(ThreadPtr(std::move(ptr)));
        }

        nextTaskThread_ = 0;

        STRATUS_LOG << "Started " << Name() << " with " << concurrency << " threads" << std::endl;

        return true;
    }

    SystemStatus TaskSystem::Update(const double) {
        for (ThreadPtr& thread : taskThreads_) {
            if (thread->Idle()) {
                thread->Dispatch();
            }
        }

        auto ul = std::unique_lock<std::mutex>(m_);
        if (waiting_.size() == 0) return SystemStatus::SYSTEM_CONTINUE;

        std::vector<TaskWait_ *> waiting;
        for (TaskWait_* wait : waiting_) {
            if (wait->CheckForCompletion()) {
                delete wait;
            }
            else {
                waiting.push_back(wait);
            }
        }

        waiting_ = std::move(waiting);
        
        return SystemStatus::SYSTEM_CONTINUE;
    }

    void TaskSystem::Shutdown() {
        bool allIdle = false;
        
        while (!allIdle) {
            allIdle = true;

            for (auto& thread : taskThreads_) {
                if (!thread->Idle()) {
                    allIdle = false;
                    break;
                }
            }

            if (!allIdle) {
                for (auto& thread : taskThreads_) {
                    thread->Dispatch();
                }
            }
        }

        taskThreads_.clear();
    }
}