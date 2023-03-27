#pragma once

#include "StratusSystemModule.h"
#include "StratusThread.h"
#include "StratusAsync.h"

#include <mutex>
#include <vector>
#include <cmath>
#include <unordered_map>

namespace stratus { 
    // Enables easy access to asynchronous processing by providing its own Task
    // Threads which are used under the hood to support Async<E>.
    SYSTEM_MODULE_CLASS(TaskSystem)
        TaskSystem(const TaskSystem&) = delete;
        TaskSystem(TaskSystem&&) = delete;
        TaskSystem& operator=(const TaskSystem&) = delete;
        TaskSystem& operator=(TaskSystem&&) = delete;

        virtual ~TaskSystem() {}

    private:
        virtual bool Initialize();
        virtual SystemStatus Update(const double);
        virtual void Shutdown();

    private:
        template<typename E, typename T>
        Async<E> _ScheduleTask(const T& process) {
            auto ul = std::unique_lock<std::mutex>(_m);
            if (_taskThreads.size() == 0) throw std::runtime_error("Task threads size equal to 0");

            // Try to find an open thread
            size_t index;
            bool found = false;
            for (const auto& entry : _threadToIndexMap) {
                if (_threadsWorking[entry.second] == 0) {
                    index = entry.second;
                    found = true;
                    break;
                }
            }

            // If we failed to then just use a generic next index
            if (!found) {
                index = _nextTaskThread;
                _nextTaskThread = (_nextTaskThread + 1) % _taskThreads.size();
            }

            // Increment the working #
            _threadsWorking[index] += 1;

            const auto processWithHook = [this, index, process]() {
                auto result = process();
                // Decrement working counter
                _threadsWorking[index] -= 1;
                return result;
            };

            auto as = Async<E>(*_taskThreads[index].get(), processWithHook);
            return as;
        }

    public:
        template<typename E>
        Async<E> ScheduleTask(const std::function<std::shared_ptr<E> (void)>& process) {
            return _ScheduleTask<E>(process);
        }

        template<typename E>
        Async<E> ScheduleTask(const std::function<E* (void)>& process) {
            return _ScheduleTask<E>(process);
        }

    private:
        mutable std::mutex _m;
        // The size of the following vectors/maps are immutable after initializing
        std::vector<ThreadPtr> _taskThreads;
        std::unordered_map<ThreadHandle, size_t> _threadToIndexMap;
        // Measures # of work items per thread
        std::vector<size_t> _threadsWorking;
        size_t _nextTaskThread;
    };
}