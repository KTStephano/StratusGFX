#pragma once

#include "StratusSystemModule.h"
#include "StratusThread.h"
#include "StratusAsync.h"

#include <mutex>
#include <vector>
#include <cmath>
#include <unordered_map>

namespace stratus { 
    // Allows groups of async processes to be waited on in an async manner
    struct __TaskWait {
        virtual ~__TaskWait() = default;
        virtual bool CheckForCompletion() const = 0;
    };

    template<typename E>
    struct __TaskWaitImpl : public __TaskWait {
        __TaskWaitImpl(const std::function<void(const std::vector<Async<E>>&)>& callback, const std::vector<Async<E>>& group)
            : callback(callback), group(group) {
            thread = &Thread::Current();
        }

        virtual bool CheckForCompletion() const override {
            for (const auto& as : group) {
                if (!as.Completed()) {
                    return false;
                }
            }

            std::function<void(const std::vector<Async<E>>&)> c = callback;
            std::vector<Async<E>> g = group;
            thread->Queue([c, g]() {
                c(g);
            });
            return true;
        }

        Thread* thread;
        std::function<void(const std::vector<Async<E>>&)> callback;
        std::vector<Async<E>> group;
    };

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

            // Enter the threads with their current number of work items in a list
            bool found = false;
            const auto currentId = Thread::Current().Id();
            std::vector<std::pair<ThreadHandle, size_t>> currentWorkLoads;
            for (const auto& entry : _threadToIndexMap) {
                if (entry.first == currentId) continue;
                currentWorkLoads.push_back(std::make_pair(entry.first, _threadsWorking[entry.second]->load()));
            }

            const auto comparison = [](const std::pair<ThreadHandle, size_t>& a, const std::pair<ThreadHandle, size_t>& b) {
                return a.second < b.second;
            };

            // Sort the threads based on how much work they currently have
            std::sort(currentWorkLoads.begin(), currentWorkLoads.end(), comparison);
            
            // Choose the first which should have the least work items
            const size_t index = _threadToIndexMap.find(currentWorkLoads[0].first)->second;

            // Increment the working #
            _threadsWorking[index]->fetch_add(1);

            const auto processWithHook = [this, index, process]() {
                auto result = process();
                // Decrement working counter
                _threadsWorking[index]->fetch_sub(1);
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

        template<typename E>
        void WaitOnTaskGroup(const std::function<void (const std::vector<Async<E>>&)>& callback, const std::vector<Async<E>>& group) {
            auto ul = std::unique_lock<std::mutex>(_m);
            _waiting.push_back(new __TaskWaitImpl<E>(callback, group));
        }

        size_t Size() const {
            return _taskThreads.size();
        }

    private:
        mutable std::mutex _m;
        // The size of the following vectors/maps are immutable after initializing
        std::vector<ThreadPtr> _taskThreads;
        std::unordered_map<ThreadHandle, size_t> _threadToIndexMap;
        // Measures # of work items per thread
        std::vector<std::unique_ptr<std::atomic<size_t>>> _threadsWorking;
        
        // This changes with every call to wait on task group
        //std::vector<std::pair<Thread *, std::vector<
        std::vector<__TaskWait *> _waiting;
        size_t _nextTaskThread;
    };
}