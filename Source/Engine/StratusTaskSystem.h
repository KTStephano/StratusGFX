#pragma once

#include "StratusSystemModule.h"
#include "StratusThread.h"
#include "StratusAsync.h"

#include <mutex>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <algorithm>

namespace stratus { 
    // Allows groups of async processes to be waited on in an async manner
    struct TaskWait_ {
        virtual ~TaskWait_() = default;
        virtual bool CheckForCompletion() const = 0;
    };

    template<typename E>
    struct TaskWaitImpl_ : public TaskWait_ {
        TaskWaitImpl_(const std::function<void(const std::vector<Async<E>>&)>& callback, const std::vector<Async<E>>& group)
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
        Async<E> CreateAsyncTask_(const T& process, const size_t index) {
            // Increment the working #
            threadsWorking_[index]->fetch_add(1);

            const auto processWithHook = [this, index, process]() {
                auto result = process();
                // Decrement working counter
                threadsWorking_[index]->fetch_sub(1);
                return result;
            };

            auto as = Async<E>(*taskThreads_[index].get(), processWithHook);
            return as;
        }

        Async<void> CreateAsyncVoidTask_(const std::function<void (void)>& process, const size_t index) {
            // Increment the working #
            threadsWorking_[index]->fetch_add(1);

            const auto processWithHook = [this, index, process]() {
                process();
                // Decrement working counter
                threadsWorking_[index]->fetch_sub(1);
            };

            auto as = Async<void>(*taskThreads_[index].get(), processWithHook);
            return as;
        }

        size_t GetNextThreadIndexForTask_() {
            if (taskThreads_.size() == 0) throw std::runtime_error("Task threads size equal to 0");

            // Enter the threads with their current number of work items in a list
            bool found = false;
            const auto currentId = Thread::Current().Id();
            std::vector<std::pair<ThreadHandle, size_t>> currentWorkLoads;
            currentWorkLoads.reserve(threadToIndexMap_.size());
            for (const auto& entry : threadToIndexMap_) {
                if (entry.first == currentId) continue;
                currentWorkLoads.push_back(std::make_pair(entry.first, threadsWorking_[entry.second]->load()));
            }

            const auto comparison = [](const std::pair<ThreadHandle, size_t>& a, const std::pair<ThreadHandle, size_t>& b) {
                return a.second < b.second;
            };

            // Sort the threads based on how much work they currently have
            std::sort(currentWorkLoads.begin(), currentWorkLoads.end(), comparison);
            
            // Choose the first which should have the least work items
            return threadToIndexMap_.find(currentWorkLoads[0].first)->second;
        }

        template<typename E, typename T>
        Async<E> ScheduleTask_(const T& process) {
            auto ul = std::unique_lock<std::mutex>(m_);

            const auto index = GetNextThreadIndexForTask_();

            return CreateAsyncTask_<E, T>(process, index);
        }

        Async<void> ScheduleVoidTask_(const std::function<void (void)>& process) {
            auto ul = std::unique_lock<std::mutex>(m_);

            const auto index = GetNextThreadIndexForTask_();

            return CreateAsyncVoidTask_(process, index);
        }

    public:
        template<typename E>
        Async<E> ScheduleTask(const std::function<std::shared_ptr<E> (void)>& process) {
            return ScheduleTask_<E>(process);
        }

        template<typename E>
        Async<E> ScheduleTask(const std::function<E * (void)>& process) {
            return ScheduleTask_<E>(process);
        }

        Async<void> ScheduleTask(const std::function<void (void)>& process) {
            return ScheduleVoidTask_(process);
        }

        template<typename E>
        void AddTaskGroupCallback(const std::function<void (const std::vector<Async<E>>&)>& callback, const std::vector<Async<E>>& group) {
            auto ul = std::unique_lock<std::mutex>(m_);
            waiting_.push_back(new TaskWaitImpl_<E>(callback, group));
        }

        size_t Size() const {
            return taskThreads_.size();
        }

    private:
        mutable std::mutex m_;
        // The size of the following vectors/maps are immutable after initializing
        std::vector<ThreadPtr> taskThreads_;
        std::unordered_map<ThreadHandle, size_t> threadToIndexMap_;
        // Measures # of work items per thread
        std::vector<std::unique_ptr<std::atomic<size_t>>> threadsWorking_;
        
        // This changes with every call to wait on task group
        //std::vector<std::pair<Thread *, std::vector<
        std::vector<TaskWait_ *> waiting_;
        size_t nextTaskThread_;
    };
}