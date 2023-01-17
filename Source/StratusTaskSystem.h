#pragma once

#include "StratusSystemModule.h"
#include "StratusThread.h"
#include "StratusAsync.h"

#include <mutex>
#include <vector>
#include <cmath>

namespace stratus { 
    // Enables easy access to asynchronous processing by providing its own Task
    // Threads which are used under the hood to support Async<E>.
    class TaskSystem : public SystemModule {
        friend class Engine;

        TaskSystem();

    public:
        TaskSystem(const TaskSystem&) = delete;
        TaskSystem(TaskSystem&&) = delete;
        TaskSystem& operator=(const TaskSystem&) = delete;
        TaskSystem& operator=(TaskSystem&&) = delete;

        virtual ~TaskSystem() {}

        static TaskSystem * Instance() { return _instance; }

        // SystemModule inteface
        virtual const char * Name() const {
            return "TaskSystem";
        }

        virtual bool Initialize();
        virtual SystemStatus Update(const double);
        virtual void Shutdown();

    private:
        template<typename E, typename T>
        Async<E> _ScheduleTask(const T& process) {
            auto ul = std::unique_lock<std::mutex>(_m);
            if (_taskThreads.size() == 0) throw std::runtime_error("Task threads size equal to 0");

            auto as = Async<E>(*_taskThreads[_nextTaskThread].get(), process);
            _nextTaskThread = (_nextTaskThread + 1) % _taskThreads.size();
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
        static TaskSystem * _instance;

        mutable std::mutex _m;
        std::vector<ThreadPtr> _taskThreads;
        size_t _nextTaskThread;
    };
}