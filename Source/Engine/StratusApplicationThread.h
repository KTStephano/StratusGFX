#pragma once

#include "StratusSystemModule.h"
#include "StratusThread.h"
#include <vector>
#include <list>
#include <mutex>

void EnsureIsApplicationThread();
#define CHECK_IS_APPLICATION_THREAD() EnsureIsApplicationThread()

namespace stratus {
    class ApplicationThread {
        friend class Engine;
        ApplicationThread();

        static ApplicationThread *& Instance_() {
            static ApplicationThread * instance = nullptr;
            return instance;
        }

    public:
        static ApplicationThread * Instance() { return Instance_(); }

        virtual ~ApplicationThread();

        // Queue functions
        void Queue(const Thread::ThreadFunction& function) {
            QueueMany<std::vector<Thread::ThreadFunction>>({function});
        }

        template<typename E>
        void QueueMany(const E& functions) {
            auto ul = _LockWrite();
            for (const Thread::ThreadFunction& function : functions) {
                _queue.push_back(function);
            }
        }

        // Checks if current executing thread is the same as the renderer thread
        bool CurrentIsApplicationThread() const;

    private:
        std::unique_lock<std::mutex> _LockWrite() const { return std::unique_lock<std::mutex>(_mutex); }
        void _QueueFront(const Thread::ThreadFunction&);
        void _Dispatch();
        void _Synchronize();
        void _DispatchAndSynchronize();

    private:
        mutable std::mutex _mutex;
        std::list<Thread::ThreadFunction> _queue;
        std::unique_ptr<Thread> _thread;
    };
}