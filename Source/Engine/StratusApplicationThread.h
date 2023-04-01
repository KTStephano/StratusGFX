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
            auto ul = LockWrite_();
            for (const Thread::ThreadFunction& function : functions) {
                queue_.push_back(function);
            }
        }

        // Checks if current executing thread is the same as the renderer thread
        bool CurrentIsApplicationThread() const;

    private:
        std::unique_lock<std::mutex> LockWrite_() const { return std::unique_lock<std::mutex>(mutex_); }
        void QueueFront_(const Thread::ThreadFunction&);
        void Dispatch_();
        void Synchronize_();
        void DispatchAndSynchronize_();

    private:
        mutable std::mutex mutex_;
        std::list<Thread::ThreadFunction> queue_;
        std::unique_ptr<Thread> thread_;
    };
}