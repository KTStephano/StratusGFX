#include "StratusApplicationThread.h"

void EnsureIsApplicationThread() {
    if ( !stratus::ApplicationThread::Instance()->CurrentIsApplicationThread() ) {
        throw std::runtime_error("Must execute on renderer thread");
    }
}

namespace stratus {
    ApplicationThread::ApplicationThread() 
        // Don't create a new thread context - use current to prevent issues with UI
        : thread_(new Thread("Renderer", false)) {}

    ApplicationThread::~ApplicationThread() {
        thread_.reset();
    }

    bool ApplicationThread::CurrentIsApplicationThread() const {
        return &Thread::Current() == thread_.get();
    }

    void ApplicationThread::QueueFront_(const Thread::ThreadFunction& function) {
        auto ul = LockWrite_();
        queue_.push_front(function);
    }

    void ApplicationThread::Dispatch_() {
        {
            auto ul = LockWrite_();
            thread_->QueueMany(queue_);
            queue_.clear();
        }
        thread_->Dispatch();
    }

    void ApplicationThread::Synchronize_() {
        thread_->Synchronize();
    }

    void ApplicationThread::DispatchAndSynchronize_() {
        Dispatch_();
        Synchronize_();
    }
}