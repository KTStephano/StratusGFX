#include "StratusApplicationThread.h"

void EnsureIsApplicationThread() {
    if ( !stratus::ApplicationThread::Instance()->CurrentIsApplicationThread() ) {
        throw std::runtime_error("Must execute on renderer thread");
    }
}

namespace stratus {
    ApplicationThread * ApplicationThread::_instance;

    ApplicationThread::ApplicationThread() 
        // Don't create a new thread context - use current to prevent issues with UI
        : _thread(new Thread("Renderer", false)) {}

    ApplicationThread::~ApplicationThread() {
        _thread.reset();
    }

    bool ApplicationThread::CurrentIsApplicationThread() const {
        return &Thread::Current() == _thread.get();
    }

    void ApplicationThread::_QueueFront(const Thread::ThreadFunction& function) {
        auto ul = _LockWrite();
        _queue.push_front(function);
    }

    void ApplicationThread::_Dispatch() {
        {
            auto ul = _LockWrite();
            _thread->QueueMany(_queue);
            _queue.clear();
        }
        _thread->Dispatch();
    }

    void ApplicationThread::_Synchronize() {
        _thread->Synchronize();
    }

    void ApplicationThread::_DispatchAndSynchronize() {
        _Dispatch();
        _Synchronize();
    }
}