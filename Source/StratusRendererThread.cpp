#include "StratusRendererThread.h"

void EnsureIsRendererThread() {
    if ( !stratus::RendererThread::Instance()->CurrentIsRendererThread() ) {
        throw std::runtime_error("Must execute on renderer thread");
    }
}

namespace stratus {
    RendererThread * RendererThread::_instance;

    RendererThread::RendererThread() 
        : _thread(new Thread("Renderer", true)) {}

    RendererThread::~RendererThread() {
        _thread.reset();
    }

    bool RendererThread::CurrentIsRendererThread() const {
        return &Thread::Current() == _thread.get();
    }

    void RendererThread::_QueueFront(const Thread::ThreadFunction& function) {
        auto ul = _LockWrite();
        _queue.push_front(function);
    }

    void RendererThread::_Dispatch() {
        {
            auto ul = _LockWrite();
            _thread->QueueMany(_queue);
            _queue.clear();
        }
        _thread->Dispatch();
    }

    void RendererThread::_Synchronize() {
        _thread->Synchronize();
    }

    void RendererThread::_DispatchAndSynchronize() {
        _Dispatch();
        _Synchronize();
    }
}