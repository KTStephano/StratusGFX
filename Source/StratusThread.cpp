#include "StratusThread.h"

namespace stratus {
    static Thread ** getCurrentThreadPtr() {
        static thread_local Thread * _current = nullptr;
        return &_current;
    }

    Thread& Thread::Current() {
        Thread ** current = getCurrentThreadPtr();
        if (*current == nullptr) throw std::runtime_error("stratus::Thread::Current called from a thread not wrapped around stratus::Thread");
        return **current;
    }
}