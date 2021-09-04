#include "StratusThread.h"
#include <chrono>
#include <string>

namespace stratus {
    static Thread ** GetCurrentThreadPtr() {
        static thread_local Thread * _current = nullptr;
        return &_current;
    }

    static void NullifyCurrentThread() {
        Thread ** current = GetCurrentThreadPtr();
        *current = nullptr;
    }

    static void SetCurrentThread(Thread * thread) {
        Thread ** current = GetCurrentThreadPtr();
        if (*current != nullptr) {
            throw std::runtime_error("Attempt to overwrite existing thread pointer");
        }
        *current = thread;
    }

    // Used for when the user does not specify their own thread name
    static std::string NextThreadName() {
        static std::atomic<uint64_t> threadId(1);
        return "Thread#" + std::to_string(threadId.fetch_add(1));
    }

    Thread& Thread::Current() {
        Thread ** current = GetCurrentThreadPtr();
        if (*current == nullptr) throw std::runtime_error("stratus::Thread::Current called from a thread not wrapped around stratus::Thread");
        return **current;
    }

    Thread::Thread(bool ownsExecutionContext) : Thread(NextThreadName(), ownsExecutionContext) {}

    Thread::Thread(const std::string& name, bool ownsExecutionContext)
        : _name(name),
          _ownsExecutionContext(ownsExecutionContext),
          _id(ThreadHandle::NextHandle()) {

        if (ownsExecutionContext) {
            _context = std::thread([this]() {
                SetCurrentThread(this);
                while (this->_running.load()) {
                    this->_ProcessNext();
                }
            });
        }
    }

    Thread::~Thread() {
        Dispose();
    }

    void Thread::Dispatch() {
        {
            std::unique_lock<std::mutex> ul(_mutex);

            // If we're still processing a previous dispatch, don't try to process another
            if (_processing.load()) return;
            
            // If nothing to process, return early
            if (_frontQueue.size() == 0) return;

            // Copy contents of front buffer to back buffer for processing
            for (const auto & func : _frontQueue) _backQueue.push_back(func);
            _frontQueue.clear();

            // Signal ready for processing
            _processing.store(true);
        }

        // If we don't own the context, use the current thread
        if (!_ownsExecutionContext) {
            SetCurrentThread(this);
            _ProcessNext();
            NullifyCurrentThread();
        }
    }

    bool Thread::Idle() const {
        return !_processing.load();
    }

    void Thread::Dispose() {
        _running.store(false);
        if (_ownsExecutionContext) _context.join();
    }

    void Thread::Synchronize() const {
        // Wait until processing is complete
        while (_processing.load()) std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }

    void Thread::Queue(const ThreadFunction& func) {
        Queue(std::vector<ThreadFunction>{func});
    }

    void Thread::Queue(const std::vector<ThreadFunction>& funcs) {
        std::unique_lock<std::mutex> ul(_mutex);
        for (auto & func : funcs) _frontQueue.push_back(func);
    }

    void Thread::_ProcessNext() {
        if (_processing.load()) {
            for (const ThreadFunction & func : _backQueue) func();
            _backQueue.clear();
            _processing.store(false); // Signal completion
        }
        else {
            // Sleep a bit to prevent hogging CPU core
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        }
    }

    const std::string& Thread::Name() const {
        return _name;
    }

    const ThreadHandle& Thread::Id() const {
        return _id;
    }
}