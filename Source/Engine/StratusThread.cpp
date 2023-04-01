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
        : name_(name),
          ownsExecutionContext_(ownsExecutionContext),
          id_(ThreadHandle::NextHandle()) {

        if (ownsExecutionContext) {
            context_ = std::thread([this]() {
                SetCurrentThread(this);
                while (this->running_.load()) {
                    this->ProcessNext_();
                }
            });
        }
    }

    Thread::~Thread() {
        Dispose();
    }

    void Thread::Dispatch() {
        {
            std::unique_lock<std::mutex> ul(mutex_);

            // If we're still processing a previous dispatch, don't try to process another
            if (processing_.load()) return;
            
            // If nothing to process, return early
            if (frontQueue_.size() == 0) return;

            // Copy contents of front buffer to back buffer for processing
            for (const auto & func : frontQueue_) backQueue_.push_back(func);
            frontQueue_.clear();

            // Signal ready for processing
            processing_.store(true);
        }

        // If we don't own the context, use the current thread
        if (!ownsExecutionContext_) {
            SetCurrentThread(this);
            ProcessNext_();
            NullifyCurrentThread();
        }
    }

    void Thread::DispatchAndSynchronize() {
        Dispatch();
        Synchronize();
    }

    bool Thread::Idle() const {
        return !processing_.load();
    }

    void Thread::Dispose() {
        running_.store(false);
        if (ownsExecutionContext_) context_.join();
    }

    void Thread::Synchronize() const {
        // Wait until processing is complete
        while (processing_.load()) std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }

    void Thread::ProcessNext_() {
        if (processing_.load()) {
            for (const ThreadFunction & func : backQueue_) func();
            backQueue_.clear();
            processing_.store(false); // Signal completion
        }
        else {
            // Sleep a bit to prevent hogging CPU core
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        }
    }

    const std::string& Thread::Name() const {
        return name_;
    }

    const ThreadHandle& Thread::Id() const {
        return id_;
    }
}