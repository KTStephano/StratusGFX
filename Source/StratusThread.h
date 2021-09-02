#pragma once

#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <functional>
#include <vector>

namespace stratus {
    // A stratus thread represents a reusable thread of execution. To use it, small
    // functions should be queued for execution on it, and these functions should have
    // a finite execution time rather than being infinite.
    //
    // To use it, functions are pushed onto the queue using Queue. These are stored
    // until the next call to Dispatch, which happens from outside the thread. This is useful
    // in the sense that the main game loop can keep all thread in-sync to some extent.
    class Thread {
    public:
        typedef std::function<void(void)> ThreadFunction;

        // If ownsExecutionContext is true, a new thread will be created to handle the work
        // at each call to Dispatch. If false, whatever thread calls Dispatch will be used to
        // perform each function.
        Thread(bool ownsExecutionContext);
        ~Thread();

        Thread(const Thread&) = delete;
        Thread(Thread&&) = delete;
        Thread& operator=(const Thread&) = delete;
        Thread& operator=(Thread&&) = delete;

        // Submits a function to be executed upon the next call to Dispatch
        void Queue(const ThreadFunction&);
        void Queue(const std::vector<ThreadFunction>&);
        // Two modes of operation: if ownsExecutionContext was true, functions will be pulled
        // off the list and executed on a private thread. Otherwise, the thread calling Dispatch
        // will be used as the context.
        void Dispatch();
        // Blocks the calling function until all functions from the previous call to Dispatch are complete
        void Synchronize() const;
        // Tells the thread to quit after it finishes executing the last call to Dispatch
        void Dispose();

        // Gets a reference to the underlying Thread object for the current active context it is called from
        static Thread& Current();

        bool operator==(const Thread & other) const { return this == &other; }
        bool operator!=(const Thread & other) const { return !((*this) == other); }

    private:
        void _ProcessNext();

    private:
        // May be empty if ownsExecutionContext is false
        std::thread _context;
        // True if a private thread is used for all function executions, false otherwise
        const bool _ownsExecutionContext;
        // While true the thread can continue servicing calls to Dispatch
        std::atomic<bool> _running{true};
        // List of functions to execute on next call to Dispatch
        std::vector<ThreadFunction> _frontQueue;
        std::vector<ThreadFunction> _backQueue;
        // Protects critical section
        mutable std::mutex _mutex;
        // When true it signals to the dispatch thread that it should begin its next batch of work
        std::atomic<bool> _processing{false};
    };
}