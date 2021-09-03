#pragma once

#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
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
        Thread(const std::string & name, bool ownsExecutionContext);
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
        // Gets thread name set in constructor (note: not required to be unique)
        const std::string& Name() const;

        // Gets a reference to the underlying Thread object for the current active context it is called from
        static Thread& Current();

        bool operator==(const Thread & other) const { return this == &other; }
        bool operator!=(const Thread & other) const { return !((*this) == other); }

    private:
        void _ProcessNext();

    private:
        // May be empty if ownsExecutionContext is false
        std::thread _context;
        // Friendly name of thread
        const std::string _name;
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

    // Thread for managing Async operations (Note: only safe to use within the context of a valid stratus::Thread,
    // so a raw pthread or std::thread are not useable)
    template<typename E>
    class __AsyncImpl : public std::enable_shared_from_this<__AsyncImpl<E>> {
    public:        
        __AsyncImpl(Thread& context, std::function<E *(void)> compute) 
            : _context(&context),
              _compute(compute) {}
        __AsyncImpl(const __AsyncImpl&) = delete;
        __AsyncImpl(__AsyncImpl&&) = delete;
        __AsyncImpl& operator=(const __AsyncImpl&) = delete;
        __AsyncImpl& operator=(__AsyncImpl&&) = delete;
        ~__AsyncImpl() = default;

        // Should only be called by the wrapper class Async
        void Start() {
            std::shared_ptr<__AsyncImpl> shared = shared_from_this();
            _context->Queue([this, shared]() {
                try {
                    E * result = this->_compute();
                    auto ul = this->_LockWrite();
                    this->_result = std::shared_ptr<E>(result);
                    this->_complete = true;
                    this->_failed = false;
                }
                catch (const std::exception& e) {
                    auto ul = this->_LockWrite();
                    this->_complete = true;
                    this->_failed = true;
                    this->_exceptionMessage = e.what();
                }
            });
        }

        // Getters for checking internal state
        bool Failed()                  const { auto sl = _LockRead(); return _failed; }
        bool Completed()               const { auto sl = _LockRead(); return _complete; }
        std::string ExceptionMessage() const { auto sl = _LockRead(); return _exceptionMessage; }

        // Getters for retrieving result
        const E& Get() const {
            if (!Completed()) {
                throw std::runtime_error("stratus::Async::Get called before completion");
            }
            return *_result;
        }

        E& Get() {
            if (!Completed()) {
                throw std::runtime_error("stratus::Async::Get called before completion");
            }
            return *_result;
        }

        std::shared_ptr<E> GetPtr() const {
            if (!Completed()) {
                throw std::runtime_error("stratus::Async::Get called before completion");
            }
            return _result;
        }

    private:
        std::unique_lock<std::shared_mutex> _LockWrite() const { return std::unique_lock<std::shared_mutex>(_mutex); }
        std::shared_lock<std::shared_mutex> _LockRead()  const { return std::shared_lock<std::shared_mutex>(_mutex); }

    private:
        std::shared_ptr<E> _result = nullptr;
        Thread * _context;
        std::function<E *(void)> _compute;
        mutable std::shared_mutex _mutex;
        std::string _exceptionMessage;
        bool _failed = false;
        bool _complete = false;
    };

    // To use this class, do something like the following:
    // Async<int> compute(thread, [](){ return new int(12); });
    // thread.Dispatch();
    // while (!compute.Complete())
    //      ;
    // if (!compute.Failed()) std::cout << compute.Get() << std::endl;
    template<typename E>
    class Async {
    public:
        Async() {}
        Async(Thread& context, std::function<E *(void)> function)
            : _impl(std::make_shared<__AsyncImpl<E>>(context, function)) {
            _impl->Start();
        }
        Async(const Async&) = default;
        Async(Async&&) = default;
        Async& operator=(const Async&) = default;
        Async& operator=(Async&&) = default;
        ~Async() = default;

        // Getters for checking internal state
        bool Failed()                  const { return _impl->Failed(); }
        bool Completed()               const { return _impl->Completed(); }
        std::string ExceptionMessage() const { return _impl->ExceptionMessage(); }

        // Getters for retrieving result
        const E& Get()              const { return _impl->Get(); }
        E& Get()                          { return _impl->Get(); }
        std::shared_ptr<E> GetPtr() const { return _impl->GetPtr(); }

    private:
        std::shared_ptr<__AsyncImpl<E>> _impl;
    };
}