#pragma once

#include "StratusThread.h"

namespace stratus { 
    // Thread for managing Async operations (Note: only safe to use within the context of a valid stratus::Thread,
    // so a raw pthread or std::thread are not useable).
    //
    // It's important to know that whatever thread calls AddCallback will be the same thread that is executes
    // the callback to let you know it completed.
    template<typename E>
    class __AsyncImpl : public std::enable_shared_from_this<__AsyncImpl<E>> {
    public:        
        __AsyncImpl(const std::shared_ptr<E>& result) {
            _result = result;
            _complete = true;
            _failed = result == nullptr;
        }

        __AsyncImpl(Thread& context, std::function<E *(void)> compute) 
            : __AsyncImpl(context, [compute]() { return std::shared_ptr<E>(compute()); }) {}

        __AsyncImpl(Thread& context, std::function<std::shared_ptr<E> (void)> compute)
            : _context(&context),
              _compute(compute) {}

        __AsyncImpl(const __AsyncImpl&) = delete;
        __AsyncImpl(__AsyncImpl&&) = delete;
        __AsyncImpl& operator=(const __AsyncImpl&) = delete;
        __AsyncImpl& operator=(__AsyncImpl&&) = delete;
        ~__AsyncImpl() = default;

        // Should only be called by the wrapper class Async
        void Start() {
            if (Completed()) return;

            std::shared_ptr<__AsyncImpl> shared = this->shared_from_this();
            _context->Queue([this, shared]() {
                try {
                    std::shared_ptr<E> result = this->_compute();
                    auto ul = this->_LockWrite();
                    this->_result = result;
                    this->_complete = true;
                    this->_failed = false;
                }
                catch (const std::exception& e) {
                    auto ul = this->_LockWrite();
                    this->_complete = true;
                    this->_failed = true;
                    this->_exceptionMessage = e.what();
                }

                // Notify everyone that we're done (even if failed)
                this->_ProcessCallbacks();
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

        void AddCallback(const Thread::ThreadFunction & callback) {
            Thread * thread = &Thread::Current();
            // If completed then schedule it immediately
            if (Completed()) {
                thread->Queue(callback);
            }
            else {
                auto ul = _LockWrite();
                if (_callbacks.find(thread) == _callbacks.end()) {
                    _callbacks.insert(std::make_pair(thread, std::vector<Thread::ThreadFunction>()));
                }
                _callbacks.find(thread)->second.push_back(callback);
            }
        }

    private:
        std::unique_lock<std::shared_mutex> _LockWrite() const { return std::unique_lock<std::shared_mutex>(_mutex); }
        std::shared_lock<std::shared_mutex> _LockRead()  const { return std::shared_lock<std::shared_mutex>(_mutex); }

        void _ProcessCallbacks() {
            std::unordered_map<Thread *, std::vector<Thread::ThreadFunction>> callbacks;
            {
                auto ul = _LockWrite();
                callbacks = std::move(_callbacks);
            }
            for (auto entry : callbacks) {
                entry.first->QueueMany(entry.second);
            }
        }

    private:
        std::shared_ptr<E> _result = nullptr;
        Thread * _context;
        std::function<std::shared_ptr<E> (void)> _compute;
        mutable std::shared_mutex _mutex;
        std::string _exceptionMessage;
        bool _failed = false;
        bool _complete = false;
        std::unordered_map<Thread *, std::vector<Thread::ThreadFunction>> _callbacks;
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
        typedef std::function<void(Async<E>)> AsyncCallback;

        Async() {}
        Async(const std::shared_ptr<E>& result)
            : _impl(std::make_shared<__AsyncImpl<E>>(result)) {}

        Async(Thread& context, std::function<E *(void)> function)
            : _impl(std::make_shared<__AsyncImpl<E>>(context, function)) {
            _impl->Start();
        }

        Async(Thread& context, std::function<std::shared_ptr<E> (void)> function)
            : _impl(std::make_shared<__AsyncImpl<E>>(context, function)) {
            _impl->Start();
        }

        Async(const Async&) = default;
        Async(Async&&) = default;
        Async& operator=(const Async&) = default;
        Async& operator=(Async&&) = default;
        ~Async() = default;

        // Getters for checking internal state
        bool Failed()                  const { return _impl == nullptr || _impl->Failed(); }
        bool Completed()               const { return _impl == nullptr || _impl->Completed(); }
        std::string ExceptionMessage() const { return _impl == nullptr ? "" : _impl->ExceptionMessage(); }

        // Getters for retrieving result
        const E& Get()              const { return _impl->Get(); }
        E& Get()                          { return _impl->Get(); }
        std::shared_ptr<E> GetPtr() const { return _impl->GetPtr(); }

        // Callback support
        void AddCallback(const AsyncCallback & callback) {
            Async<E> copy = *this;
            _impl->AddCallback([copy, callback]() { callback(copy); });
        }

    private:
        std::shared_ptr<__AsyncImpl<E>> _impl;
    };
}