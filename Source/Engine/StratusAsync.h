#pragma once

#include "StratusThread.h"

namespace stratus { 
    // Thread for managing Async operations (Note: only safe to use within the context of a valid stratus::Thread,
    // so a raw pthread or std::thread are not useable).
    //
    // It's important to know that whatever thread calls AddCallback will be the same thread that is executes
    // the callback to let you know it completed.
    template<typename E>
    class AsyncImpl_ : public std::enable_shared_from_this<AsyncImpl_<E>> {
    public:        
        AsyncImpl_(const std::shared_ptr<E>& result) {
            result_ = result;
            complete_ = true;
            failed_ = result == nullptr;
        }

        AsyncImpl_(Thread& context, std::function<E *(void)> compute) 
            : AsyncImpl_(context, [compute]() { return std::shared_ptr<E>(compute()); }) {}

        AsyncImpl_(Thread& context, std::function<std::shared_ptr<E> (void)> compute)
            : context_(&context),
              compute_(compute) {}

        AsyncImpl_(const AsyncImpl_&) = delete;
        AsyncImpl_(AsyncImpl_&&) = delete;
        AsyncImpl_& operator=(const AsyncImpl_&) = delete;
        AsyncImpl_& operator=(AsyncImpl_&&) = delete;
        ~AsyncImpl_() = default;

        // Should only be called by the wrapper class Async
        void Start() {
            if (Completed()) return;

            std::shared_ptr<AsyncImpl_> shared = this->shared_from_this();
            context_->Queue([this, shared]() {
                bool failed = false;
                try {
                    std::shared_ptr<E> result = this->compute_();
                    auto ul = this->LockWrite_();
                    this->result_ = result;
                    this->complete_ = true;
                    this->failed_ = this->result_ == nullptr;
                    failed = this->failed_;
                }
                catch (const std::exception& e) {
                    auto ul = this->LockWrite_();
                    this->complete_ = true;
                    this->failed_ = true;
                    this->exceptionMessage_ = e.what();
                    failed = true;
                }

                // End early to prevent waiting callbacks from receiving null pointers
                if (failed) return;

                // Notify everyone that we're done
                this->ProcessCallbacks_();
            });
        }

        // Getters for checking internal state
        bool Failed()                  const { auto sl = LockRead_(); return failed_; }
        bool Completed()               const { auto sl = LockRead_(); return complete_; }
        bool CompleteAndValid()        const { auto sl = LockRead_(); return complete_ && !failed_; }
        bool CompleteAndInvalid()      const { auto sl = LockRead_(); return complete_ && failed_; }
        std::string ExceptionMessage() const { auto sl = LockRead_(); return exceptionMessage_; }

        // Getters for retrieving result
        const E& Get() const {
            if (!Completed()) {
                throw std::runtime_error("stratus::Async::Get called before completion");
            }

            if (Failed()) {
                throw std::runtime_error("Get() called on a failed Async operation");
            }

            return *result_;
        }

        E& Get() {
            if (!Completed()) {
                throw std::runtime_error("stratus::Async::Get called before completion");
            }

            if (Failed()) {
                throw std::runtime_error("Get() called on a failed Async operation");
            }

            return *result_;
        }

        std::shared_ptr<E> GetPtr() const {
            if (!Completed()) {
                throw std::runtime_error("stratus::Async::Get called before completion");
            }

            if (Failed()) {
                throw std::runtime_error("Get() called on a failed Async operation");
            }

            return result_;
        }

        void AddCallback(const Thread::ThreadFunction & callback) {
            Thread * thread = &Thread::Current();
            // If completed then schedule it immediately
            if (Completed() && !Failed()) {
                thread->Queue(callback);
            }
            else {
                auto ul = LockWrite_();
                if (callbacks_.find(thread) == callbacks_.end()) {
                    callbacks_.insert(std::make_pair(thread, std::vector<Thread::ThreadFunction>()));
                }
                callbacks_.find(thread)->second.push_back(callback);
            }
        }

    private:
        std::unique_lock<std::shared_mutex> LockWrite_() const { return std::unique_lock<std::shared_mutex>(mutex_); }
        std::shared_lock<std::shared_mutex> LockRead_()  const { return std::shared_lock<std::shared_mutex>(mutex_); }

        void ProcessCallbacks_() {
            std::unordered_map<Thread *, std::vector<Thread::ThreadFunction>> callbacks;
            {
                auto ul = LockWrite_();
                callbacks = std::move(callbacks_);
            }
            for (auto entry : callbacks) {
                entry.first->QueueMany(entry.second);
            }
        }

    private:
        std::shared_ptr<E> result_ = nullptr;
        Thread * context_;
        std::function<std::shared_ptr<E> (void)> compute_;
        mutable std::shared_mutex mutex_;
        std::string exceptionMessage_;
        bool failed_ = false;
        bool complete_ = false;
        std::unordered_map<Thread *, std::vector<Thread::ThreadFunction>> callbacks_;
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
            : impl_(std::make_shared<AsyncImpl_<E>>(result)) {}

        Async(Thread& context, std::function<E *(void)> function)
            : impl_(std::make_shared<AsyncImpl_<E>>(context, function)) {
            impl_->Start();
        }

        Async(Thread& context, std::function<std::shared_ptr<E> (void)> function)
            : impl_(std::make_shared<AsyncImpl_<E>>(context, function)) {
            impl_->Start();
        }

        Async(const Async&) = default;
        Async(Async&&) = default;
        Async& operator=(const Async&) = default;
        Async& operator=(Async&&) = default;
        ~Async() = default;

        // Getters for checking internal state
        bool Failed()                  const { return impl_ == nullptr || impl_->Failed(); }
        bool Completed()               const { return impl_ == nullptr || impl_->Completed(); }
        bool CompleteAndValid()        const { return impl_ != nullptr && impl_->CompleteAndValid(); }
        bool CompleteAndInvalid()      const { return impl_ == nullptr || impl_->CompleteAndInvalid(); }
        std::string ExceptionMessage() const { return impl_ == nullptr ? "" : impl_->ExceptionMessage(); }

        // Getters for retrieving result
        const E& Get()              const { return impl_->Get(); }
        E& Get()                          { return impl_->Get(); }
        std::shared_ptr<E> GetPtr() const { return impl_->GetPtr(); }

        // Callback support
        void AddCallback(const AsyncCallback & callback) {
            Async<E> copy = *this;
            impl_->AddCallback([copy, callback]() { callback(copy); });
        }

    private:
        std::shared_ptr<AsyncImpl_<E>> impl_;
    };
}