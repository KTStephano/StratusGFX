#pragma once

#include <cstddef>

namespace stratus {
    template<typename T>
    struct DefaultUnsafePtrDeleter_ {
        static void DestroyAndDeallocate(T * ptr) {
            delete ptr;
        }
    };

    template<typename T, typename Deleter>
    struct UnsafePtrControlBlock_ {
        Deleter deleter;
        size_t refcount = 1;
        T * ptr = nullptr;

        UnsafePtrControlBlock_(T * ptr, Deleter deleter)
            : ptr(ptr), deleter(deleter) {}
    };

    // Lightweight reference-counted thread-unsafe pointer wrapper
    template<typename T, typename Deleter = DefaultUnsafePtrDeleter_<T>>
    class UnsafePtr final {
        UnsafePtr(UnsafePtrControlBlock_<T, Deleter> * control)
            : control_(control) {}

    public:
        UnsafePtr() {}

        UnsafePtr(T * ptr, Deleter deleter = Deleter()) {
            if (ptr == nullptr) return;

            control_ = new UnsafePtrControlBlock_<T, Deleter>(ptr, deleter);
        }

        UnsafePtr(const UnsafePtr& other) {
            if (other.control_ == nullptr) return;
            control_ = other.control_;
            control_->refcount++;
        }

        UnsafePtr(UnsafePtr&& other) {
            control_ = other.control_;
            other.control_ = nullptr;
        }

        UnsafePtr& operator=(const UnsafePtr& other) {
            if (operator==(other)) return *this;

            Reset();
            control_ = other.control_;
            other.control_->refcount++;

            return *this;
        }

        UnsafePtr& operator=(UnsafePtr&& other) {
            if (operator==(other)) return *this;

            Reset();
            control_ = other.control_;
            other.control_ = nullptr;

            return *this;
        }

        ~UnsafePtr() {
            Reset();
        }

        bool operator==(const UnsafePtr& other) const {
            return control_ == other.control_;
        }

        bool operator!=(const UnsafePtr& other) const {
            return !(operator==(other));
        }

        bool operator==(std::nullptr_t) const {
            return control_ == nullptr;
        }

        bool operator!=(std::nullptr_t) const {
            return !(operator==(nullptr));
        }

        operator bool() const {
            return control_ != nullptr;
        }

        T& operator*() {
            return *(control_->ptr);
        }

        const T& operator*() const {
            return *(control_->ptr);
        }

        T * operator->() {
            return control_->ptr;
        }

        const T * operator->() const {
            return control_->ptr;
        }

        T * Get() {
            return control_ == nullptr ? nullptr : control_->ptr;
        }

        const T * Get() const {
            return control_ == nullptr ? nullptr : control_->ptr;
        }

        void Reset() {
            if (control_ == nullptr) return;

            control_->refcount--;
            if (control_->refcount == 0) {
                auto ptr = control_->ptr;
                control_->deleter.DestroyAndDeallocate(ptr);

                delete control_;
            }
            control_ = nullptr;
        }

        size_t RefCount() const {
            return control_ == nullptr ? 0 : control_->refcount;
        }

        template<typename E, typename ... Args>
        friend UnsafePtr<E> MakeUnsafe(Args&&... args);

    private:
        UnsafePtrControlBlock_<T, Deleter> * control_ = nullptr;
    };

    template<typename E, typename ... Args>
    UnsafePtr<E> MakeUnsafe(Args&&... args) {
        auto ptr = new E(std::forward<Args>(args)...);
        return UnsafePtr<E>(new UnsafePtrControlBlock_(ptr, DefaultUnsafePtrDeleter_<E>()));
    }

    static_assert(sizeof(UnsafePtr<int>) == sizeof(void *));
}