#pragma once

#include <typeinfo>
#include <atomic>
#include <iostream>
#include <ostream>

namespace stratus {
    inline uint64_t NextHandle_() {
        static std::atomic<uint64_t> next(1);
        return next.fetch_add(1);
    }

    // Simple class for providing extremely light weight comparable handles. The purpose
    // of the template is to allow for differentiation between handles use by different
    // subsystems. For example, Handle<Texture> effectively creates a Texture Handle that
    // cannot be compared to, for example, Handle<Camera>.
    template<typename E>
    class Handle {
        // Private: only accessible by NextHandle()
        Handle(const uint64_t handle) : handle_(handle) {}

    public:
        // Default constructor creates the Null Handle
        Handle() : Handle(0) {}
        Handle(const Handle&) = default;
        Handle(Handle&&) = default;
        ~Handle() = default;

        // Copy operators
        Handle<E>& operator=(const Handle&) = default;
        Handle<E>& operator=(Handle&&) = default;

        // Static methods for creating new handles
        static Handle<E> NextHandle() {
            return Handle<E>(NextHandle_());
        }
        
        static Handle<E> Null() { return Handle<E>(); }

        size_t HashCode() const { return std::hash<uint64_t>{}(handle_); }
        // Unsigned 64-bit integer representation
        uint64_t Integer() const { return handle_; }

        // Comparison operators

        bool operator==(const Handle<E>& other) const { return handle_ == other.handle_; }
        bool operator!=(const Handle<E>& other) const { return handle_ != other.handle_; }
        bool operator< (const Handle<E>& other) const { return handle_ <  other.handle_; }
        bool operator<=(const Handle<E>& other) const { return handle_ <= other.handle_; }
        bool operator> (const Handle<E>& other) const { return handle_ >  other.handle_; }
        bool operator>=(const Handle<E>& other) const { return handle_ >= other.handle_; }
        operator bool() const { return handle_ != 0; }

        friend std::ostream& operator<<(std::ostream& os, const Handle<E>& h) {
            return os << "Handle{" << h.handle_ << "}";
        }

    private:
        // Local 64-bit unsigned handle - 0 == Null Handle
        uint64_t handle_ = 0;
    };
}

namespace std {
    template<typename E>
    struct hash<stratus::Handle<E>> {
        size_t operator()(const stratus::Handle<E> & h) const {
            return h.HashCode();
        }
    };
}