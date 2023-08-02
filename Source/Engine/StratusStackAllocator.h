#pragma once

#include <memory>
#include <typeinfo>
#include <exception>
#include "StratusPointer.h"

// A stack allocator is meant to provide O(1) allocation by only ever moving
// down the stack. This is best used for short duration allocations such as a
// game engine frame where things are allocated during the frame and then bulk freed
// at the end.

namespace stratus {
	struct StackAllocator {
		StackAllocator(const size_t maxBytes) {
			start_ = (uint8_t *)std::malloc(maxBytes);
			end_ = start_ + maxBytes;
			current_ = start_;
		}

		~StackAllocator() {
			if (start_ != nullptr) {
				std::free((void *)start_);
				start_ = nullptr;
				end_ = nullptr;
				current_ = nullptr;
			}
		}

		// Allocates a block of memory
		void * Allocate(const size_t bytes) {
			uint8_t * memory = current_;
			current_ = current_ + bytes;
			if (current_ > end_) {
				throw std::bad_alloc();
			}

			return reinterpret_cast<void *>(memory);
		}

		// Deallocates ALL memory
		void Deallocate() {
			current_ = start_;
		}

		// Capacity in bytes
		size_t Capacity() const noexcept {
			return end_ - start_;
		}

		// Remaining bytes
		size_t Remaining() const noexcept {
			if (current_ >= end_) return 0;
			return end_ - current_;
		}

	private:
		uint8_t * start_ = nullptr;
		uint8_t * end_ = nullptr;
		uint8_t * current_ = nullptr;
	};

	inline static UnsafePtr<StackAllocator> GetDefaultStackAllocator_() {
		thread_local static UnsafePtr<StackAllocator> allocator = MakeUnsafe<StackAllocator>(1024);
		return allocator;
	}

	// This is designed to work with C++ standard library containers so
	// it follows Allocator requirements
	//
	// The way this is meant to be used is as follows:
	//		1) Begin frame
	//		2) Allocate many small objects onto the stack-based pool allocator
	//		3) Bulk free everything
	//		4) End frame
	template<typename T>
	struct StackBasedPoolAllocator {
		typedef T              value_type;
		typedef T*             pointer;
		typedef const T*       const_pointer;
		typedef T&             reference;
		typedef const T&       const_reference;
		typedef std::size_t    size_type;
		typedef std::ptrdiff_t difference_type;

		StackBasedPoolAllocator()
			: StackBasedPoolAllocator(GetDefaultStackAllocator_()) {}

		StackBasedPoolAllocator(const size_t maxObjects)
			: StackBasedPoolAllocator(MakeUnsafe<StackAllocator>(sizeof(value_type) * maxObjects)) {}

		StackBasedPoolAllocator(const UnsafePtr<StackAllocator>& allocator)
			: allocator_(allocator) {}

		template<typename U>
		StackBasedPoolAllocator(const StackBasedPoolAllocator<U>& other)
			: allocator_(other.Allocator()) {}

		// Capacity in terms of # objects
		size_t Capacity() const noexcept {
			return allocator_->Capacity() / sizeof(value_type);
		}

		// Remaining in terms of # objects
		size_t Remaining() const noexcept {
			return allocator_->Remaining() / sizeof(value_type);
		}

		UnsafePtr<StackAllocator> Allocator() const noexcept {
			return allocator_;
		}

		// C++ Allocator requirements
		pointer address(reference x) const noexcept {
			return &x;
		}

		const_pointer address(const_reference x) const noexcept {
			return &x;
		}

		pointer allocate(std::size_t n) {
			return (pointer)allocator_->Allocate(sizeof(value_type) * n);
		}

		void deallocate(pointer p, std::size_t n) {
			// Do nothing
		}

		size_type max_size() const noexcept {
			return Remaining();
		}

		template<typename U, typename ... Args>
		void construct(U * p, Args&&... args) {
			uint8_t * memory = reinterpret_cast<uint8_t *>(p);
			::new (memory) U(std::forward<Args>(args)...);
		}

		template<typename U>
		void destroy(U * p) {
			p->~U();
		}

		bool operator==(const StackBasedPoolAllocator<T>& other) const noexcept {
			return allocator_ == other.allocator_;
		}

		bool operator!=(const StackBasedPoolAllocator<T>& other) const noexcept {
			return !(operator==(other));
		}

	private:
		UnsafePtr<StackAllocator> allocator_;
	};
}