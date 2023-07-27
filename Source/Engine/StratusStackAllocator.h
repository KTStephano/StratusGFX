#pragma once

#include <memory>
#include <typeinfo>
#include <exception>

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
		size_t Capacity() const {
			return end_ - start_;
		}

		// Remaining bytes
		size_t Remaining() const {
			if (current_ >= end_) return 0;
			return end_ - current_;
		}

	private:
		uint8_t * start_ = nullptr;
		uint8_t * end_ = nullptr;
		uint8_t * current_ = nullptr;
	};
}