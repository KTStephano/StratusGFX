#pragma once

#include <utility>
#include <cmath>
#include "glm/glm.hpp"
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <exception>

// Printing helper functions
std::ostream& operator<<(std::ostream& os, const glm::vec2& v);
std::ostream& operator<<(std::ostream& os, const glm::vec3& v);
std::ostream& operator<<(std::ostream& os, const glm::vec4& v);
std::ostream& operator<<(std::ostream& os, const glm::mat2& m);
std::ostream& operator<<(std::ostream& os, const glm::mat3& m);
std::ostream& operator<<(std::ostream& os, const glm::mat4& m);

namespace stratus {
	// Fixes compiler errors which seem more like compiler bugs... maybe just a namespace scope thing?
	extern std::ostream& operator<<(std::ostream& os, const glm::vec2& v);
	extern std::ostream& operator<<(std::ostream& os, const glm::vec3& v);
	extern std::ostream& operator<<(std::ostream& os, const glm::vec4& v);
	extern std::ostream& operator<<(std::ostream& os, const glm::mat2& m);
	extern std::ostream& operator<<(std::ostream& os, const glm::mat3& m);
	extern std::ostream& operator<<(std::ostream& os, const glm::mat4& m);

	// Replace the first instance of oldstr with newstr
	// @return true if any replacements were made and falce otherwise
	bool ReplaceFirst(std::string& src, const std::string& oldstr, const std::string& newstr);

	// Replace all instances of oldstr with newstr
	// @return true if any replacements were made and false otherwise
	bool ReplaceAll(std::string& src, const std::string& oldstr, const std::string& newstr);

	bool BeginsWith(const std::string& src, const std::string& phrase);

	// Fixed-size array that allows constant-time insertion and deletion
	// Uses swap back to quickly remove elements, meaning that the order of
	// elements won't be maintained unless you only ever remove from the back.
	template<typename T, std::size_t Cap>
	struct FixedCapArray final {
		~FixedCapArray() {
			for (std::size_t i = 0; i < size_; i++) {
				// Zero out the memory
				elems_[i] = T();
			}
		}

		T& operator=(const std::size_t index) {
			EnsureValid_(index);
			return elems_[index];
		}

		const T& operator=(const std::size_t index) const {
			EnsureValid_(index);
			return elems_[index];
		}

		void Insert(const T& elem) {
			EnsureCapacity_();
			elems_[size_] = elem;
			++size_;
		}

		void Insert(T&& elem) {
			EnsureCapacity_();
			elems_[size_] = std::forward<T>(elem);
			++size_;
		}

		void Erase(const std::size_t index) {
			EnsureValid_(index);

			const std::size_t last = size_ - 1;
			if (index == last) {
				// Unset last element
				elems_[last] = T();
			}
			else {
				// Perform swap back
				elems_[index] = std::move(elems_[last]);
			}

			--size_;
		}

		bool Contains(const T& elem) const {
			for (std::size_t i = 0; i < size_; i++) {
				if (elems_[i] == elem) {
					return true;
				}
			}

			return false;
		}

		std::size_t Size() const {
			return size_;
		}

		constexpr std::size_t Capacity() const {
			return Cap;
		}

		T* Data() {
			return &elems_[0];
		}

		const T* Data() const {
			return &elems_[0];
		}

	private:
		inline void EnsureCapacity_() const {
			if (size_ >= Cap) {
				throw std::runtime_error("Max capacity exceeded");
			}
		}

		inline void EnsureValid_(const std::size_t index) const {
			if (index >= size_) {
				throw std::runtime_error("Index out of bounds");
			}
		}

	private:
		T elems_[Cap];
		std::size_t size_ = 0;
	};
}