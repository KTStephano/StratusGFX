#pragma once

#include <utility>
#include <cmath>
#include "glm/glm.hpp"
#include <iomanip>
#include <iostream>
#include <ostream>

// Printing helper functions
inline std::ostream& operator<<(std::ostream& os, const glm::vec2& v) {
    return os << "[" << v.x << ", " << v.y << "]";
}

inline std::ostream& operator<<(std::ostream& os, const glm::vec3& v) {
    return os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
}

inline std::ostream& operator<<(std::ostream& os, const glm::vec4& v) {
    return os << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
}

inline std::ostream& operator<<(std::ostream& os, const glm::mat2& m) {
    static constexpr size_t size = 2;
    os << std::fixed << std::showpoint << std::setprecision(5);
    // glm::mat4 is column-major, and m[0] is the first column
    os << "[";
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            os << m[c][r];
            if (c < (size - 1)) os << ", ";
        }
        if (r < (size - 1)) os << std::endl;
    }
    return os << "]";
}

inline std::ostream& operator<<(std::ostream& os, const glm::mat3& m) {
    static constexpr size_t size = 3;
    os << std::fixed << std::showpoint << std::setprecision(5);
    // glm::mat4 is column-major, and m[0] is the first column
    os << "[";
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            os << m[c][r];
            if (c < (size - 1)) os << ", ";
        }
        if (r < (size - 1)) os << std::endl;
    }
    return os << "]";
}

inline std::ostream& operator<<(std::ostream& os, const glm::mat4& m) {
    static constexpr size_t size = 4;
    os << std::fixed << std::showpoint << std::setprecision(5);
    // glm::mat4 is column-major, and m[0] is the first column
    os << "[";
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            os << m[c][r];
            if (c < (size - 1)) os << ", ";
        }
        if (r < (size - 1)) os << std::endl;
    }
    return os << "]";
}

namespace stratus {

}