#include "StratusUtils.h"

std::ostream& operator<<(std::ostream& os, const glm::vec2& v) {
    return os << "[" << v.x << ", " << v.y << "]";
}

std::ostream& operator<<(std::ostream& os, const glm::vec3& v) {
    return os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
}

std::ostream& operator<<(std::ostream& os, const glm::vec4& v) {
    return os << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
}

std::ostream& operator<<(std::ostream& os, const glm::mat2& m) {
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

std::ostream& operator<<(std::ostream& os, const glm::mat3& m) {
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

std::ostream& operator<<(std::ostream& os, const glm::mat4& m) {
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
    std::ostream& operator<<(std::ostream& os, const glm::vec2& v) {
        return ::operator<<(os, v);
    }

    std::ostream& operator<<(std::ostream& os, const glm::vec3& v) {
        return ::operator<<(os, v);
    }

    std::ostream& operator<<(std::ostream& os, const glm::vec4& v) {
        return ::operator<<(os, v);
    }

    std::ostream& operator<<(std::ostream& os, const glm::mat2& m) {
        return ::operator<<(os, m);
    }

    std::ostream& operator<<(std::ostream& os, const glm::mat3& m) {
        return ::operator<<(os, m);
    }

    std::ostream& operator<<(std::ostream& os, const glm::mat4& m) {
        return ::operator<<(os, m);
    }
}