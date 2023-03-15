#include "StratusUtils.h"
#include <sstream>

std::ostream& operator<<(std::ostream& os, const glm::vec2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}

std::ostream& operator<<(std::ostream& os, const glm::vec3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

std::ostream& operator<<(std::ostream& os, const glm::vec4& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
}

std::ostream& operator<<(std::ostream& os, const glm::mat2& m) {
    static constexpr size_t size = 2;
    os << std::fixed << std::showpoint << std::setprecision(5);
    // glm::mat4 is column-major, and m[0] is the first column
    os << "(";
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            os << m[c][r];
            if (c < (size - 1)) os << ", ";
        }
        if (r < (size - 1)) os << std::endl;
    }
    return os << ")";
}

std::ostream& operator<<(std::ostream& os, const glm::mat3& m) {
    static constexpr size_t size = 3;
    os << std::fixed << std::showpoint << std::setprecision(5);
    // glm::mat4 is column-major, and m[0] is the first column
    os << "(";
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            os << m[c][r];
            if (c < (size - 1)) os << ", ";
        }
        if (r < (size - 1)) os << std::endl;
    }
    return os << ")";
}

std::ostream& operator<<(std::ostream& os, const glm::mat4& m) {
    static constexpr size_t size = 4;
    os << std::fixed << std::showpoint << std::setprecision(5);
    // glm::mat4 is column-major, and m[0] is the first column
    os << "(";
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            os << m[c][r];
            if (c < (size - 1)) os << ", ";
        }
        if (r < (size - 1)) os << std::endl;
    }
    return os << ")";
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

    // See https://stackoverflow.com/questions/5878775/how-to-find-and-replace-string
	bool ReplaceFirst(std::string& src, const std::string& oldstr, const std::string& newstr) {
        const std::size_t pos = src.find(oldstr);
        if (pos == std::string::npos) {
            return false;
        }
        src.replace(pos, oldstr.size(), newstr);
        return true;
    }

	bool ReplaceAll(std::string& src, const std::string& oldstr, const std::string& newstr) {
        size_t pos = 0;
        bool replaced = false;
        while ((pos = src.find(oldstr, pos)) != std::string::npos) {
            replaced = true;
            src.replace(pos, oldstr.size(), newstr);
            pos += newstr.length();
        }
    
        return replaced;
    }

    bool BeginsWith(const std::string& src, const std::string& phrase) {
        if (phrase.size() > src.size()) return false;

        for (size_t i = 0; i < phrase.size(); ++i) {
            if (src[i] != phrase[i]) return false;
        }

        return true;
    }
}