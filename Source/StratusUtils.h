#pragma once

#include <utility>
#include <cmath>
#include "glm/glm.hpp"
#include <iomanip>
#include <iostream>
#include <ostream>

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
}