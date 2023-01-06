#pragma once

#include <utility>
#include <cmath>
#include "glm/glm.hpp"
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>

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
}