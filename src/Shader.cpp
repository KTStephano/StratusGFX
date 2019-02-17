
#include <includes/Shader.h>

#include "includes/Shader.h"

Shader::Shader(const std::string &vertexShader,
        const std::string &fragShader)
        : _vtFile(vertexShader),
        _fsFile(fragShader) {
}

bool Shader::_compile() {
    return false;
}

