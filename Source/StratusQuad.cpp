
#include "StratusQuad.h"
#include "StratusEngine.h"
#include <vector>
#include <iostream>

namespace stratus {
static const std::vector<GLfloat> quadData = std::vector<GLfloat>{
    // positions            normals                 texture coordinates     // tangents  // bitangents
    -1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,		0.0f, 0.0f,             1, 0, 0,     0, 1, 0,
    1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,		1.0f, 0.0f,             1, 0, 0,     0, 1, 0,
    1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		1.0f, 1.0f,             1, 0, 0,     0, 1, 0,
    -1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,      0.0f, 0.0f,             1, 0, 0,     0, 1, 0,
    1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		1.0f, 1.0f,             1, 0, 0,     0, 1, 0,
    -1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		0.0f, 1.0f,             1, 0, 0,     0, 1, 0,
};

/*
static const std::vector<GLfloat> quadData = std::vector<GLfloat>{
    // positions            normals                 texture coordinates
    -1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,		0.0f, 0.0f,
     1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,		1.0f, 0.0f,
     1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		1.0f, 1.0f,
    -1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,      0.0f, 0.0f,
     1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		1.0f, 1.0f,
    -1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		0.0f, 1.0f,
};
*/
/*
static const std::vector<GLfloat> quadData = std::vector<GLfloat>{
        // positions            normals                 texture coordinates
        -1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,		0.0f, 0.0f,
        -1.0f, -1.0f, -1.0f,     0.0f, 0.0f, -1.0f,		1.0f, 0.0f,
        1.0f,  -1.0f, -1.0f,	    0.0f, 0.0f, -1.0f,		1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,      0.0f, 0.0f,
        1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		0.0f, 1.0f,
};
 */

static const size_t quadStride = 14;
static const size_t quadNumVertices = quadData.size() / quadStride;
static std::vector<glm::vec3> getQuadVertices() {
    std::vector<glm::vec3> vertices(quadNumVertices);
    for (size_t i = 0, f = 0; i < quadNumVertices; ++i, f += quadStride) {
        vertices[i] = glm::vec3(quadData[f], quadData[f + 1], quadData[f + 2]);
    }
    return vertices;
}

static std::vector<glm::vec2> getQuadTexCoords() {
    std::vector<glm::vec2> uvs(quadNumVertices);
    const size_t offset = 6;
    for (size_t i = 0, f = offset; i < quadNumVertices; ++i, f += quadStride) {
        uvs[i] = glm::vec2(quadData[f], quadData[f + 1]);
    }
    return uvs;
}

static std::vector<glm::vec3> getQuadNormals() {
    std::vector<glm::vec3> normals(quadNumVertices);
    const size_t offset = 3;
    for (size_t i = 0, f = offset; i < quadNumVertices; ++i, f += quadStride) {
        normals[i] = glm::vec3(quadData[f], quadData[f + 1], quadData[f + 2]);
    }
    return normals;
}

Quad::Quad() : Mesh(getQuadVertices(), getQuadTexCoords(), getQuadNormals()) {
    this->cullingMode = CULLING_NONE;
}

Quad::~Quad() {}
}