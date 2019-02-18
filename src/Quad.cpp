
#include "includes/Quad.h"
#include <vector>

static const std::vector<GLfloat> data = std::vector<GLfloat>{
        // positions            normals                 texture coordinates
        -0.5f, -0.5f, -0.5f,	0.0f, 0.0f, -1.0f,		0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,	0.0f, 0.0f, -1.0f,		1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,	0.0f, 0.0f, -1.0f,		1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,	0.0f, 0.0f, -1.0f,		1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,	0.0f, 0.0f, -1.0f,		0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,    0.0f, 0.0f, -1.0f,      0.0f, 0.0f,
};

Quad::Quad(RenderMode mode) :
    RenderEntity(mode, RenderProperties::FLAT) {
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_buffer);

    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _buffer);

    glBufferData(GL_ARRAY_BUFFER, data.size(), &data[0], GL_STATIC_DRAW);


}

Quad::~Quad() {
    glDeleteVertexArrays(1, &_vao);
    glDeleteBuffers(1, &_buffer);
}

void Quad::render() {

}
