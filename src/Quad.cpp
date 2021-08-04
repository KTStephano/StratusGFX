
#include "Quad.h"
#include "Engine.h"
#include <vector>
#include <iostream>

static const std::vector<GLfloat> quadData = std::vector<GLfloat>{
    // positions            normals                 texture coordinates
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

Quad::Quad() :
    RenderEntity(RenderProperties::FLAT) {
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_buffer);

    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _buffer);

    glBufferData(GL_ARRAY_BUFFER, quadData.size() * sizeof(float), &quadData[0], GL_STATIC_DRAW);

    // positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,   // attrib index
            3,                 // elems per attrib
            GL_FLOAT,          // type
            GL_FALSE,          // normalized
            14 * sizeof(float), // stride
            nullptr);          // initial offset

    // tex coords
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,
                          2,
                          GL_FLOAT, GL_FALSE,
                          sizeof(float) * 14,
                          (void *)(sizeof(float) * 6));
    //glBindBuffer(GL_ARRAY_BUFFER, 0);

    // normals
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2,
            3,
            GL_FLOAT,
            GL_FALSE,
            sizeof(float) * 14,
            (void *)(sizeof(float) * 3));

    // tangents
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(float) * 14,
                          (void *)(sizeof(float) * 8));

    // bitangents
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(float) * 14,
                          (void *)(sizeof(float) * 11));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

Quad::~Quad() {
    glDeleteVertexArrays(1, &_vao);
    glDeleteBuffers(1, &_buffer);
}

void Quad::render() {
    glDisable(GL_CULL_FACE);
    bindVertexAttribArray();
    glDrawArrays(GL_TRIANGLES, 0, 6);
    unbindVertexAttribArray();
    glEnable(GL_CULL_FACE);
}

void Quad::renderInstanced(const int numInstances) {
    glDisable(GL_CULL_FACE);
    bindVertexAttribArray();
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, numInstances);
    unbindVertexAttribArray();
    glEnable(GL_CULL_FACE);
}

void Quad::bindVertexAttribArray() {
    glBindVertexArray(_vao);
}

void Quad::unbindVertexAttribArray() {
    glBindVertexArray(0);
}