
#include <vector>
#include <iostream>
#include "Cube.h"

namespace stratus {
static const std::vector<GLfloat> cubeData = std::vector<GLfloat>{
        // back face
        // positions          // normals          // tex coords     // tangent   // bitangent
        -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f,       1, 0, 0,     0, 1, 0, // bottom-left
        1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f,        1, 0, 0,     0, 1, 0,// top-right
        1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f,        1, 0, 0,     0, 1, 0,// bottom-right
        1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f,        1, 0, 0,     0, 1, 0, // top-right
        -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f,       1, 0, 0,     0, 1, 0, // bottom-left
        -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f,       1, 0, 0,     0, 1, 0, // top-left
        // front face        
        -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f,       1, 0, 0,     0, 1, 0, // bottom-left
        1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f,        1, 0, 0,     0, 1, 0,// bottom-right
        1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f,        1, 0, 0,     0, 1, 0,// top-right
        1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f,        1, 0, 0,     0, 1, 0, // top-right
        -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f,       1, 0, 0,     0, 1, 0, // top-left
        -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f,       1, 0, 0,     0, 1, 0, // bottom-left
        // left face        
        -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f,       0, 1, 0,     0, 0, -1, // top-right
        -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f,       0, 1, 0,     0, 0, -1,// top-left
        -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f,       0, 1, 0,     0, 0, -1,// bottom-left
        -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f,       0, 1, 0,     0, 0, -1, // bottom-left
        -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f,       0, 1, 0,     0, 0, -1, // bottom-right
        -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f,       0, 1, 0,     0, 0, -1,// top-right
        // right face        
        1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f,        0, 1, 0,     0, 0, -1,// top-left
        1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,        0, 1, 0,     0, 0, -1, // bottom-right
        1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f,        0, 1, 0,     0, 0, -1,// top-right
        1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,        0, 1, 0,     0, 0, -1, // bottom-right
        1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f,        0, 1, 0,     0, 0, -1, // top-left
        1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f,        0, 1, 0,     0, 0, -1,// bottom-left
        // bottom face        
        -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f,       1, 0, 0,     0, 0, -1,// top-right
        1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f,        1, 0, 0,     0, 0, -1,// top-left
        1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,        1, 0, 0,     0, 0, -1,// bottom-left
        1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,        1, 0, 0,     0, 0, -1,// bottom-left
        -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f,       1, 0, 0,     0, 0, -1,// bottom-right
        -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f,       1, 0, 0,     0, 0, -1,// top-right
        // top face        
        -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f,       1, 0, 0,     0, 0, -1,// top-left
        1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f,        1, 0, 0,     0, 0, -1,// bottom-right
        1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f,        1, 0, 0,     0, 0, -1,// top-right
        1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f,        1, 0, 0,     0, 0, -1,// bottom-right
        -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f,       1, 0, 0,     0, 0, -1,// top-left
        -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f,       1, 0, 0,     0, 0, -1// bottom-left
};

static void createCubeVAO(GLuint & vao, GLuint & buffer) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &buffer);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    //std::cout << cubeData.size() << std::endl;
    glBufferData(GL_ARRAY_BUFFER, cubeData.size() * sizeof(float), &cubeData[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
            0,                   // attrib index
            3,                   // elems per attrib
            GL_FLOAT,            // data type
            GL_FALSE,            // normalized?
            14 * sizeof(float),   // offset until next vertex
            nullptr);            // initial offset

    // tex coords
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,
                          2,
                          GL_FLOAT, 
                          GL_FALSE,
                          sizeof(float) * 14,
                          (void *)(sizeof(float) * 6));
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

struct CubeData {
    GLuint vao = 0;
    GLuint buffer = 0;
};
static CubeData __data;

Cube::Cube() {
    if (__data.vao == 0 || __data.buffer == 0) {
        createCubeVAO(__data.vao, __data.buffer);
    }
    _vao = __data.vao;
    _buffer = __data.buffer;
    _data.data = (void *)&__data;
}

Cube::~Cube() {
    //glDeleteVertexArrays(1, &_vao);
    //glDeleteBuffers(1, &_buffer);
}

void Cube::render() {
    glFrontFace(GL_CCW);
    bindVertexAttribArray();
    glDrawArrays(GL_TRIANGLES, 0, 36);
    unbindVertexAttribArray();
    /*
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_buffer);

    glBindVertexArray(_vao);

    glBindBuffer(GL_ARRAY_BUFFER, _buffer);
    //std::cout << cubeData.size() << std::endl;
    glBufferData(GL_ARRAY_BUFFER, cubeData.size() * sizeof(float), &cubeData[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
            0,                   // attrib index
            3,                   // elems per attrib
            GL_FLOAT,            // data type
            GL_FALSE,            // normalized?
            8 * sizeof(float),   // offset until next vertex
            nullptr);            // initial offset

    // tex coords
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,
                          2,
                          GL_FLOAT, GL_FALSE,
                          sizeof(float) * 8,
                          (void *)(sizeof(float) * 6));
    // normals
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(float) * 8,
                          (void *)(sizeof(float) * 3));

    glDrawArrays(GL_TRIANGLES, 0, 36);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
     */
}

void Cube::renderInstanced(const int numInstances) {
    glFrontFace(GL_CCW);
    bindVertexAttribArray();
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, numInstances);
    unbindVertexAttribArray();
}

void Cube::bindVertexAttribArray() {
    glBindVertexArray(_vao);
}

void Cube::unbindVertexAttribArray() {
    glBindVertexArray(0);
}
}