
#ifndef STRATUSGFX_CUBE_H
#define STRATUSGFX_CUBE_H

#include "RenderEntity.h"

class Cube : public RenderEntity {
    GLuint _vao;
    GLuint _buffer;

public:
    Cube();
    ~Cube() override;
    void render() override;
};

#endif //STRATUSGFX_CUBE_H
