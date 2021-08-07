
#ifndef STRATUSGFX_CUBE_H
#define STRATUSGFX_CUBE_H

#include "RenderEntity.h"

namespace stratus {
class Cube : public RenderEntity {
    GLuint _vao;
    GLuint _buffer;

public:
    Cube();
    ~Cube() override;
    void bindVertexAttribArray() override;
    void unbindVertexAttribArray() override;
    void render(const int) override;
};
}

#endif //STRATUSGFX_CUBE_H
