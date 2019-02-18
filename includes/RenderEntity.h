//
// Created by stephano on 2/17/19.
//

#ifndef STRATUSGFX_RENDERENTITY_H
#define STRATUSGFX_RENDERENTITY_H

#include "Common.h"

enum class RenderMode {
    ORTHOGRAPHIC,   // 2d - good for menus
    PERSPECTIVE     // 3d
};

enum RenderProperties {
    INVISIBLE = 2,      // material will not be rendered
    FLAT = 4,           // material will not interact with light
    DYNAMIC = 8,        // material fully interacts with all lights
    TEXTURED = 16,       // material has one or more textures
    REFLECTIVE = 32      // material reflects world around it
};

/**
 * @see http://devernay.free.fr/cours/opengl/materials.html
 *
 * A material specifies how light will interact with a surface.
 */
struct RenderMaterial {
    glm::vec3 diffuseColor = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 specularColor = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 ambientColor = glm::vec3(1.0f, 0.0f, 0.0f);
    float specularShininess = 0.0f;
    // Not required to have a texture
    TextureHandle texture = -1;
};

class RenderEntity {
    /**
     * The render mode specifies whether we should
     * be dealing with 2d or 3d.
     */
    RenderMode _mode;

    /**
     * This is used by the renderer to decide which shader
     * program to use.
     */
    RenderProperties _properties;

    /**
     * Defines certain characteristics such as the texture
     * used, diffuse/specular/ambient colors, as well as the
     * shininess factor.
     */
    RenderMaterial _material;

public:
    /**
     * @param mode determines whether 2d/3d is necessary
     * @param properties render properties which decides which
     *      shader to use
     */
    RenderEntity(RenderMode mode, RenderProperties properties = FLAT);
    virtual ~RenderEntity();

    /**
     * This is false by default.
     *
     * If invisible is set to true, this entity will stop being
     * rendered on the screen.
     */
    void enableInvisibility(bool invisible);

    /**
     * This is false by default.
     *
     * If light interaction is enabled, this entity will react
     * to all lights in the environment (directional, spot, point).
     */
    void enableLightInteraction(bool enabled);

    /**
     * Functions for getting and setting the render material that
     * defines the way light interacts with the surface of this
     * object.
     */
    void setMaterial(const RenderMaterial & material);
    const RenderMaterial & getMaterial() const;

    RenderMode getRenderMode() const;
    RenderProperties getRenderProperties() const;

    /**
     * This gets called by the renderer when it is time
     * for the object to be drawn.
     */
    virtual void render() = 0;

private:
    void _setProperties(uint32_t properties);
    void _enableProperties(uint32_t properties);
    void _disableProperties(uint32_t properties);
};

#endif //STRATUSGFX_RENDERENTITY_H
