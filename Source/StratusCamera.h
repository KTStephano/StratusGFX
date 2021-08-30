
#ifndef STRATUSGFX_CAMERA_H
#define STRATUSGFX_CAMERA_H

#include "StratusEntity.h"
#include "StratusMath.h"

namespace stratus {
/**
 * A camera is an object that can view the world
 * from a certain perspective.
 */
class Camera {
    mutable glm::mat4 _viewTransform;
    mutable glm::mat4 _worldTransform;
    glm::vec3 _position = glm::vec3(0.0f);
    glm::vec3 _speed = glm::vec3(0.0f);
    Rotation _rotation;
    bool _rangeCheckAngles;
    mutable bool _viewTransformValid = false;

public:
    Camera(bool rangeCheckAngles = true);
    ~Camera() = default;

    /**
     * Adjusts the angle pitch/yaw of the camera by
     * some delta amount.
     * @param deltaYaw change in angle yaw
     * @param deltaPitch change in angle pitch
     * @param deltaRoll change angle in roll
     */
    void modifyAngle(Degrees deltaYaw, Degrees deltaPitch, Degrees deltaRoll);

    // Sets the x, y and z angles in degrees
    void setAngle(const Rotation & rotation);
    const Rotation & getRotation() const;

    void setPosition(float x, float y, float z);
    void setPosition(const glm::vec3 & position);
    const glm::vec3 & getPosition() const;

    glm::vec3 getDirection() const;
    glm::vec3 getUp() const;
    glm::vec3 getSide() const;

    /**
     * Sets the speed x/y/z of the camera.
     *
     * @param forward negative values send it back, positive send
     *      it forward
     * @param up negative values send it down, positive send it
     *      up
     * @param strafe negative values send it left, positive send
     *      it right
     */
    void setSpeed(float forward, float up, float strafe);
    void setSpeed(const glm::vec3 &);
    const glm::vec3 & getSpeed() const;

    /**
     * Helper functions that return the pitch/yaw.
     */
    float getYaw() const;
    float getPitch() const;

    // Applies position += speed * deltaSeconds
    void update(double deltaSeconds);

    /**
     * @return view transform associated with this camera (world -> camera)
     */
    const glm::mat4& getViewTransform() const;

    // Gets the camera -> world transform
    const glm::mat4& getWorldTransform() const;

private:
    void _invalidateView();
    void _updateViewTransform() const;
    // void _updateCameraAxes();
};
}

#endif //STRATUSGFX_CAMERA_H
