
#ifndef STRATUSGFX_CAMERA_H
#define STRATUSGFX_CAMERA_H

#include "StratusEntity.h"
#include "StratusMath.h"

namespace stratus {
    class Camera;
    typedef std::shared_ptr<Camera> CameraPtr;

/**
 * A camera is an object that can view the world
 * from a certain perspective.
 */
class Camera {
    mutable glm::mat4 viewTransform_ = glm::mat4(1.0f);
    mutable glm::mat4 worldTransform_ = glm::mat4(1.0f);
    glm::vec3 position_ = glm::vec3(0.0f);
    glm::vec3 speed_ = glm::vec3(0.0f);
    Rotation rotation_;
    bool reOrthonormalizeBasisVectors_;
    bool rangeCheckAngles_;
    mutable bool viewTransformValid_ = false;

public:
    Camera(bool reOrthonormalizeBasisVectors, bool rangeCheckAngles);
    Camera(const Camera&) = default;
    Camera(Camera&&) = default;
    ~Camera() = default;

    /**
     * Adjusts the angle pitch/yaw of the camera by
     * some delta amount.
     * @param deltaYaw change in angle yaw
     * @param deltaPitch change in angle pitch
     * @param deltaRoll change angle in roll
     */
    void ModifyAngle(Degrees deltaPitch, Degrees deltaYaw, Degrees deltaRoll);

    // Sets the x, y and z angles in degrees
    void SetAngle(const Rotation & rotation);
    const Rotation & GetRotation() const;

    void SetPosition(float x, float y, float z);
    void SetPosition(const glm::vec3 & position);
    const glm::vec3 & GetPosition() const;

    glm::vec3 GetDirection() const;
    glm::vec3 GetUp() const;
    glm::vec3 GetSide() const;

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
    void SetSpeed(float forward, float up, float strafe);
    void SetSpeed(const glm::vec3 &);
    const glm::vec3 & GetSpeed() const;

    /**
     * Helper functions that return the pitch/yaw.
     */
    float GetYaw() const;
    float GetPitch() const;

    // Applies position += speed * deltaSeconds
    void Update(double deltaSeconds);

    /**
     * @return view transform associated with this camera (world -> camera)
     */
    const glm::mat4& GetViewTransform() const;

    // Gets the camera -> world transform
    const glm::mat4& GetWorldTransform() const;

    CameraPtr Copy() const;

private:
    void InvalidateView_();
    void UpdateViewTransform_() const;
    // void _updateCameraAxes();
};
}

#endif //STRATUSGFX_CAMERA_H
