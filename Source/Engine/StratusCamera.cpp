
#include <StratusCamera.h>
#include <iostream>

#include "StratusCamera.h"
#include "StratusUtils.h"

namespace stratus {
    static inline const glm::vec3& GetWorldUp() {
        static const glm::vec3 worldUp(0.0f, -1.0f, 0.0f);
        return worldUp;
    }

    // With reOrthonormalizeBasisVectors set to true, it will attempt to stabilize the
    // direction and side vectors each frame so that the camera behaves in an FPS-like way
    Camera::Camera(bool reOrthonormalizeBasisVectors, bool rangeCheckAngles) 
        : reOrthonormalizeBasisVectors_(reOrthonormalizeBasisVectors), rangeCheckAngles_(rangeCheckAngles) {}

    void Camera::ModifyAngle(Degrees deltaPitch, Degrees deltaYaw, Degrees deltaRoll) {
        SetAngle(Rotation(rotation_.x + deltaPitch, rotation_.y + deltaYaw, rotation_.z + deltaRoll));
    }

    void Camera::SetAngle(const Rotation & rotation) {
        rotation_ = rotation;
        if (rangeCheckAngles_) {
            const float minMax = 75.0f;
            if (rotation_.x.value() > minMax) rotation_.x = Degrees(minMax);
            else if (rotation_.x.value() < -minMax) rotation_.x = Degrees(-minMax);
        }
        InvalidateView_();
    }

    const Rotation & Camera::GetRotation() const {
        return rotation_;
    }

    void Camera::SetPosition(float x, float y, float z) {
        SetPosition(glm::vec3(x, y, z));
    }

    void Camera::SetPosition(const glm::vec3 & position) {
        position_ = position;
        InvalidateView_();
    }

    const glm::vec3 & Camera::GetPosition() const {
        return position_;
    }

    glm::vec3 Camera::GetDirection() const {
        return glm::normalize(-GetWorldTransform()[2]);
    }

    glm::vec3 Camera::GetUp() const {
        return glm::normalize(-GetWorldTransform()[1]);
    }

    glm::vec3 Camera::GetSide() const {
        return -glm::cross(GetDirection(), GetUp());
    }

    void Camera::SetSpeed(float forward, float up, float strafe) {
        SetSpeed(glm::vec3(forward, up, strafe));
    }

    void Camera::SetSpeed(const glm::vec3 & speed) {
        speed_ = speed;
    }

    const glm::vec3 & Camera::GetSpeed() const {
        return speed_;
    }

    //void Camera::Update(double deltaSeconds) {
    //    glm::vec3 dir = GetDirection();
    //    glm::vec3 up = GetUp();
    //    glm::vec3 side = GetSide();

    //    // Update the position
    //    position_ += dir * speed_.z * (float)deltaSeconds;
    //    position_ += up * speed_.y * (float)deltaSeconds;
    //    position_ += side * speed_.x * (float)deltaSeconds;

    //    InvalidateView_();
    //}

    void Camera::Update(double deltaSeconds) {
        glm::vec3 dir = GetDirection();
        glm::vec3 up = GetUp();
        glm::vec3 side = GetSide();

        if (reOrthonormalizeBasisVectors_) {
            up = GetWorldUp();
            dir = glm::normalize(glm::cross(-up, side));
            side = -glm::normalize(glm::cross(-up, dir));
        }

        // Update the position
        position_ += dir  * speed_.z * (float)deltaSeconds;
        position_ += up   * speed_.y * (float)deltaSeconds;
        position_ += side * speed_.x * (float)deltaSeconds;

        InvalidateView_();
    }

    const glm::mat4 & Camera::GetViewTransform() const {
        UpdateViewTransform_();
        return viewTransform_;
    }

    const glm::mat4 & Camera::GetWorldTransform() const {
        UpdateViewTransform_();
        return worldTransform_;
    }

    void Camera::InvalidateView_() {
        viewTransformValid_ = false;
    }

    //void Camera::UpdateViewTransform_() const {
    //    if (viewTransformValid_) return;
    //    worldTransform_ = constructTransformMat(rotation_, position_, glm::vec3(1.0f));
    //    viewTransform_ = glm::inverse(worldTransform_);
    //    viewTransformValid_ = true;
    //}

    void Camera::UpdateViewTransform_() const {
        if (viewTransformValid_) return;
        viewTransformValid_ = true;

        // Store new transform in a temporary variable so we can use the previous frame's data for
        // side and direction
        auto tmp = glm::mat4(1.0f);

        if (reOrthonormalizeBasisVectors_) {
            glm::vec3 dir = GetDirection();
            glm::vec3 up = GetWorldUp();
            glm::vec3 side = GetSide();

            dir = glm::normalize(glm::cross(-up, side));
            side = glm::normalize(glm::cross(-up, dir));

            tmp = RotationAboutAxis(side, rotation_.x) * RotationAboutAxis(-GetWorldUp(), rotation_.y);
            matTranslate(tmp, position_);
        }
        else {
            tmp = constructTransformMat(rotation_, position_, glm::vec3(1.0f));
        }
        
        // Now update
        worldTransform_ = tmp;
        viewTransform_ = glm::inverse(worldTransform_);
    }

    CameraPtr Camera::Copy() const {
        return CameraPtr(new Camera(*this));
    }

// Camera::Camera(bool rangeCheckAngles) : _rangeCheckAngles(rangeCheckAngles) {
//     _side = glm::cross(_up, _dir);
//     _viewTransform = glm::lookAt(position, position + _dir, _up);
// }

// void Camera::modifyAngle(double deltaYaw, double deltaPitch) {
//     // rotation.x += deltaYaw;
//     // rotation.y += deltaPitch;
//     rotation.x -= deltaYaw;
//     rotation.y -= deltaPitch;

//     setAngle(rotation);
// }

// void Camera::setAngle(const glm::vec3 & angle) {
//     rotation = angle;
//     //if (_rangeCheckAngles) {
//     //    if (rotation.y > 89) rotation.y = 89.0f;
//     //    else if (rotation.y < -89.0f) rotation.y = -89.0f;
//     //}

//     // _dir = glm::normalize(
//     //     glm::vec3(cos(glm::radians(-getPitch())) * cos(glm::radians(getYaw())),
//     //             sin(glm::radians(-getPitch())),
//     //             cos(glm::radians(-getPitch())) * sin(glm::radians(getYaw()))));

//     _updateCameraAxes();
//     _updateViewTransform();

//     //_dir = glm::vec3(_worldTransform[2].x, _worldTransform[2].y, _worldTransform[2].z) * -1.0f;
//     //_dir = glm::vec3(_viewTransform[2]);
//     _dir = -glm::vec3(_worldTransform[2]);
// }

// void Camera::setSpeed(float forward, float up, float strafe) {
//     speed.z = forward;
//     speed.y = up;
//     speed.x = strafe;
// }

// const glm::vec3 &Camera::getSpeed() const {
//     return speed;
// }

// void Camera::update(double deltaSeconds) {
//     // Update the position
//     position += _dir * speed.z * (float)deltaSeconds;
//     position += _up * speed.y * (float)deltaSeconds;
//     position += _side * speed.x * (float)deltaSeconds;

//     // Update the view transform
//     _updateViewTransform();
//     //std::cout << "[cam pos] x: " << position.x <<
//     //    ", y: " << position.y << ", z: " << position.z << std::endl;
// }

// float Camera::getYaw() const {
//     return rotation.x;
// }

// float Camera::getPitch() const {
//     return rotation.y;
// }

// const glm::mat4 &Camera::getViewTransform() const {
//     return _viewTransform;
// }

// const glm::mat4& Camera::getWorldTransform() const {
//     return _worldTransform;
// }

// void Camera::setPosition(float x, float y, float z) {
//     Entity::setPosition(x, y, z);
// }

// const glm::vec3 &Camera::getPosition() const {
//     return position;
// }

// const glm::vec3 & Camera::getDirection() const {
//     return _dir;
// }

// void Camera::_updateViewTransform() {
//     // _viewTransform = glm::lookAt(position, position + _dir, _up);
//     //_viewTransform = constructViewMatrix(glm::vec3(getPitch(), getYaw(), rotation.z), position);
//     //_worldTransform = glm::inverse(_viewTransform);
//     _worldTransform = constructTransformMat(rotation, position, glm::vec3(1.0f));
//     _viewTransform = glm::inverse(_worldTransform);
//     //_viewTransform = glm::inverse(_worldTransform);
//     // See https://learnopengl.com/Getting-started/Camera for information about rotation + translation when
//     // creating view matrix. We use transpose of the rotation component since doing inverse results in precision errors.
//     // _viewTransform = glm::mat4(glm::mat3(glm::transpose(_worldTransform)));
//     // _viewTransform[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
//     // glm::mat4 translate = glm::mat4(1.0f);
//     // translate[3] = glm::vec4(-position, 1.0f);
//     // _viewTransform = _viewTransform * translate;
// }

// void Camera::_updateCameraAxes() {
//     // Update the camera's up and side to form an
//     // orthonormal basis
//     _up = glm::normalize(_worldUp - glm::dot(_worldUp, _dir) * _dir); // Gram-Schmidt
//     _side = glm::cross(_up, _dir);
// }
}