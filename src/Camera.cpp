
#include <includes/Camera.h>

#include "includes/Camera.h"

Camera::Camera() {
    _side = glm::cross(_up, _dir);
    _viewTransform = glm::lookAt(position, position + _dir, _up);
}

void Camera::modifyAngle(double deltaYaw, double deltaPitch) {
    rotation.x += deltaYaw;
    rotation.y += deltaPitch;
    if (rotation.y > 89) rotation.y = 89.0f;
    else if (rotation.y < -89.0f) rotation.y = -89.0f;
}

void Camera::setSpeed(float forward, float up, float strafe) {
    speed.z = forward;
    speed.y = up;
    speed.x = strafe;
}

const glm::vec3 &Camera::getSpeed() const {
    return speed;
}

void Camera::update(double deltaSeconds) {
    // Update the camera's direction, up and side to form an
    // orthonormal basis
    _dir = glm::normalize(
            glm::vec3(cos(glm::radians(-getPitch())) * cos(glm::radians(getYaw())),
                    sin(glm::radians(-getPitch())),
                    cos(glm::radians(-getPitch())) * sin(glm::radians(getYaw()))));

    _up = glm::normalize(_worldUp - glm::dot(_worldUp, _dir) * _dir); // Gram-Schmidt
    _side = glm::cross(_up, _dir);

    // Update the position
    position += _dir * speed.z * (float)deltaSeconds;
    position += _up * speed.y * (float)deltaSeconds;
    position += _side * speed.x * (float)deltaSeconds;

    // Update the view transform
    _viewTransform = glm::lookAt(position, position + _dir, _up);
}

float Camera::getYaw() const {
    return rotation.x;
}

float Camera::getPitch() const {
    return rotation.z;
}

const glm::mat4 &Camera::getViewTransform() const {
    return _viewTransform;
}
