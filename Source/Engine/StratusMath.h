#pragma once

#include <utility>
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <iostream>
#include <ostream>
#include <vector>
#include "glm/glm.hpp"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <fstream>
#include <string>
#include <limits>
#include <random>

#define STRATUS_PI 3.14159265358979323846

namespace stratus {
	class Degrees;
	class Radians {
		float rad_;

	public:
		Radians() : Radians(0.0f) {}
		// Constructor assumes input is already in radians
		explicit Radians(float rad) : rad_(rad) {}
		Radians(const Degrees&);
		Radians(const Radians&) = default;
		Radians(Radians&&) = default;

		float value() const { return rad_; }

		// We allow * and / with a float to act as scaling, but + and - should be with either Radians or Degrees
		Radians operator*(const float f) const { return Radians(value() * f); }
		Radians operator/(const float f) const { return Radians(value() / f); }
		
		Radians& operator*=(const float f) { rad_ *= f; return *this; }
		Radians& operator/=(const float f) { rad_ /= f; return *this; }

		// Operators which require no conversions
		Radians& operator=(const Radians&) = default;
		Radians& operator=(Radians&&) = default;

		Radians operator+(const Radians& r) const { return Radians(value() + r.value()); }
		Radians operator-(const Radians& r) const { return Radians(value() - r.value()); }
		Radians operator*(const Radians& r) const { return Radians(value() * r.value()); }
		Radians operator/(const Radians& r) const { return Radians(value() / r.value()); }

		Radians& operator+=(const Radians& r) { rad_ += r.value(); return *this; }
		Radians& operator-=(const Radians& r) { rad_ -= r.value(); return *this; }
		Radians& operator*=(const Radians& r) { rad_ *= r.value(); return *this; }
		Radians& operator/=(const Radians& r) { rad_ /= r.value(); return *this; }

		// Operators that need a conversion
		Radians& operator=(const Degrees& d) { rad_ = Radians(d).value(); return *this; }

		Radians operator+(const Degrees& d) const { return (*this) + Radians(d); }
		Radians operator-(const Degrees& d) const { return (*this) - Radians(d); }
		Radians operator*(const Degrees& d) const { return (*this) * Radians(d); }
		Radians operator/(const Degrees& d) const { return (*this) / Radians(d); }

		Radians& operator+=(const Degrees& d) { return (*this) += Radians(d); }
		Radians& operator-=(const Degrees& d) { return (*this) -= Radians(d); }
		Radians& operator*=(const Degrees& d) { return (*this) *= Radians(d); }
		Radians& operator/=(const Degrees& d) { return (*this) /= Radians(d); }

        // Printing helper functions --> bug in Windows compiler: can't add it here
        //friend std::ostream& operator<<(std::ostream& os, const Radians & r) {
        //    return os << r.value() << " rad";
        //}
	};

	class Degrees {
		float deg_;
	
	public:
		Degrees() : Degrees(0.0f) {}
		// Constructor assumes input is already in degrees
		explicit Degrees(float deg) : deg_(deg) {}
		Degrees(const Radians&);
		Degrees(const Degrees&) = default;
		Degrees(Degrees&&) = default;

		float value() const { return deg_; }

		// We allow * and / with a float to act as scaling, but + and - should be with either Radians or Degrees
		Degrees operator*(const float f) const { return Degrees(value() * f); }
		Degrees operator/(const float f) const { return Degrees(value() / f); }
		
		Degrees& operator*=(const float f) { deg_ *= f; return *this; }
		Degrees& operator/=(const float f) { deg_ /= f; return *this; }

		// Operators which require no conversions
		Degrees& operator=(const Degrees&) = default;
		Degrees& operator=(Degrees&&) = default;

		Degrees operator+(const Degrees& d) const { return Degrees(value() + d.value()); }
		Degrees operator-(const Degrees& d) const { return Degrees(value() - d.value()); }
		Degrees operator*(const Degrees& d) const { return Degrees(value() * d.value()); }
		Degrees operator/(const Degrees& d) const { return Degrees(value() / d.value()); }

		Degrees& operator+=(const Degrees& d) { deg_ += d.value(); return *this; }
		Degrees& operator-=(const Degrees& d) { deg_ -= d.value(); return *this; }
		Degrees& operator*=(const Degrees& d) { deg_ *= d.value(); return *this; }
		Degrees& operator/=(const Degrees& d) { deg_ /= d.value(); return *this; }

		// Operators that need a conversion
		Degrees& operator=(const Radians& r) { deg_ = Degrees(r).value(); return *this; }

		Degrees operator+(const Radians& r) const { return (*this) + Degrees(r); }
		Degrees operator-(const Radians& r) const { return (*this) - Degrees(r); }
		Degrees operator*(const Radians& r) const { return (*this) * Degrees(r); }
		Degrees operator/(const Radians& r) const { return (*this) / Degrees(r); }

		Degrees& operator+=(const Radians& r) { return (*this) += Degrees(r); }
		Degrees& operator-=(const Radians& r) { return (*this) -= Degrees(r); }
		Degrees& operator*=(const Radians& r) { return (*this) *= Degrees(r); }
		Degrees& operator/=(const Radians& r) { return (*this) /= Degrees(r); }

        // Printing helper functions
        //friend std::ostream& operator<<(std::ostream& os, const Degrees& d) {
        //    return os << d.value() << " deg";
        //}
	};

        // See http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/
    // for more info on quaternions
	struct Rotation {
		Degrees x;
		Degrees y;
		Degrees z;

        Rotation() : Rotation(Degrees(0.0f), Degrees(0.0f), Degrees(0.0f)) {}
		Rotation(const Degrees& x, const Degrees& y, const Degrees& z)
			: x(x), y(y), z(z) {}
        

        glm::vec3 asVec3() const {
            return glm::vec3(x.value(), y.value(), z.value());
        }

        glm::mat4 asMat4() const;
        glm::mat3 asMat3() const;
	};

	inline Radians cosine(const Radians& r) { return Radians(cosf(r.value())); }
	inline Radians cosine(const Degrees& d) { return Radians(cosf(Radians(d).value())); }
	
	inline Radians sine(const Radians& r) { return Radians(sinf(r.value())); }
	inline Radians sine(const Degrees& d) { return Radians(sinf(Radians(d).value())); }

	inline Radians tangent(const Radians& r) { return Radians(tanf(r.value())); }
	inline Radians tangent(const Degrees& d) { return Radians(tanf(Radians(d).value())); }

    struct TangentBitangent {
        glm::vec3 tangent;
        glm::vec3 bitangent;
    };

    /**
     * Given 3 points and 3 texture coordinates, this calculates
     * the tangent and bitangent. This is especially useful for
     * performing bump mapping where normal maps need to be transformed
     * into tangent space before calculations are performed.
     *
     * @see https://learnopengl.com/Advanced-Lighting/Normal-Mapping
     * @see https://marti.works/posts/post-calculating-tangents-for-your-mesh/post/
     */
    inline TangentBitangent calculateTangentAndBitangent(
        const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3,
        const glm::vec2& uv1, const glm::vec2& uv2, const glm::vec2& uv3) {

        // Calculate reference lines E1 and E2
        glm::vec3 edge1 = p2 - p1;
        glm::vec3 edge2 = p3 - p1;

        // Calculate the change in the uv coordinates
        // from one point to another
        glm::vec2 deltaUV1 = uv2 - uv1;
        glm::vec2 deltaUV2 = uv3 - uv1;

        // Compute the determinant
        float uvDet = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);

        glm::vec3 tangent(
            ((edge1.x * deltaUV2.y) - (edge2.x * deltaUV1.y)) * uvDet,
            ((edge1.y * deltaUV2.y) - (edge2.y * deltaUV1.y)) * uvDet,
            ((edge1.z * deltaUV2.y) - (edge2.z * deltaUV1.y)) * uvDet
        );

        glm::vec3 bitangent(
            ((edge1.x * deltaUV2.x) - (edge2.x * deltaUV1.x)) * uvDet,
            ((edge1.y * deltaUV2.x) - (edge2.y * deltaUV1.x)) * uvDet,
            ((edge1.z * deltaUV2.x) - (edge2.z * deltaUV1.x)) * uvDet
        );

        return TangentBitangent{ std::move(tangent), std::move(bitangent) };
    }

    static void matRotate(glm::mat4& out, const Rotation& rotation) {
        float cx = cosine(rotation.x).value();
        float cy = cosine(rotation.y).value();
        float cz = cosine(rotation.z).value();

        float sx = sine(rotation.x).value();
        float sy = sine(rotation.y).value();
        float sz = sine(rotation.z).value();

        out[0] = glm::vec4(cy * cz,
            sx * sy * cz + cx * sz,
            -cx * sy * cz + sx * sz,
            out[0].w);

        out[1] = glm::vec4(-cy * sz,
            -sx * sy * sz + cx * cz,
            cx * sy * sz + sx * cz,
            out[1].w);

        out[2] = glm::vec4(sy,
            -sx * cy,
            cx * cy, out[2].w);
    }

    // Inserts a 3x3 matrix into the upper section of a 4x4 matrix
    static void matInset(glm::mat4& out, const glm::mat3& in) {
        out[0].x = in[0].x;
        out[0].y = in[0].y;
        out[0].z = in[0].z;

        out[1].x = in[1].x;
        out[1].y = in[1].y;
        out[1].z = in[1].z;

        out[2].x = in[2].x;
        out[2].y = in[2].y;
        out[2].z = in[2].z;
    }

    // Equiv to Out * Mat(Scale)
    static void matScale(glm::mat4& out, const glm::vec3& scale) {
        out[0].x = out[0].x * scale.x;
        out[0].y = out[0].y * scale.y;
        out[0].z = out[0].z * scale.z;

        out[1].x = out[1].x * scale.x;
        out[1].y = out[1].y * scale.y;
        out[1].z = out[1].z * scale.z;

        out[2].x = out[2].x * scale.x;
        out[2].y = out[2].y * scale.y;
        out[2].z = out[2].z * scale.z;
    }

    // Equiv to Out * Mat(Scale)
    static void matScale(glm::mat3& out, const glm::vec3& scale) {
        out[0].x = out[0].x * scale.x;
        out[0].y = out[0].y * scale.y;
        out[0].z = out[0].z * scale.z;

        out[1].x = out[1].x * scale.x;
        out[1].y = out[1].y * scale.y;
        out[1].z = out[1].z * scale.z;

        out[2].x = out[2].x * scale.x;
        out[2].y = out[2].y * scale.y;
        out[2].z = out[2].z * scale.z;
    }

    // Equiv to T * Out
    static void matTranslate(glm::mat4& out, const glm::vec3& translate) {
        out[3].x = translate.x;
        out[3].y = translate.y;
        out[3].z = translate.z;
        out[3].w = 1.0f;
    }

    static glm::vec3 GetTranslate(const glm::mat4& mat) {
        return mat[3];
    }

    static glm::mat4 constructTransformMat(const Rotation& rotation, const glm::vec3& translation, const glm::vec3& scale) {
        glm::mat4 id(1.0f);
        matRotate(id, rotation);
        matScale(id, scale);
        matTranslate(id, translation);
        return id;
    }

    // See http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/
    static glm::mat4 RotationAboutAxis(const glm::vec3& axis, const Radians& angle) {
        const float halfAngle = angle.value() / 2.0f;
        const float x = axis.x * glm::sin(halfAngle);
        const float y = axis.y * glm::sin(halfAngle);
        const float z = axis.z * glm::sin(halfAngle);
        const float w = glm::cos(halfAngle);

        const auto rotation = glm::quat(w, x, y, z);

        return glm::toMat4(rotation);
    }

    // @See https://www.3dgep.com/understanding-the-view-matrix/
    static glm::mat4 constructViewMatrix(const Rotation& rotation, const glm::vec3& translation) {
        glm::mat4 world = constructTransformMat(rotation, translation, glm::vec3(1.0f));
        // return glm::inverse(world);
        glm::mat4 worldTranspose = glm::mat4(glm::transpose(glm::mat3(world)));
        worldTranspose[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        glm::mat4 invTranslation = glm::mat4(1.0f);
        invTranslation[3] = glm::vec4(-translation, 1.0f);
        return worldTranspose * invTranslation;
    }

    static glm::mat4 ToMat4(const aiMatrix4x4& _aim) {
        glm::mat4 gm(1.0f);
        // See https://gamedev.stackexchange.com/questions/178554/opengl-strange-mesh-when-animating-assimp
        // aiMatrix4x4 are row-major so we need to transpose it first before using it as a
        // column-major GLM matrix
        aiMatrix4x4 aim = _aim;
        aim = aim.Transpose();
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                gm[i][j] = aim[i][j];
            }
        }
        return gm;
    }

    // static glm::mat4 ToMat4(const aiMatrix3x3& aim) {
    //     glm::mat4 gm(1.0f);
    //     for (size_t i = 0; i < 3; ++i) {
    //         for (size_t j = 0; j < 3; ++j) {
    //             gm[i][j] = aim[i][j];
    //         }
    //     }
    //     return gm;
    // }

    // Generates a vector of size count containing elements ranging from [start, end] where each element
    // is logarithmically spaced.
    // See https://www.codeproject.com/Questions/188926/Generating-a-logarithmically-spaced-numbers
    template<typename T>
    static std::vector<T> LogSpace(const T start, const T end, const size_t count) {
        if (count < 2 || end <= start || start < 1) throw std::runtime_error("Bad range to LogSpace");

        const T logBase = M_E;
        const T logMin = std::log(start);
        const T logMax = std::log(end);
        const T delta = (logMax - logMin) / T(count - 1);
        
        std::vector<T> result(count);
        T accDelta = T(0);
        for (size_t i = 0; i < count; ++i) {
            result[i] = std::pow(logBase, logMin + accDelta);
            accDelta += delta;
        }

        return result;
    }

    // These are the first 512 values of the Halton sequence. For more information see:
    //     https://en.wikipedia.org/wiki/Halton_sequence
    //     https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler
    static const std::vector<std::pair<float, float>> haltonSequence = {
        {1.0f / 2.0f, 1.0f / 3.0f},
        {1.0f / 4.0f, 2.0f / 3.0f},
        {3.0f / 4.0f, 1.0f / 9.0f},
        {1.0f / 8.0f, 4.0f / 9.0f},
        {5.0f / 8.0f, 7.0f / 9.0f},
        {3.0f / 8.0f, 2.0f / 9.0f},
        {7.0f / 8.0f, 5.0f / 9.0f},
        {1.0f / 16.0f, 8.0f / 9.0f},
        {9.0f / 16.0f, 1.0f / 27.0f},
        {5.0f / 16.0f, 10.0f / 27.0f},
        {13.0f / 16.0f, 19.0f / 27.0f},
        {3.0f / 16.0f, 4.0f / 27.0f},
        {11.0f / 16.0f, 13.0f / 27.0f},
        {7.0f / 16.0f, 22.0f / 27.0f},
        {15.0f / 16.0f, 7.0f / 27.0f},
        {1.0f / 32.0f, 16.0f / 27.0f},
        {17.0f / 32.0f, 25.0f / 27.0f},
        {9.0f / 32.0f, 2.0f / 27.0f},
        {25.0f / 32.0f, 11.0f / 27.0f},
        {5.0f / 32.0f, 20.0f / 27.0f},
        {21.0f / 32.0f, 5.0f / 27.0f},
        {13.0f / 32.0f, 14.0f / 27.0f},
        {29.0f / 32.0f, 23.0f / 27.0f},
        {3.0f / 32.0f, 8.0f / 27.0f},
        {19.0f / 32.0f, 17.0f / 27.0f},
        {11.0f / 32.0f, 26.0f / 27.0f},
        {27.0f / 32.0f, 1.0f / 81.0f},
        {7.0f / 32.0f, 28.0f / 81.0f},
        {23.0f / 32.0f, 55.0f / 81.0f},
        {15.0f / 32.0f, 10.0f / 81.0f},
        {31.0f / 32.0f, 37.0f / 81.0f},
        {1.0f / 64.0f, 64.0f / 81.0f},
        {33.0f / 64.0f, 19.0f / 81.0f},
        {17.0f / 64.0f, 46.0f / 81.0f},
        {49.0f / 64.0f, 73.0f / 81.0f},
        {9.0f / 64.0f, 4.0f / 81.0f},
        {41.0f / 64.0f, 31.0f / 81.0f},
        {25.0f / 64.0f, 58.0f / 81.0f},
        {57.0f / 64.0f, 13.0f / 81.0f},
        {5.0f / 64.0f, 40.0f / 81.0f},
        {37.0f / 64.0f, 67.0f / 81.0f},
        {21.0f / 64.0f, 22.0f / 81.0f},
        {53.0f / 64.0f, 49.0f / 81.0f},
        {13.0f / 64.0f, 76.0f / 81.0f},
        {45.0f / 64.0f, 7.0f / 81.0f},
        {29.0f / 64.0f, 34.0f / 81.0f},
        {61.0f / 64.0f, 61.0f / 81.0f},
        {3.0f / 64.0f, 16.0f / 81.0f},
        {35.0f / 64.0f, 43.0f / 81.0f},
        {19.0f / 64.0f, 70.0f / 81.0f},
        {51.0f / 64.0f, 25.0f / 81.0f},
        {11.0f / 64.0f, 52.0f / 81.0f},
        {43.0f / 64.0f, 79.0f / 81.0f},
        {27.0f / 64.0f, 2.0f / 81.0f},
        {59.0f / 64.0f, 29.0f / 81.0f},
        {7.0f / 64.0f, 56.0f / 81.0f},
        {39.0f / 64.0f, 11.0f / 81.0f},
        {23.0f / 64.0f, 38.0f / 81.0f},
        {55.0f / 64.0f, 65.0f / 81.0f},
        {15.0f / 64.0f, 20.0f / 81.0f},
        {47.0f / 64.0f, 47.0f / 81.0f},
        {31.0f / 64.0f, 74.0f / 81.0f},
        {63.0f / 64.0f, 5.0f / 81.0f},
        {1.0f / 128.0f, 32.0f / 81.0f},
        {65.0f / 128.0f, 59.0f / 81.0f},
        {33.0f / 128.0f, 14.0f / 81.0f},
        {97.0f / 128.0f, 41.0f / 81.0f},
        {17.0f / 128.0f, 68.0f / 81.0f},
        {81.0f / 128.0f, 23.0f / 81.0f},
        {49.0f / 128.0f, 50.0f / 81.0f},
        {113.0f / 128.0f, 77.0f / 81.0f},
        {9.0f / 128.0f, 8.0f / 81.0f},
        {73.0f / 128.0f, 35.0f / 81.0f},
        {41.0f / 128.0f, 62.0f / 81.0f},
        {105.0f / 128.0f, 17.0f / 81.0f},
        {25.0f / 128.0f, 44.0f / 81.0f},
        {89.0f / 128.0f, 71.0f / 81.0f},
        {57.0f / 128.0f, 26.0f / 81.0f},
        {121.0f / 128.0f, 53.0f / 81.0f},
        {5.0f / 128.0f, 80.0f / 81.0f},
        {69.0f / 128.0f, 1.0f / 243.0f},
        {37.0f / 128.0f, 82.0f / 243.0f},
        {101.0f / 128.0f, 163.0f / 243.0f},
        {21.0f / 128.0f, 28.0f / 243.0f},
        {85.0f / 128.0f, 109.0f / 243.0f},
        {53.0f / 128.0f, 190.0f / 243.0f},
        {117.0f / 128.0f, 55.0f / 243.0f},
        {13.0f / 128.0f, 136.0f / 243.0f},
        {77.0f / 128.0f, 217.0f / 243.0f},
        {45.0f / 128.0f, 10.0f / 243.0f},
        {109.0f / 128.0f, 91.0f / 243.0f},
        {29.0f / 128.0f, 172.0f / 243.0f},
        {93.0f / 128.0f, 37.0f / 243.0f},
        {61.0f / 128.0f, 118.0f / 243.0f},
        {125.0f / 128.0f, 199.0f / 243.0f},
        {3.0f / 128.0f, 64.0f / 243.0f},
        {67.0f / 128.0f, 145.0f / 243.0f},
        {35.0f / 128.0f, 226.0f / 243.0f},
        {99.0f / 128.0f, 19.0f / 243.0f},
        {19.0f / 128.0f, 100.0f / 243.0f},
        {83.0f / 128.0f, 181.0f / 243.0f},
        {51.0f / 128.0f, 46.0f / 243.0f},
        {115.0f / 128.0f, 127.0f / 243.0f},
        {11.0f / 128.0f, 208.0f / 243.0f},
        {75.0f / 128.0f, 73.0f / 243.0f},
        {43.0f / 128.0f, 154.0f / 243.0f},
        {107.0f / 128.0f, 235.0f / 243.0f},
        {27.0f / 128.0f, 4.0f / 243.0f},
        {91.0f / 128.0f, 85.0f / 243.0f},
        {59.0f / 128.0f, 166.0f / 243.0f},
        {123.0f / 128.0f, 31.0f / 243.0f},
        {7.0f / 128.0f, 112.0f / 243.0f},
        {71.0f / 128.0f, 193.0f / 243.0f},
        {39.0f / 128.0f, 58.0f / 243.0f},
        {103.0f / 128.0f, 139.0f / 243.0f},
        {23.0f / 128.0f, 220.0f / 243.0f},
        {87.0f / 128.0f, 13.0f / 243.0f},
        {55.0f / 128.0f, 94.0f / 243.0f},
        {119.0f / 128.0f, 175.0f / 243.0f},
        {15.0f / 128.0f, 40.0f / 243.0f},
        {79.0f / 128.0f, 121.0f / 243.0f},
        {47.0f / 128.0f, 202.0f / 243.0f},
        {111.0f / 128.0f, 67.0f / 243.0f},
        {31.0f / 128.0f, 148.0f / 243.0f},
        {95.0f / 128.0f, 229.0f / 243.0f},
        {63.0f / 128.0f, 22.0f / 243.0f},
        {127.0f / 128.0f, 103.0f / 243.0f},
        {1.0f / 256.0f, 184.0f / 243.0f},
        {129.0f / 256.0f, 49.0f / 243.0f},
        {65.0f / 256.0f, 130.0f / 243.0f},
        {193.0f / 256.0f, 211.0f / 243.0f},
        {33.0f / 256.0f, 76.0f / 243.0f},
        {161.0f / 256.0f, 157.0f / 243.0f},
        {97.0f / 256.0f, 238.0f / 243.0f},
        {225.0f / 256.0f, 7.0f / 243.0f},
        {17.0f / 256.0f, 88.0f / 243.0f},
        {145.0f / 256.0f, 169.0f / 243.0f},
        {81.0f / 256.0f, 34.0f / 243.0f},
        {209.0f / 256.0f, 115.0f / 243.0f},
        {49.0f / 256.0f, 196.0f / 243.0f},
        {177.0f / 256.0f, 61.0f / 243.0f},
        {113.0f / 256.0f, 142.0f / 243.0f},
        {241.0f / 256.0f, 223.0f / 243.0f},
        {9.0f / 256.0f, 16.0f / 243.0f},
        {137.0f / 256.0f, 97.0f / 243.0f},
        {73.0f / 256.0f, 178.0f / 243.0f},
        {201.0f / 256.0f, 43.0f / 243.0f},
        {41.0f / 256.0f, 124.0f / 243.0f},
        {169.0f / 256.0f, 205.0f / 243.0f},
        {105.0f / 256.0f, 70.0f / 243.0f},
        {233.0f / 256.0f, 151.0f / 243.0f},
        {25.0f / 256.0f, 232.0f / 243.0f},
        {153.0f / 256.0f, 25.0f / 243.0f},
        {89.0f / 256.0f, 106.0f / 243.0f},
        {217.0f / 256.0f, 187.0f / 243.0f},
        {57.0f / 256.0f, 52.0f / 243.0f},
        {185.0f / 256.0f, 133.0f / 243.0f},
        {121.0f / 256.0f, 214.0f / 243.0f},
        {249.0f / 256.0f, 79.0f / 243.0f},
        {5.0f / 256.0f, 160.0f / 243.0f},
        {133.0f / 256.0f, 241.0f / 243.0f},
        {69.0f / 256.0f, 2.0f / 243.0f},
        {197.0f / 256.0f, 83.0f / 243.0f},
        {37.0f / 256.0f, 164.0f / 243.0f},
        {165.0f / 256.0f, 29.0f / 243.0f},
        {101.0f / 256.0f, 110.0f / 243.0f},
        {229.0f / 256.0f, 191.0f / 243.0f},
        {21.0f / 256.0f, 56.0f / 243.0f},
        {149.0f / 256.0f, 137.0f / 243.0f},
        {85.0f / 256.0f, 218.0f / 243.0f},
        {213.0f / 256.0f, 11.0f / 243.0f},
        {53.0f / 256.0f, 92.0f / 243.0f},
        {181.0f / 256.0f, 173.0f / 243.0f},
        {117.0f / 256.0f, 38.0f / 243.0f},
        {245.0f / 256.0f, 119.0f / 243.0f},
        {13.0f / 256.0f, 200.0f / 243.0f},
        {141.0f / 256.0f, 65.0f / 243.0f},
        {77.0f / 256.0f, 146.0f / 243.0f},
        {205.0f / 256.0f, 227.0f / 243.0f},
        {45.0f / 256.0f, 20.0f / 243.0f},
        {173.0f / 256.0f, 101.0f / 243.0f},
        {109.0f / 256.0f, 182.0f / 243.0f},
        {237.0f / 256.0f, 47.0f / 243.0f},
        {29.0f / 256.0f, 128.0f / 243.0f},
        {157.0f / 256.0f, 209.0f / 243.0f},
        {93.0f / 256.0f, 74.0f / 243.0f},
        {221.0f / 256.0f, 155.0f / 243.0f},
        {61.0f / 256.0f, 236.0f / 243.0f},
        {189.0f / 256.0f, 5.0f / 243.0f},
        {125.0f / 256.0f, 86.0f / 243.0f},
        {253.0f / 256.0f, 167.0f / 243.0f},
        {3.0f / 256.0f, 32.0f / 243.0f},
        {131.0f / 256.0f, 113.0f / 243.0f},
        {67.0f / 256.0f, 194.0f / 243.0f},
        {195.0f / 256.0f, 59.0f / 243.0f},
        {35.0f / 256.0f, 140.0f / 243.0f},
        {163.0f / 256.0f, 221.0f / 243.0f},
        {99.0f / 256.0f, 14.0f / 243.0f},
        {227.0f / 256.0f, 95.0f / 243.0f},
        {19.0f / 256.0f, 176.0f / 243.0f},
        {147.0f / 256.0f, 41.0f / 243.0f},
        {83.0f / 256.0f, 122.0f / 243.0f},
        {211.0f / 256.0f, 203.0f / 243.0f},
        {51.0f / 256.0f, 68.0f / 243.0f},
        {179.0f / 256.0f, 149.0f / 243.0f},
        {115.0f / 256.0f, 230.0f / 243.0f},
        {243.0f / 256.0f, 23.0f / 243.0f},
        {11.0f / 256.0f, 104.0f / 243.0f},
        {139.0f / 256.0f, 185.0f / 243.0f},
        {75.0f / 256.0f, 50.0f / 243.0f},
        {203.0f / 256.0f, 131.0f / 243.0f},
        {43.0f / 256.0f, 212.0f / 243.0f},
        {171.0f / 256.0f, 77.0f / 243.0f},
        {107.0f / 256.0f, 158.0f / 243.0f},
        {235.0f / 256.0f, 239.0f / 243.0f},
        {27.0f / 256.0f, 8.0f / 243.0f},
        {155.0f / 256.0f, 89.0f / 243.0f},
        {91.0f / 256.0f, 170.0f / 243.0f},
        {219.0f / 256.0f, 35.0f / 243.0f},
        {59.0f / 256.0f, 116.0f / 243.0f},
        {187.0f / 256.0f, 197.0f / 243.0f},
        {123.0f / 256.0f, 62.0f / 243.0f},
        {251.0f / 256.0f, 143.0f / 243.0f},
        {7.0f / 256.0f, 224.0f / 243.0f},
        {135.0f / 256.0f, 17.0f / 243.0f},
        {71.0f / 256.0f, 98.0f / 243.0f},
        {199.0f / 256.0f, 179.0f / 243.0f},
        {39.0f / 256.0f, 44.0f / 243.0f},
        {167.0f / 256.0f, 125.0f / 243.0f},
        {103.0f / 256.0f, 206.0f / 243.0f},
        {231.0f / 256.0f, 71.0f / 243.0f},
        {23.0f / 256.0f, 152.0f / 243.0f},
        {151.0f / 256.0f, 233.0f / 243.0f},
        {87.0f / 256.0f, 26.0f / 243.0f},
        {215.0f / 256.0f, 107.0f / 243.0f},
        {55.0f / 256.0f, 188.0f / 243.0f},
        {183.0f / 256.0f, 53.0f / 243.0f},
        {119.0f / 256.0f, 134.0f / 243.0f},
        {247.0f / 256.0f, 215.0f / 243.0f},
        {15.0f / 256.0f, 80.0f / 243.0f},
        {143.0f / 256.0f, 161.0f / 243.0f},
        {79.0f / 256.0f, 242.0f / 243.0f},
        {207.0f / 256.0f, 1.0f / 729.0f},
        {47.0f / 256.0f, 244.0f / 729.0f},
        {175.0f / 256.0f, 487.0f / 729.0f},
        {111.0f / 256.0f, 82.0f / 729.0f},
        {239.0f / 256.0f, 325.0f / 729.0f},
        {31.0f / 256.0f, 568.0f / 729.0f},
        {159.0f / 256.0f, 163.0f / 729.0f},
        {95.0f / 256.0f, 406.0f / 729.0f},
        {223.0f / 256.0f, 649.0f / 729.0f},
        {63.0f / 256.0f, 28.0f / 729.0f},
        {191.0f / 256.0f, 271.0f / 729.0f},
        {127.0f / 256.0f, 514.0f / 729.0f},
        {255.0f / 256.0f, 109.0f / 729.0f},
        {1.0f / 512.0f, 352.0f / 729.0f},
        {257.0f / 512.0f, 595.0f / 729.0f},
        {129.0f / 512.0f, 190.0f / 729.0f},
        {385.0f / 512.0f, 433.0f / 729.0f},
        {65.0f / 512.0f, 676.0f / 729.0f},
        {321.0f / 512.0f, 55.0f / 729.0f},
        {193.0f / 512.0f, 298.0f / 729.0f},
        {449.0f / 512.0f, 541.0f / 729.0f},
        {33.0f / 512.0f, 136.0f / 729.0f},
        {289.0f / 512.0f, 379.0f / 729.0f},
        {161.0f / 512.0f, 622.0f / 729.0f},
        {417.0f / 512.0f, 217.0f / 729.0f},
        {97.0f / 512.0f, 460.0f / 729.0f},
        {353.0f / 512.0f, 703.0f / 729.0f},
        {225.0f / 512.0f, 10.0f / 729.0f},
        {481.0f / 512.0f, 253.0f / 729.0f},
        {17.0f / 512.0f, 496.0f / 729.0f},
        {273.0f / 512.0f, 91.0f / 729.0f},
        {145.0f / 512.0f, 334.0f / 729.0f},
        {401.0f / 512.0f, 577.0f / 729.0f},
        {81.0f / 512.0f, 172.0f / 729.0f},
        {337.0f / 512.0f, 415.0f / 729.0f},
        {209.0f / 512.0f, 658.0f / 729.0f},
        {465.0f / 512.0f, 37.0f / 729.0f},
        {49.0f / 512.0f, 280.0f / 729.0f},
        {305.0f / 512.0f, 523.0f / 729.0f},
        {177.0f / 512.0f, 118.0f / 729.0f},
        {433.0f / 512.0f, 361.0f / 729.0f},
        {113.0f / 512.0f, 604.0f / 729.0f},
        {369.0f / 512.0f, 199.0f / 729.0f},
        {241.0f / 512.0f, 442.0f / 729.0f},
        {497.0f / 512.0f, 685.0f / 729.0f},
        {9.0f / 512.0f, 64.0f / 729.0f},
        {265.0f / 512.0f, 307.0f / 729.0f},
        {137.0f / 512.0f, 550.0f / 729.0f},
        {393.0f / 512.0f, 145.0f / 729.0f},
        {73.0f / 512.0f, 388.0f / 729.0f},
        {329.0f / 512.0f, 631.0f / 729.0f},
        {201.0f / 512.0f, 226.0f / 729.0f},
        {457.0f / 512.0f, 469.0f / 729.0f},
        {41.0f / 512.0f, 712.0f / 729.0f},
        {297.0f / 512.0f, 19.0f / 729.0f},
        {169.0f / 512.0f, 262.0f / 729.0f},
        {425.0f / 512.0f, 505.0f / 729.0f},
        {105.0f / 512.0f, 100.0f / 729.0f},
        {361.0f / 512.0f, 343.0f / 729.0f},
        {233.0f / 512.0f, 586.0f / 729.0f},
        {489.0f / 512.0f, 181.0f / 729.0f},
        {25.0f / 512.0f, 424.0f / 729.0f},
        {281.0f / 512.0f, 667.0f / 729.0f},
        {153.0f / 512.0f, 46.0f / 729.0f},
        {409.0f / 512.0f, 289.0f / 729.0f},
        {89.0f / 512.0f, 532.0f / 729.0f},
        {345.0f / 512.0f, 127.0f / 729.0f},
        {217.0f / 512.0f, 370.0f / 729.0f},
        {473.0f / 512.0f, 613.0f / 729.0f},
        {57.0f / 512.0f, 208.0f / 729.0f},
        {313.0f / 512.0f, 451.0f / 729.0f},
        {185.0f / 512.0f, 694.0f / 729.0f},
        {441.0f / 512.0f, 73.0f / 729.0f},
        {121.0f / 512.0f, 316.0f / 729.0f},
        {377.0f / 512.0f, 559.0f / 729.0f},
        {249.0f / 512.0f, 154.0f / 729.0f},
        {505.0f / 512.0f, 397.0f / 729.0f},
        {5.0f / 512.0f, 640.0f / 729.0f},
        {261.0f / 512.0f, 235.0f / 729.0f},
        {133.0f / 512.0f, 478.0f / 729.0f},
        {389.0f / 512.0f, 721.0f / 729.0f},
        {69.0f / 512.0f, 4.0f / 729.0f},
        {325.0f / 512.0f, 247.0f / 729.0f},
        {197.0f / 512.0f, 490.0f / 729.0f},
        {453.0f / 512.0f, 85.0f / 729.0f},
        {37.0f / 512.0f, 328.0f / 729.0f},
        {293.0f / 512.0f, 571.0f / 729.0f},
        {165.0f / 512.0f, 166.0f / 729.0f},
        {421.0f / 512.0f, 409.0f / 729.0f},
        {101.0f / 512.0f, 652.0f / 729.0f},
        {357.0f / 512.0f, 31.0f / 729.0f},
        {229.0f / 512.0f, 274.0f / 729.0f},
        {485.0f / 512.0f, 517.0f / 729.0f},
        {21.0f / 512.0f, 112.0f / 729.0f},
        {277.0f / 512.0f, 355.0f / 729.0f},
        {149.0f / 512.0f, 598.0f / 729.0f},
        {405.0f / 512.0f, 193.0f / 729.0f},
        {85.0f / 512.0f, 436.0f / 729.0f},
        {341.0f / 512.0f, 679.0f / 729.0f},
        {213.0f / 512.0f, 58.0f / 729.0f},
        {469.0f / 512.0f, 301.0f / 729.0f},
        {53.0f / 512.0f, 544.0f / 729.0f},
        {309.0f / 512.0f, 139.0f / 729.0f},
        {181.0f / 512.0f, 382.0f / 729.0f},
        {437.0f / 512.0f, 625.0f / 729.0f},
        {117.0f / 512.0f, 220.0f / 729.0f},
        {373.0f / 512.0f, 463.0f / 729.0f},
        {245.0f / 512.0f, 706.0f / 729.0f},
        {501.0f / 512.0f, 13.0f / 729.0f},
        {13.0f / 512.0f, 256.0f / 729.0f},
        {269.0f / 512.0f, 499.0f / 729.0f},
        {141.0f / 512.0f, 94.0f / 729.0f},
        {397.0f / 512.0f, 337.0f / 729.0f},
        {77.0f / 512.0f, 580.0f / 729.0f},
        {333.0f / 512.0f, 175.0f / 729.0f},
        {205.0f / 512.0f, 418.0f / 729.0f},
        {461.0f / 512.0f, 661.0f / 729.0f},
        {45.0f / 512.0f, 40.0f / 729.0f},
        {301.0f / 512.0f, 283.0f / 729.0f},
        {173.0f / 512.0f, 526.0f / 729.0f},
        {429.0f / 512.0f, 121.0f / 729.0f},
        {109.0f / 512.0f, 364.0f / 729.0f},
        {365.0f / 512.0f, 607.0f / 729.0f},
        {237.0f / 512.0f, 202.0f / 729.0f},
        {493.0f / 512.0f, 445.0f / 729.0f},
        {29.0f / 512.0f, 688.0f / 729.0f},
        {285.0f / 512.0f, 67.0f / 729.0f},
        {157.0f / 512.0f, 310.0f / 729.0f},
        {413.0f / 512.0f, 553.0f / 729.0f},
        {93.0f / 512.0f, 148.0f / 729.0f},
        {349.0f / 512.0f, 391.0f / 729.0f},
        {221.0f / 512.0f, 634.0f / 729.0f},
        {477.0f / 512.0f, 229.0f / 729.0f},
        {61.0f / 512.0f, 472.0f / 729.0f},
        {317.0f / 512.0f, 715.0f / 729.0f},
        {189.0f / 512.0f, 22.0f / 729.0f},
        {445.0f / 512.0f, 265.0f / 729.0f},
        {125.0f / 512.0f, 508.0f / 729.0f},
        {381.0f / 512.0f, 103.0f / 729.0f},
        {253.0f / 512.0f, 346.0f / 729.0f},
        {509.0f / 512.0f, 589.0f / 729.0f},
        {3.0f / 512.0f, 184.0f / 729.0f},
        {259.0f / 512.0f, 427.0f / 729.0f},
        {131.0f / 512.0f, 670.0f / 729.0f},
        {387.0f / 512.0f, 49.0f / 729.0f},
        {67.0f / 512.0f, 292.0f / 729.0f},
        {323.0f / 512.0f, 535.0f / 729.0f},
        {195.0f / 512.0f, 130.0f / 729.0f},
        {451.0f / 512.0f, 373.0f / 729.0f},
        {35.0f / 512.0f, 616.0f / 729.0f},
        {291.0f / 512.0f, 211.0f / 729.0f},
        {163.0f / 512.0f, 454.0f / 729.0f},
        {419.0f / 512.0f, 697.0f / 729.0f},
        {99.0f / 512.0f, 76.0f / 729.0f},
        {355.0f / 512.0f, 319.0f / 729.0f},
        {227.0f / 512.0f, 562.0f / 729.0f},
        {483.0f / 512.0f, 157.0f / 729.0f},
        {19.0f / 512.0f, 400.0f / 729.0f},
        {275.0f / 512.0f, 643.0f / 729.0f},
        {147.0f / 512.0f, 238.0f / 729.0f},
        {403.0f / 512.0f, 481.0f / 729.0f},
        {83.0f / 512.0f, 724.0f / 729.0f},
        {339.0f / 512.0f, 7.0f / 729.0f},
        {211.0f / 512.0f, 250.0f / 729.0f},
        {467.0f / 512.0f, 493.0f / 729.0f},
        {51.0f / 512.0f, 88.0f / 729.0f},
        {307.0f / 512.0f, 331.0f / 729.0f},
        {179.0f / 512.0f, 574.0f / 729.0f},
        {435.0f / 512.0f, 169.0f / 729.0f},
        {115.0f / 512.0f, 412.0f / 729.0f},
        {371.0f / 512.0f, 655.0f / 729.0f},
        {243.0f / 512.0f, 34.0f / 729.0f},
        {499.0f / 512.0f, 277.0f / 729.0f},
        {11.0f / 512.0f, 520.0f / 729.0f},
        {267.0f / 512.0f, 115.0f / 729.0f},
        {139.0f / 512.0f, 358.0f / 729.0f},
        {395.0f / 512.0f, 601.0f / 729.0f},
        {75.0f / 512.0f, 196.0f / 729.0f},
        {331.0f / 512.0f, 439.0f / 729.0f},
        {203.0f / 512.0f, 682.0f / 729.0f},
        {459.0f / 512.0f, 61.0f / 729.0f},
        {43.0f / 512.0f, 304.0f / 729.0f},
        {299.0f / 512.0f, 547.0f / 729.0f},
        {171.0f / 512.0f, 142.0f / 729.0f},
        {427.0f / 512.0f, 385.0f / 729.0f},
        {107.0f / 512.0f, 628.0f / 729.0f},
        {363.0f / 512.0f, 223.0f / 729.0f},
        {235.0f / 512.0f, 466.0f / 729.0f},
        {491.0f / 512.0f, 709.0f / 729.0f},
        {27.0f / 512.0f, 16.0f / 729.0f},
        {283.0f / 512.0f, 259.0f / 729.0f},
        {155.0f / 512.0f, 502.0f / 729.0f},
        {411.0f / 512.0f, 97.0f / 729.0f},
        {91.0f / 512.0f, 340.0f / 729.0f},
        {347.0f / 512.0f, 583.0f / 729.0f},
        {219.0f / 512.0f, 178.0f / 729.0f},
        {475.0f / 512.0f, 421.0f / 729.0f},
        {59.0f / 512.0f, 664.0f / 729.0f},
        {315.0f / 512.0f, 43.0f / 729.0f},
        {187.0f / 512.0f, 286.0f / 729.0f},
        {443.0f / 512.0f, 529.0f / 729.0f},
        {123.0f / 512.0f, 124.0f / 729.0f},
        {379.0f / 512.0f, 367.0f / 729.0f},
        {251.0f / 512.0f, 610.0f / 729.0f},
        {507.0f / 512.0f, 205.0f / 729.0f},
        {7.0f / 512.0f, 448.0f / 729.0f},
        {263.0f / 512.0f, 691.0f / 729.0f},
        {135.0f / 512.0f, 70.0f / 729.0f},
        {391.0f / 512.0f, 313.0f / 729.0f},
        {71.0f / 512.0f, 556.0f / 729.0f},
        {327.0f / 512.0f, 151.0f / 729.0f},
        {199.0f / 512.0f, 394.0f / 729.0f},
        {455.0f / 512.0f, 637.0f / 729.0f},
        {39.0f / 512.0f, 232.0f / 729.0f},
        {295.0f / 512.0f, 475.0f / 729.0f},
        {167.0f / 512.0f, 718.0f / 729.0f},
        {423.0f / 512.0f, 25.0f / 729.0f},
        {103.0f / 512.0f, 268.0f / 729.0f},
        {359.0f / 512.0f, 511.0f / 729.0f},
        {231.0f / 512.0f, 106.0f / 729.0f},
        {487.0f / 512.0f, 349.0f / 729.0f},
        {23.0f / 512.0f, 592.0f / 729.0f},
        {279.0f / 512.0f, 187.0f / 729.0f},
        {151.0f / 512.0f, 430.0f / 729.0f},
        {407.0f / 512.0f, 673.0f / 729.0f},
        {87.0f / 512.0f, 52.0f / 729.0f},
        {343.0f / 512.0f, 295.0f / 729.0f},
        {215.0f / 512.0f, 538.0f / 729.0f},
        {471.0f / 512.0f, 133.0f / 729.0f},
        {55.0f / 512.0f, 376.0f / 729.0f},
        {311.0f / 512.0f, 619.0f / 729.0f},
        {183.0f / 512.0f, 214.0f / 729.0f},
        {439.0f / 512.0f, 457.0f / 729.0f},
        {119.0f / 512.0f, 700.0f / 729.0f},
        {375.0f / 512.0f, 79.0f / 729.0f},
        {247.0f / 512.0f, 322.0f / 729.0f},
        {503.0f / 512.0f, 565.0f / 729.0f},
        {15.0f / 512.0f, 160.0f / 729.0f},
        {271.0f / 512.0f, 403.0f / 729.0f},
        {143.0f / 512.0f, 646.0f / 729.0f},
        {399.0f / 512.0f, 241.0f / 729.0f},
        {79.0f / 512.0f, 484.0f / 729.0f},
        {335.0f / 512.0f, 727.0f / 729.0f},
        {207.0f / 512.0f, 2.0f / 729.0f},
        {463.0f / 512.0f, 245.0f / 729.0f},
        {47.0f / 512.0f, 488.0f / 729.0f},
        {303.0f / 512.0f, 83.0f / 729.0f},
        {175.0f / 512.0f, 326.0f / 729.0f},
        {431.0f / 512.0f, 569.0f / 729.0f},
        {111.0f / 512.0f, 164.0f / 729.0f},
        {367.0f / 512.0f, 407.0f / 729.0f},
        {239.0f / 512.0f, 650.0f / 729.0f},
        {495.0f / 512.0f, 29.0f / 729.0f},
        {31.0f / 512.0f, 272.0f / 729.0f},
        {287.0f / 512.0f, 515.0f / 729.0f},
        {159.0f / 512.0f, 110.0f / 729.0f},
        {415.0f / 512.0f, 353.0f / 729.0f},
        {95.0f / 512.0f, 596.0f / 729.0f},
        {351.0f / 512.0f, 191.0f / 729.0f},
        {223.0f / 512.0f, 434.0f / 729.0f},
        {479.0f / 512.0f, 677.0f / 729.0f},
        {63.0f / 512.0f, 56.0f / 729.0f},
        {319.0f / 512.0f, 299.0f / 729.0f},
        {191.0f / 512.0f, 542.0f / 729.0f},
        {447.0f / 512.0f, 137.0f / 729.0f},
        {127.0f / 512.0f, 380.0f / 729.0f},
        {383.0f / 512.0f, 623.0f / 729.0f},
        {255.0f / 512.0f, 218.0f / 729.0f},
        {511.0f / 512.0f, 461.0f / 729.0f},
        {1.0f / 1024.0f, 704.0f / 729.0f}
    };

    // Interval is [0, 1)
    inline float RandomFloat() {
        static std::random_device device;
        static std::mt19937 generator(device());
        static std::uniform_real_distribution<> distribution(0.0, 1.0);
        return static_cast<float>(distribution(generator));
    }

    inline float RandomFloat(const float fmin, const float fmax) {
        return fmin + (fmax - fmin) * RandomFloat();
    }

    inline glm::vec3 RandomVector(const float fmin, const float fmax) {
        return glm::vec3(RandomFloat(fmin, fmax), RandomFloat(fmin, fmax), RandomFloat(fmin, fmax));
    }

    // Uses a simple guess and reject method. Generates a random point, checks to see
    // if the point exceeds the radius of 1, and if so rejects. Otherwise return result.
    //
    // Keep in mind the unit sphere in this function is centered at the origin, so z coordinate is always 0.
    //
    // Also keep in mind this function generates points within the sphere. It's not trying to generate surface points.
    // Source is here: https://github.com/RayTracing/raytracing.github.io/blob/master/src/common/vec3.h
    inline glm::vec3 RandomPointInUnitSphere() {
        static constexpr float radius = 1.0f;
        while (true) {
            const glm::vec3 position = RandomVector(-1.0f, 1.0f);
            const float lengthSquared = glm::dot(position, position);
            if (lengthSquared >= radius) continue;
            return position;
        }
    }
}

// Printing helper functions --> Putting here due to bug in Windows compiler
inline std::ostream& operator<<(std::ostream& os, const stratus::Radians & r) {
   return os << r.value() << " rad";
}

inline std::ostream& operator<<(std::ostream& os, const stratus::Degrees& d) {
    return os << d.value() << " deg";
}

inline std::ostream& operator<<(std::ostream& os, const stratus::Rotation& r) {
    return os << "[" << r.x << ", " << r.y << ", " << r.z << "]";
}