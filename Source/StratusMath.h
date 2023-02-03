#pragma once

#include <utility>
#include <cmath>
#include <math.h>
#include <iostream>
#include <ostream>
#include "glm/glm.hpp"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <glm/gtc/quaternion.hpp> 
#include <glm/gtx/quaternion.hpp>

#define STRATUS_PI (M_PI)

namespace stratus {
	class Degrees;
	class Radians {
		float _rad;

	public:
		Radians() : Radians(0.0f) {}
		// Constructor assumes input is already in radians
		explicit Radians(float rad) : _rad(rad) {}
		Radians(const Degrees&);
		Radians(const Radians&) = default;
		Radians(Radians&&) = default;

		float value() const { return _rad; }

		// We allow * and / with a float to act as scaling, but + and - should be with either Radians or Degrees
		Radians operator*(const float f) const { return Radians(value() * f); }
		Radians operator/(const float f) const { return Radians(value() / f); }
		
		Radians& operator*=(const float f) { _rad *= f; return *this; }
		Radians& operator/=(const float f) { _rad /= f; return *this; }

		// Operators which require no conversions
		Radians& operator=(const Radians&) = default;
		Radians& operator=(Radians&&) = default;

		Radians operator+(const Radians& r) const { return Radians(value() + r.value()); }
		Radians operator-(const Radians& r) const { return Radians(value() - r.value()); }
		Radians operator*(const Radians& r) const { return Radians(value() * r.value()); }
		Radians operator/(const Radians& r) const { return Radians(value() / r.value()); }

		Radians& operator+=(const Radians& r) { _rad += r.value(); return *this; }
		Radians& operator-=(const Radians& r) { _rad -= r.value(); return *this; }
		Radians& operator*=(const Radians& r) { _rad *= r.value(); return *this; }
		Radians& operator/=(const Radians& r) { _rad /= r.value(); return *this; }

		// Operators that need a conversion
		Radians& operator=(const Degrees& d) { _rad = Radians(d).value(); return *this; }

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
		float _deg;
	
	public:
		Degrees() : Degrees(0.0f) {}
		// Constructor assumes input is already in degrees
		explicit Degrees(float deg) : _deg(deg) {}
		Degrees(const Radians&);
		Degrees(const Degrees&) = default;
		Degrees(Degrees&&) = default;

		float value() const { return _deg; }

		// We allow * and / with a float to act as scaling, but + and - should be with either Radians or Degrees
		Degrees operator*(const float f) const { return Degrees(value() * f); }
		Degrees operator/(const float f) const { return Degrees(value() / f); }
		
		Degrees& operator*=(const float f) { _deg *= f; return *this; }
		Degrees& operator/=(const float f) { _deg /= f; return *this; }

		// Operators which require no conversions
		Degrees& operator=(const Degrees&) = default;
		Degrees& operator=(Degrees&&) = default;

		Degrees operator+(const Degrees& d) const { return Degrees(value() + d.value()); }
		Degrees operator-(const Degrees& d) const { return Degrees(value() - d.value()); }
		Degrees operator*(const Degrees& d) const { return Degrees(value() * d.value()); }
		Degrees operator/(const Degrees& d) const { return Degrees(value() / d.value()); }

		Degrees& operator+=(const Degrees& d) { _deg += d.value(); return *this; }
		Degrees& operator-=(const Degrees& d) { _deg -= d.value(); return *this; }
		Degrees& operator*=(const Degrees& d) { _deg *= d.value(); return *this; }
		Degrees& operator/=(const Degrees& d) { _deg /= d.value(); return *this; }

		// Operators that need a conversion
		Degrees& operator=(const Radians& r) { _deg = Degrees(r).value(); return *this; }

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

    static glm::mat4 ToMat4(const aiMatrix4x4& aim) {
        glm::mat4 gm(1.0f);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                gm[i][j] = aim[i][j];
            }
        }
        return gm;
    }

    static glm::mat4 ToMat4(const aiMatrix3x3& aim) {
        glm::mat4 gm(1.0f);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                gm[i][j] = aim[i][j];
            }
        }
        return gm;
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