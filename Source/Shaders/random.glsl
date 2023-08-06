STRATUS_GLSL_VERSION

// See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].

void typeOffset(inout float x) { x += 8192; }
void typeOffset(inout vec2 x) { x += vec2(x.r, 8192); }
void typeOffset(inout vec3 x) { x += vec3(x.r, x.g, 8192); }
void typeOffset(inout vec4 x) { x += vec4(x.r, x.g, x.b, 8192); }

#define RANDOM_FUNCTION_TEMPLATE(type)  \
    float random( inout type x ) { float result = floatConstruct(hash(floatBitsToUint(x))); typeOffset(x); return result; }

RANDOM_FUNCTION_TEMPLATE(float)
RANDOM_FUNCTION_TEMPLATE(vec2)
RANDOM_FUNCTION_TEMPLATE(vec3)
RANDOM_FUNCTION_TEMPLATE(vec4)

#define RANDOM_FLOAT_TEMPLATE(type) \
    float random( inout type x, float fmin, float fmax ) { return fmin + (fmax - fmin) * random(x); }

RANDOM_FLOAT_TEMPLATE(float)
RANDOM_FLOAT_TEMPLATE(vec2)
RANDOM_FLOAT_TEMPLATE(vec3)
RANDOM_FLOAT_TEMPLATE(vec4)

#define RANDOM_VECTOR_TEMPLATE(type) \
    vec3 randomVector( inout type seed, float fmin, float fmax) { \
        return vec3(random(seed, fmin, fmax), random(seed, fmin, fmax), random(seed, fmin, fmax)); \
    }

RANDOM_VECTOR_TEMPLATE(float)
RANDOM_VECTOR_TEMPLATE(vec2)
RANDOM_VECTOR_TEMPLATE(vec3)
RANDOM_VECTOR_TEMPLATE(vec4)

// Uses a simple guess and reject method. Generates a random point, checks to see
// if the point exceeds the radius of 1, and if so rejects. Otherwise return result.
//
// Keep in mind the unit sphere in this function is centered at the origin, so z coordinate is always 0.
//
// Also keep in mind this function generates points within the sphere. It's not trying to generate surface points.
// Source is here: https://github.com/RayTracing/raytracing.github.io/blob/master/src/common/vec3.h
#define RANDOM_UNIT_SPHERE_VECTOR(type) \
    vec3 randomPointInUnitSphere( inout type x ) {          \
        const float radius = 1.0f;                          \
        while (true) {                                      \
            vec3 position = randomVector(x, -1.0, 1.0);     \
            float lengthSquared = dot(position, position);  \
            if (lengthSquared >= radius) continue;          \
            return position;                                \
        }                                                   \
    }

RANDOM_UNIT_SPHERE_VECTOR(float)
RANDOM_UNIT_SPHERE_VECTOR(vec2)
RANDOM_UNIT_SPHERE_VECTOR(vec3)
RANDOM_UNIT_SPHERE_VECTOR(vec4)

#define RANDOM_UNIT_VECTOR(type) \
    vec3 randomUnitVector( inout type x ) { return normalize(randomPointInUnitSphere(x)); }

////#define RANDOM_UNIT_VECTOR(type) \
////    vec3 randomUnitVector( inout type x ) { return normalize(randomVector(x, -1.0, 1.0)); }

RANDOM_UNIT_VECTOR(float)
RANDOM_UNIT_VECTOR(vec2)
RANDOM_UNIT_VECTOR(vec3)
RANDOM_UNIT_VECTOR(vec4)