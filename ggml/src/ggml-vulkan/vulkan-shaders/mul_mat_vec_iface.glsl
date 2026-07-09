#include "types.glsl"

#define MAT_VEC_FUSION_FLAGS_BIAS0 0x1
#define MAT_VEC_FUSION_FLAGS_BIAS1 0x2
#define MAT_VEC_FUSION_FLAGS_SCALE0 0x4
#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
#if defined(A_TYPEV4)
layout (binding = 0) readonly buffer AV4 {A_TYPEV4 data_a_v4[];};
#endif
#if defined(A_TYPE_PACKED16)
layout (binding = 0) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 0) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
#endif

#ifdef Q4_0_SOA
// SoA aliases on binding 0: WQ (weights) viewed as u32 or u16.
layout (binding = 0) readonly buffer A_WQ32 {uint32_t  data_a_wq32[];};
layout (binding = 0) readonly buffer A_WQ16 {uint16_t  data_a_wq16[];};
// Scales live in a separate buffer (binding 5) so Fuse0/Fuse1 slots are untouched.
layout (binding = 5) readonly buffer A_WS   {float16_t data_a_ws[];};
#endif

layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
#ifdef B_TYPEV2
layout (binding = 1) readonly buffer BV2 {B_TYPEV2 data_b_v2[];};
#endif
#ifdef B_TYPEV4
layout (binding = 1) readonly buffer BV4 {B_TYPEV4 data_b_v4[];};
#endif

layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (binding = 3) readonly buffer Fuse0 {D_TYPE data_fuse0[];};
layout (binding = 4) readonly buffer Fuse1 {D_TYPE data_fuse1[];};

#if defined(MUL_MAT_ID) && !defined(Q4_0_SOA)
layout (binding = 5) readonly buffer IDS {int data_ids[];};
#endif

