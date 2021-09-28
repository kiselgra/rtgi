
// main a1 themes
#define RTGI_CONFIG_A1
#define RTGI_SKIP_CAMRAY_SETUP
#define RTGI_SKIP_SEQ_IS_IMPL
// requires
RTGI_SKIP_BVH
RTGI_SKIP_PRIM_HIT_IMPL
RTGI_SKIP_LOCAL_ILLUM
RTGI_SKIP_LOCAL_ILLUM_IMPL
RTGI_SKIP_RAY_TRI_IS_IMPL
RTGI_SKIP_RAY_BOX_IS
RTGI_SKIP_RAY_BOX_IS_1
RTGI_SKIP_RAY_BOX_IS_2
RTGI_SKIP_RAY_BOX_IS_3
RTGI_SKIP_RAY_BOX_IS_4
RTGI_SKIP_BRDF

// main a2 themes
#define RTGI_CONFIG_A2
#undef  RTGI_SKIP_BVH1
#define RTGI_SKIP_BVH1_OM_IMPL
#define RTGI_SKIP_BVH1_TRAV_IMPL
#undef  RTGI_SKIP_BVH2
#define RTGI_SKIP_BVH2_OM_IMPL 
#define RTGI_SKIP_BVH2_SM_IMPL 
#define RTGI_SKIP_BVH2_SAH_IMPL 
#define RTGI_SKIP_BVH2_TRAV_IMPL

// main a3 themes
#define RTGI_CONFIG_A3
#undef  RTGI_SKIP_LOCAL_ILLUM
#undef  RTGI_SKIP_BRDF
#define RTGI_SKIP_BRDF_IMPL

// main a4 themes
#define RTGI_CONFIG_A4
#undef  RTGI_SKIP_LAYERED_BRDF
#define RTGI_SKIP_LAYERED_BRDF_IMPL
#undef  RTGI_SKIP_MF_BRDF
#define RTGI_SKIP_MF_BRDF_IMPL

// main a5 themes
#undef  RTGI_SKIP_DIRECT_ILLUM
#define RTGI_SKIP_DIRECT_ILLUM_LIGHT_POWER_IMPL
#define RTGI_SKIP_LIGHT_SOURCE_SAMPLING

// main a6 themes
#undef  RTGI_SKIP_DIRECT_ILLUM_LIGHT_POWER_IMPL
#undef  RTGI_SKIP_LIGHT_SOURCE_SAMPLING
