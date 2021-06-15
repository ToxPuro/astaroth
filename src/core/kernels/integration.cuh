#pragma once

typedef enum {
    STENCIL_VALUE,
    STENCIL_DERX,
    STENCIL_DERY,
    STENCIL_DERZ,
    STENCIL_DERXX,
    STENCIL_DERYY,
    STENCIL_DERZZ,
    STENCIL_DERXY,
    STENCIL_DERXZ,
    STENCIL_DERYZ,
    NUM_STENCILS,
} Stencils;

#define STENCIL_WIDTH (STENCIL_ORDER + 1)
#define STENCIL_HEIGHT (STENCIL_WIDTH)
#define STENCIL_DEPTH (STENCIL_WIDTH)

#define STENCIL_MIDX (STENCIL_WIDTH / 2 + 1)
#define STENCIL_MIDY (STENCIL_HEIGHT / 2 + 1)
#define STENCIL_MIDZ (STENCIL_DEPTH / 2 + 1)

#define INV_DS (1.0l / 0.04908738521l)

#define DER1_3 (AcReal(INV_DS * 1.0l / 60.0l))
#define DER1_2 (AcReal(INV_DS * -3.0l / 20.0l))
#define DER1_1 (AcReal(INV_DS * 3.0l / 4.0l))
#define DER1_0 (0)

#define DER2_3 (AcReal(INV_DS * INV_DS * 1.0l / 90.0l))
#define DER2_2 (AcReal(INV_DS * INV_DS * -3.0l / 20.0l))
#define DER2_1 (AcReal(INV_DS * INV_DS * 3.0l / 2.0l))
#define DER2_0 (AcReal(INV_DS * INV_DS * -49.0l / 18.0l))

#define DERX_3 (AcReal(INV_DS * INV_DS * 2.0l / 720.0l))
#define DERX_2 (AcReal(INV_DS * INV_DS * -27.0l / 720.0l))
#define DERX_1 (AcReal(INV_DS * INV_DS * 270.0l / 720.0l))
#define DERX_0 (0)

// Or simpler, just set nonzeros
// WARNING TODO FIX NOTE: may be invalid when accessed from host code!
static __device__ const AcReal
    stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH] = {
        {
            // Value
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 1, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
        }, // derx
        {
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},                                       //
                {0, 0, 0, 0, 0, 0, 0},                                       //
                {0, 0, 0, 0, 0, 0, 0},                                       //
                {-DER1_3, -DER1_2, -DER1_1, DER1_0, DER1_1, DER1_2, DER1_3}, //
                {0, 0, 0, 0, 0, 0, 0},                                       //
                {0, 0, 0, 0, 0, 0, 0},                                       //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
        },
        {
            // dery
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, -DER1_3, 0, 0, 0}, //
                {0, 0, 0, -DER1_2, 0, 0, 0}, //
                {0, 0, 0, -DER1_1, 0, 0, 0}, //
                {0, 0, 0, DER1_0, 0, 0, 0},  //
                {0, 0, 0, DER1_1, 0, 0, 0},  //
                {0, 0, 0, DER1_2, 0, 0, 0},  //
                {0, 0, 0, DER1_3, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
        },
        {
            // derz
            {
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, -DER1_3, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, -DER1_2, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, -DER1_1, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER1_0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER1_1, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER1_2, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER1_3, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
        }, // derxx
        {
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},                                    //
                {0, 0, 0, 0, 0, 0, 0},                                    //
                {0, 0, 0, 0, 0, 0, 0},                                    //
                {DER2_3, DER2_2, DER2_1, DER2_0, DER2_1, DER2_2, DER2_3}, //
                {0, 0, 0, 0, 0, 0, 0},                                    //
                {0, 0, 0, 0, 0, 0, 0},                                    //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
        },
        {
            // deryy
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, DER2_3, 0, 0, 0}, //
                {0, 0, 0, DER2_2, 0, 0, 0}, //
                {0, 0, 0, DER2_1, 0, 0, 0}, //
                {0, 0, 0, DER2_0, 0, 0, 0}, //
                {0, 0, 0, DER2_1, 0, 0, 0}, //
                {0, 0, 0, DER2_2, 0, 0, 0}, //
                {0, 0, 0, DER2_3, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
        },
        {
            // derzz
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER2_3, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER2_2, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER2_1, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER2_0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER2_1, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER2_2, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DER2_3, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
        },
        // derxy
        {
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {DERX_3, 0, 0, 0, 0, 0, -DERX_3}, //
                {0, DERX_2, 0, 0, 0, -DERX_2, 0}, //
                {0, 0, DERX_1, 0, -DERX_1, 0, 0}, //
                {0, 0, 0, DERX_0, 0, 0, 0},       //
                {0, 0, -DERX_1, 0, DERX_1, 0, 0}, //
                {0, -DERX_2, 0, 0, 0, DERX_2, 0}, //
                {-DERX_3, 0, 0, 0, 0, 0, DERX_3},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
        },
        {
            // derxz
            {
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {DERX_3, 0, 0, 0, 0, 0, -DERX_3}, //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, DERX_2, 0, 0, 0, -DERX_2, 0}, //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, DERX_1, 0, -DERX_1, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DERX_0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, -DERX_1, 0, DERX_1, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, -DERX_2, 0, 0, 0, DERX_2, 0}, //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {-DERX_3, 0, 0, 0, 0, 0, DERX_3}, //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},            //
                {0, 0, 0, 0, 0, 0, 0},
            },
        },
        {
            // deryz
            {
                {0, 0, 0, DERX_3, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, -DERX_3, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, DERX_2, 0, 0, 0},  //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, -DERX_2, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, DERX_1, 0, 0, 0},  //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, -DERX_1, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, DERX_0, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},      //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, -DERX_1, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, DERX_1, 0, 0, 0},  //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, -DERX_2, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, DERX_2, 0, 0, 0},  //
                {0, 0, 0, 0, 0, 0, 0},
            },
            {
                {0, 0, 0, -DERX_3, 0, 0, 0}, //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, 0, 0, 0, 0},       //
                {0, 0, 0, DERX_3, 0, 0, 0},
            },
        },
};

static __global__ void
dummy_kernel(void)
{
    /*
    // TODO RE-ENABLE WIP
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    */
    acComplex a = exp(AcReal(1) * acComplex(1, 1) * AcReal(1));
    a* a;
}

AcResult
acKernelDummy(void)
{
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    return AC_SUCCESS;
}

#define GEN_KERNEL (0)

static void
gen_kernel(void)
{
    // kernel unroller
    FILE* fp = fopen("kernel.out", "w");
    ERRCHK_ALWAYS(fp);

    // clang-format off
    const int NUM_FIELDS = NUM_VTXBUF_HANDLES;
    for (int field = 0; field < NUM_FIELDS; ++field) {
        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {

            fprintf(fp,
            "{const int i = blockIdx.x * blockDim.x + start.x - STENCIL_ORDER/2;\n"
            "const int j = blockIdx.y * blockDim.y + start.y - STENCIL_ORDER/2;\n"
            "const int k = blockIdx.z * blockDim.z + start.z - STENCIL_ORDER/2 + %d;\n"
            "const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;\n"
            "const int smem_width = blockDim.x + STENCIL_ORDER;\n"
            "const int smem_height = blockDim.y + STENCIL_ORDER;\n"
            "if (tid < smem_width) {\n"
                "\tfor (int height = 0; height < smem_height; ++height) {\n"
                "\t\tsmem[tid + height * smem_width] = vba.in[%d][IDX(i + tid, j + height, k)];\n"
                "\t}\n"
            "}\n"
            "__syncthreads();\n", depth, field);


            // WRITE BLOCK START
            fprintf(fp, "if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {\n");
            

            for (int height = 0; height < STENCIL_HEIGHT; ++height) {
                for (int width = 0; width < STENCIL_WIDTH; ++width) {
                    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                        if (stencils[stencil][depth][height][width] != 0) {
                            /*
                            const int i        = vertexIdx.x - STENCIL_ORDER / 2 + width;
                            const int j        = vertexIdx.y - STENCIL_ORDER / 2 + height;
                            const int k        = vertexIdx.z - STENCIL_ORDER / 2 + depth;
                            const int idx      = IDX(i, j, k);
                            const double value = vba.in[field][idx];

                            for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                                processed_stencils[field][stencil] += stencils[stencil][depth]
                                                                              [height][width] *
                                                                      value;
                                                                      */
                            
                            // NON-SMEM
                            
                            fprintf(fp,
                                    "//processed_stencils[%d][%d] += stencils[%d][%d][%d][%d] * vba.in[%d][IDX(vertexIdx.x + (%d), vertexIdx.y + (%d), vertexIdx.z + (%d))];\n",
                                    field, stencil, stencil, depth, height, width, field,
                                    -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height,
                                    -STENCIL_ORDER / 2 + depth);
                                    
                                    
                            fprintf(fp,
                                    "processed_stencils[%d][%d] += stencils[%d][%d][%d][%d] * smem[(threadIdx.x + %d) + (threadIdx.y + %d) * smem_width];\n",
                                    field, stencil, stencil, depth, height, width, width, height);
                        }
                    }
                }
            }

            // WRITE BLOCK END
            fprintf(fp, "}\n"); 

            fprintf(fp, "__syncthreads();}\n");
        }

        // WRITE BLOCK START
        fprintf(fp, "if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {\n");

        /*
        const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);
        // vba.out[field][idx] = processed_stencils[field][STENCIL_VALUE];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERX];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERXX];
        vba.out[field][idx] = processed_stencils[field][STENCIL_DERYZ];
        */
        //fprintf(fp,"vba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = processed_stencils[%d][STENCIL_DERYZ];\n", field, field);
        fprintf(fp,"vba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = processed_stencils[%d][STENCIL_DERYZ];\n", field, field);

        // WRITE BLOCK END
        fprintf(fp, "}\n");

        // clang-format on
    }

    fclose(fp);
}

template <int step_number>
static __global__ void
solve(const int3 start, const int3 end, VertexBufferArray vba)
{
    extern __shared__ AcReal smem[];

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x + start.x,
        threadIdx.y + blockIdx.y * blockDim.y + start.y,
        threadIdx.z + blockIdx.z * blockDim.z + start.z,
    };
    const int3 globalVertexIdx = (int3){
        d_multigpu_offset.x + vertexIdx.x,
        d_multigpu_offset.y + vertexIdx.y,
        d_multigpu_offset.z + vertexIdx.z,
    };

    /*
    if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z)
        return;

    assert(vertexIdx.x < DCONST(AC_nx_max) && vertexIdx.y < DCONST(AC_ny_max) &&
           vertexIdx.z < DCONST(AC_nz_max));

    assert(vertexIdx.x >= DCONST(AC_nx_min) && vertexIdx.y >= DCONST(AC_ny_min) &&
           vertexIdx.z >= DCONST(AC_nz_min));
    */

    assert(blockDim.x * blockDim.y * blockDim.z >= blockDim.x + STENCIL_ORDER); // needed for smem

    const int NUM_FIELDS = NUM_VTXBUF_HANDLES;

    // TODO test: what's the best we can do? And then build from there.
    // THE SIMPLEST POSSIBLE -> Incremental improvements

    AcReal processed_stencils[NUM_FIELDS][NUM_STENCILS] = {0};
#if !GEN_KERNEL
#include "kernel.out"
#endif
    /*
        for (int field = 0; field < NUM_FIELDS; ++field) {
            for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
                for (int height = 0; height < STENCIL_HEIGHT; ++height) {
                    for (int width = 0; width < STENCIL_WIDTH; ++width) {
                        const int i        = vertexIdx.x - STENCIL_ORDER / 2 + width;
                        const int j        = vertexIdx.y - STENCIL_ORDER / 2 + height;
                        const int k        = vertexIdx.z - STENCIL_ORDER / 2 + depth;
                        const int idx      = IDX(i, j, k);
                        const double value = vba.in[field][idx];

    #pragma unroll
                        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                            processed_stencils[field][stencil] +=
    stencils[stencil][depth][height] [width] * value;
                        }
                    }
                }
            }

            const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);
            // vba.out[field][idx] = processed_stencils[field][STENCIL_VALUE];
            // vba.out[field][idx] = processed_stencils[field][STENCIL_DERX];
            // vba.out[field][idx] = processed_stencils[field][STENCIL_DERXX];
            vba.out[field][idx] = processed_stencils[field][STENCIL_DERYZ];
        }
        */
}

#if 0
template <int step_number>
static __global__ void
solve(const int3 start, const int3 end, VertexBufferArray vba)
{
    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x + start.x,
        threadIdx.y + blockIdx.y * blockDim.y + start.y,
        threadIdx.z + blockIdx.z * blockDim.z + start.z,
    };
    const int3 globalVertexIdx = (int3){
        d_multigpu_offset.x + vertexIdx.x,
        d_multigpu_offset.y + vertexIdx.y,
        d_multigpu_offset.z + vertexIdx.z,
    };

    if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z)
        return;

    assert(vertexIdx.x < DCONST(AC_nx_max) && vertexIdx.y < DCONST(AC_ny_max) &&
           vertexIdx.z < DCONST(AC_nz_max));

    assert(vertexIdx.x >= DCONST(AC_nx_min) && vertexIdx.y >= DCONST(AC_ny_min) &&
           vertexIdx.z >= DCONST(AC_nz_min));

    assert(blockDim.x * blockDim.y * blockDim.z >= blockDim.x + STENCIL_ORDER);

    const int NUM_FIELDS = NUM_VTXBUF_HANDLES;

    // TODO test: what's the best we can do? And then build from there.
    // THE SIMPLEST POSSIBLE -> Incremental improvements

    AcReal processed_stencils[NUM_FIELDS][NUM_STENCILS] = {0};
    for (int field = 0; field < NUM_FIELDS; ++field) {
        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
            for (int height = 0; height < STENCIL_HEIGHT; ++height) {
                for (int width = 0; width < STENCIL_WIDTH; ++width) {
                    const int i        = vertexIdx.x - STENCIL_ORDER / 2 + width;
                    const int j        = vertexIdx.y - STENCIL_ORDER / 2 + height;
                    const int k        = vertexIdx.z - STENCIL_ORDER / 2 + depth;
                    const int idx      = IDX(i, j, k);
                    const double value = vba.in[field][idx];

#pragma unroll
                    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                        processed_stencils[field][stencil] += stencils[stencil][depth][height]
                                                                      [width] *
                                                              value;
                    }
                }
            }
        }

        const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);
        // vba.out[field][idx] = processed_stencils[field][STENCIL_VALUE];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERX];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERXX];
        vba.out[field][idx] = processed_stencils[field][STENCIL_DERYZ];
    }

    /*
    extern __shared__ AcReal smem[];
    AcReal processed_stencils[NUM_FIELDS][NUM_STENCILS] = {0};
    for (int field = 0; field < NUM_FIELDS; ++field) {
        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {

            const int3 baseIdx = (int3){
                blockIdx.x * blockDim.x + start.x,
                blockIdx.y * blockDim.y + start.y,
                blockIdx.z * blockDim.z + start.z,
            };
            const int smem_idx = threadIdx.x + threadIdx.y * blockDim.x +
                                 threadIdx.z * blockDim.x * blockDim.y;
            if (smem_idx < blockDim.x + STENCIL_ORDER) {
                for (int height = 0; height < blockDim.y + STENCIL_ORDER; ++height) {

                    const int smem_out   = smem_idx + height * (blockDim.x + STENCIL_ORDER);
                    const int3 vtxbuf_in = baseIdx -
                                           (int3){STENCIL_ORDER / 2, STENCIL_ORDER / 2,
                                                  STENCIL_ORDER / 2} +
                                           (int3){smem_idx, height, depth};
                    smem[smem_out] = vba.in[field][IDX(vtxbuf_in)];
                }
            }

            __syncthreads();
            for (int height = 0; height < STENCIL_HEIGHT; ++height) {
                for (int width = 0; width < STENCIL_WIDTH; ++width) {
                    const int i        = threadIdx.x + width;
                    const int j        = threadIdx.y + height;
                    const int idx      = i + j * (blockDim.x + STENCIL_ORDER);
                    const double value = smem[idx];
                    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                        processed_stencils[field][stencil] += stencils[stencil][depth][height]
                                                                      [width] *
                                                              value;
                    }
                }
            }
        }

        const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);
        // vba.out[field][idx] = processed_stencils[field][STENCIL_VALUE];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERX];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERXX];
        vba.out[field][idx] = processed_stencils[field][STENCIL_DERYZ];
    }
    */
}
#endif

static dim3 rk3_tpb(32, 4, 1);

AcResult
acKernelIntegrateSubstep(const cudaStream_t stream, const int step_number, const int3 start,
                         const int3 end, VertexBufferArray vba)
{
    // optimal integration dims in a hash table (hashes start & end indices) + timeout
    // Mesh dimensions available with DCONST(AC_nx) etc.

    // For field f
    //  For stencil depth k
    //      For stencil height j
    //          For stencil width i
    //              For stencils w
    //                  processed_stencils[w] = stencils[w][k][j][i] *
    //            field[f][vertexIdx.z - MIDZ + k][vertexIdx.y - MIDY + j][vertexIdx.x - MIDX +
    //            i];
    //
    // Even better, template magic unroll and skip if stencils[w][k][j][i] == 0
    //
    // Test also space filling! if time
    //
    // ALSO NEW MENTALITY! We do not have preprocessed any more! We have actual
    // physical stencils! User sets stencils and does stencil fetches!
    //
    // Stencil value = {
    //      [-1][0] = 60.0 / 30.0,
    //      [0][0] = 1.0,
    // };
    // Field a attach value, derx; // VALUE IS ALWAYS IMPLICITLY ATTACHED
    //
    // value(a) * something
    // derx(a) * something
    //
    // ALSO CONSISTENCY! a is either a value or handle, not both
    //
    // OTA TEMPLATE STEP NUMBER MYOS POIS! RK3 from DSL stdlib!

    ERRCHK_ALWAYS(step_number >= 0);
    ERRCHK_ALWAYS(step_number < 3);
    const dim3 tpb    = rk3_tpb; //(dim3){32, 4, 1};
    const size_t smem = (tpb.x + STENCIL_ORDER) * (tpb.y + STENCIL_ORDER) * sizeof(AcReal);

    const int3 n = end - start;
    const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                   (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                   (unsigned int)ceil(n.z / AcReal(tpb.z)));

    if (step_number == 0)
        solve<0><<<bpg, tpb, smem, stream>>>(start, end, vba);
    else if (step_number == 1)
        solve<1><<<bpg, tpb, smem, stream>>>(start, end, vba);
    else
        solve<2><<<bpg, tpb, smem, stream>>>(start, end, vba);

    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

AcResult
acKernelAutoOptimizeIntegration(const int3 start, const int3 end, VertexBufferArray vba)
{
// DEBUG UNROOL
#if GEN_KERNEL
    gen_kernel(); // Debug TODO REMOVE
    exit(0);
#endif
    //// DEBUG UNROOLL

    // Device info (TODO GENERIC)
#define REGISTERS_PER_THREAD (255)
#define MAX_REGISTERS_PER_BLOCK (65536)
#define MAX_THREADS_PER_BLOCK (1024)
#define WARP_SIZE (32)

    printf("Autotuning... ");
    // RK3
    dim3 best_dims(0, 0, 0);
    float best_time          = INFINITY;
    const int num_iterations = 10;

    for (int z = 1; z <= 1; ++z) { // TODO CHECK Z GOES ONLY TO 1
        for (int y = 1; y <= MAX_THREADS_PER_BLOCK; ++y) {
            for (int x = 4; x <= MAX_THREADS_PER_BLOCK; x += 4) {

                if (x > end.x - start.x || y > end.y - start.y || z > end.z - start.z)
                    break;
                if (x * y * z > MAX_THREADS_PER_BLOCK)
                    break;

                if (x * y * z * REGISTERS_PER_THREAD > MAX_REGISTERS_PER_BLOCK)
                    break;

                if (((x * y * z) % WARP_SIZE) != 0)
                    continue;

                if ((x * y * z) < x + STENCIL_ORDER) // WARNING NOTE: Only use if using smem
                    continue;

                const dim3 tpb(x, y, z);
                const int3 n = end - start;
                const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                               (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                               (unsigned int)ceil(n.z / AcReal(tpb.z)));
                const size_t smem = (tpb.x + STENCIL_ORDER) * (tpb.y + STENCIL_ORDER) *
                                    sizeof(AcReal);

                cudaDeviceSynchronize();
                if (cudaGetLastError() != cudaSuccess) // resets the error if any
                    continue;

                // printf("(%d, %d, %d)\n", x, y, z);

                cudaEvent_t tstart, tstop;
                cudaEventCreate(&tstart);
                cudaEventCreate(&tstop);

                cudaEventRecord(tstart); // ---------------------------------------- Timing start
                for (int i = 0; i < num_iterations; ++i)
                    solve<2><<<bpg, tpb, smem, 0>>>(start, end, vba);

                cudaEventRecord(tstop); // ----------------------------------------- Timing end
                cudaEventSynchronize(tstop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, tstart, tstop);

                ERRCHK_CUDA_KERNEL_ALWAYS();
                printf("(%d, %d, %d): %.4g ms\n", x, y, z, (double)milliseconds / num_iterations);
                fflush(stdout);
                if (milliseconds < best_time) {
                    best_time = milliseconds;
                    best_dims = tpb;
                }
            }
        }
    }
    printf("\x1B[32m%s\x1B[0m\n", "OK!");
    fflush(stdout);
#if 1 || AC_VERBOSE // DEBUG always on
    printf("Auto-optimization done. The best threadblock dimensions for rkStep: (%d, %d, %d) "
           "%f "
           "ms\n",
           best_dims.x, best_dims.y, best_dims.z, double(best_time) / num_iterations);
#endif

    rk3_tpb = best_dims;

    // Failed to find valid thread block dimensions
    ERRCHK_ALWAYS(rk3_tpb.x * rk3_tpb.y * rk3_tpb.z > 0);

    return AC_SUCCESS;
}

/*
#if USE_SMEM
            extern __shared__ AcReal smem[];

            const int3 baseIdx = (int3){
                blockIdx.x * blockDim.x + start.x,
                blockIdx.y * blockDim.y + start.y,
                blockIdx.z * blockDim.z + start.z,
            };
            for (int height = 0; height < blockDim.y + STENCIL_ORDER; ++height) {
                const int smem_idx = threadIdx.x + threadIdx.y * blockDim.x +
                                     threadIdx.z * blockDim.x * blockDim.y;
                if (smem_idx < blockDim.x + STENCIL_ORDER) {
                    const int smem_out   = smem_idx + height * (blockDim.y + STENCIL_ORDER);
                    const int3 vtxbuf_in = baseIdx -
                                           (int3){STENCIL_ORDER / 2, STENCIL_ORDER / 2,
                                                  STENCIL_ORDER / 2} +
                                           (int3){smem_idx, height, depth};
                    smem[smem_out] = vba.in[field][IDX(vtxbuf_in)];
                }
            }

            __syncthreads();
#endif
*/