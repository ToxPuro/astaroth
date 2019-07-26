Preprocessed Scalar
value(in Scalar vertex)
{
    return vertex[vertexIdx];
}

#if LNONUNIFORM
Preprocessed Scalar
grid_function(const Scalar a_grid, const Scalar zeta, const int der_degree)
{

    //Using now sinh() as an example.
    if (der_degree == 0) {
       return sinh(zeta);
    } else if (der_degree == 1) {
       return a_grid*cosh(zeta);
    } else if (der_degree == 2) {
       return (a_grid*a_grid)*sinh(zeta);
    }

}

Preprocessed Vector
dgrid(Vector zeta, const int der_degree)
{
    return (Vector){
    grid_function(a_grid,(zeta.x - zeta_star.x), der_degree)/
    (grid_function(a_grid,(Scalar(DCONST_INT(AC_nx)) - Scalar(1) - zeta_star.x), 0) 
   + grid_function(a_grid,(zeta_star.x - Scalar(0)), 0)), 

    grid_function(a_grid,(zeta.y - zeta_star.y), der_degree)/
    (grid_function(a_grid,(Scalar(DCONST_INT(AC_ny)) - Scalar(1) - zeta_star.y), 0) 
   + grid_function(a_grid,(zeta_star.y - Scalar(0)), 0)),

    grid_function(a_grid,(zeta.z - zeta_star.z), der_degree)/
    (grid_function(a_grid,(Scalar(DCONST_INT(AC_nz)) - Scalar(1) - zeta_star.z), 0) 
   + grid_function(a_grid,(zeta_star.z - Scalar(0)), 0))
    };
}

//First derivative of the grid
Preprocessed Vector
dzeta1()
{

    Vector zeta = (Vector) { globalVertexIdx.x, globalVertexIdx.y, globalVertexIdx.z};

    Vector _dgrid = dgrid(zeta, 1);

    return  (Vector){Scalar(1.0)/_dgrid.x, Scalar(1.0)/_dgrid.y, Scalar(1.0)/_dgrid.z}; 
}

//Second derivative of the grid
Preprocessed Vector
dzeta2()
{
    Vector zeta = (Vector) { globalVertexIdx.x, globalVertexIdx.y, globalVertexIdx.z};

    Vector _dgrid = dgrid(zeta, 1);
    Vector _ddgrid = dgrid(zeta, 2);

    return  (Vector){-_ddgrid.x/(_dgrid.x), -_ddgrid.y/(_dgrid.y), -_ddgrid.z/(_dgrid.z)}; 

}

//Non-uniform grid derivative. 
Preprocessed Vector
gradient(in Scalar vertex)
{
    return (Vector){dzeta1().x * derx(vertexIdx, vertex),
                    dzeta1().y * dery(vertexIdx, vertex),
                    dzeta1().z * derz(vertexIdx, vertex)};
}
#else
Preprocessed Vector
gradient(in Scalar vertex)
{
    return (Vector){derx(vertexIdx, vertex),
                    dery(vertexIdx, vertex),
                    derz(vertexIdx, vertex)};
}
#endif


#if LUPWD

Preprocessed Scalar
der6x_upwd(in Scalar vertex)
{
    Scalar inv_ds = DCONST_REAL(AC_inv_dsx);

    return (Scalar){  Scalar(1.0/60.0)*inv_ds* (
                    - Scalar(20.0)* vertex[vertexIdx.x,   vertexIdx.y, vertexIdx.z]
                    + Scalar(15.0)*(vertex[vertexIdx.x+1, vertexIdx.y, vertexIdx.z] + vertex[vertexIdx.x-1, vertexIdx.y, vertexIdx.z])
                    - Scalar( 6.0)*(vertex[vertexIdx.x+2, vertexIdx.y, vertexIdx.z] + vertex[vertexIdx.x-2, vertexIdx.y, vertexIdx.z])
                    +               vertex[vertexIdx.x+3, vertexIdx.y, vertexIdx.z] + vertex[vertexIdx.x-3, vertexIdx.y, vertexIdx.z])};
}

Preprocessed Scalar
der6y_upwd(in Scalar vertex)
{
    Scalar inv_ds = DCONST_REAL(AC_inv_dsy);

    return (Scalar){ Scalar(1.0/60.0)*inv_ds* (
                    -Scalar( 20.0)* vertex[vertexIdx.x,   vertexIdx.y, vertexIdx.z]
                    +Scalar( 15.0)*(vertex[vertexIdx.x, vertexIdx.y+1, vertexIdx.z] + vertex[vertexIdx.x, vertexIdx.y-1, vertexIdx.z])
                    -Scalar(  6.0)*(vertex[vertexIdx.x, vertexIdx.y+2, vertexIdx.z] + vertex[vertexIdx.x, vertexIdx.y-2, vertexIdx.z])
                    +               vertex[vertexIdx.x, vertexIdx.y+3, vertexIdx.z] + vertex[vertexIdx.x, vertexIdx.y-3, vertexIdx.z])};
}

Preprocessed Scalar
der6z_upwd(in Scalar vertex)
{
    Scalar inv_ds = DCONST_REAL(AC_inv_dsz);

    return (Scalar){ Scalar(1.0/60.0)*inv_ds* (
                    -Scalar( 20.0)* vertex[vertexIdx.x,   vertexIdx.y, vertexIdx.z]
                    +Scalar( 15.0)*(vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z+1] + vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z-1])
                    -Scalar(  6.0)*(vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z+2] + vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z-2])
                    +               vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z+3] + vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z-3])};
}

#endif

#if LNONUNIFORM

// Experimental method 
Preprocessed Scalar
derxy_bruteforce(in Scalar vertex)
{
    const Scalar coefficients[] =  {0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
    const Scalar inv_ds = DCONST_REAL(AC_inv_dsx)*DCONST_REAL(AC_inv_dsy);

    Scalar derivative = 0.0;
    for (int i= -3; i <= 3; ++i) {
        for (int j= -3; j <= 3; ++j) {
            Scalar coeff_ij = coefficients[abs(i)] * coefficients[abs(j)];
            sum_tot = sum_tot + coeff_ij*vertex[vertexIdx.x+i, vertexIdx.y+j, vertexIdx.z];
        }
    }   

    return derivative;
}

// Experimental method 
Preprocessed Scalar
derxz_bruteforce(in Scalar  vertex)
{
    const Scalar coefficients[] = {0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
    const Scalar inv_ds = DCONST_REAL(AC_inv_dsx)*DCONST_REAL(AC_inv_dsz);

    Scalar derivative = 0.0;
    for (int i= -3; i <= 3; ++i) {
        for (int j= -3; j <= 3; ++j) {
            Scalar coeff_ij = coefficients[abs(i)] * coefficients[abs(j)];
            sum_tot = sum_tot + coeff_ij*vertex[vertexIdx.x+i, vertexIdx.y, vertexIdx.z+j];
        }
    }   

    return derivative;
}

// Experimental method 
Preprocessed Scalar
deryz_bruteforce(in Scalar  vertex)
{
    const Scalar coefficients[] = {0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
    const Scalar inv_ds = DCONST_REAL(AC_inv_dsy)*DCONST_REAL(AC_inv_dsz);

    Scalar derivative = 0.0;
    for (int i= -3; i <= 3; ++i) {
        for (int j= -3; j <= 3; ++j) {
            Scalar coeff_ij = coefficients[abs(i)] * coefficients[abs(j)];
            sum_tot = sum_tot + coeff_ij*vertex[vertexIdx, vertexIdx.y+i, vertexIdx.z+j];
        }
    }   

    return derivative;
}


Preprocessed Matrix
hessian(in Scalar vertex)
{
    Matrix hessian;

    Scalar dzeta1x_t_dzeta1x = dzeta1().x * dzeta1().x;
    Scalar dzeta2x_p_dzeta1x = dzeta2().x / dzeta1().x;
    Scalar dzeta1y_t_dzeta1y = dzeta1().y * dzeta1().y;
    Scalar dzeta2y_p_dzeta1y = dzeta2().y / dzeta1().y;
    Scalar dzeta1z_t_dzeta1z = dzeta1().z * dzeta1().z;
    Scalar dzeta2z_p_dzeta1z = dzeta2().z / dzeta1().z;

    //derxy(vertexIdx, vertex), derxz(vertexIdx, vertex), deryz(vertexIdx, vertex) = Scalar(0.0) 
    //We will require a stencil which does not assume an equidistant grid
    hessian.row[0] = (Vector){dzeta1x_t_dzeta1x*derxx(vertexIdx, vertex) 
                            + dzeta2x_p_dzeta1x*derx(vertexIdx, vertex), 
                             (dzeta1().x * dzeta1().y)*derxy_bruteforce(vertex), 
                             (dzeta1().x * dzeta1().z)*derxz_bruteforce(vertex)};
    hessian.row[1] = (Vector){hessian.row[0].y, 
                              dzeta1y_t_dzeta1y*deryy(vertexIdx, vertex) 
                            + dzeta2y_p_dzeta1y*dery(vertexIdx, vertex), 
                             (dzeta1().y * dzeta1().z)*deryz_bruteforce(vertex)};
    hessian.row[2] = (Vector){hessian.row[0].z, hessian.row[1].z, 
                              dzeta1z_t_dzeta1z*derzz(vertexIdx, vertex)  
                            + dzeta2z_p_dzeta1z*derz(vertexIdx, vertex)};

    return hessian;
}
#else
Preprocessed Matrix
hessian(in Scalar vertex)
{
    Matrix hessian;

    hessian.row[0] = (Vector){derxx(vertexIdx, vertex), derxy(vertexIdx, vertex), derxz(vertexIdx, vertex)};
    hessian.row[1] = (Vector){hessian.row[0].y,       deryy(vertexIdx, vertex), deryz(vertexIdx, vertex)};
    hessian.row[2] = (Vector){hessian.row[0].z,       hessian.row[1].z,       derzz(vertexIdx, vertex)};

    return hessian;
}
#endif
