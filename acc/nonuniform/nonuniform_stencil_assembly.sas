


Preprocessed Scalar
value(in Scalar vertex)
{
    return vertex[vertexIdx];
}


Preprocessed Scalar
grid_function(const AcReal a_grid, const AcReal zeta, const int der_degree)
{

    //Using now sinh() as an example.
    if (der_degree == 0) {
        sinh(AcReal*zeta);
    } else if (der_degree == 1) {
        a_grid*cosh(AcReal*zeta);
    } else if (der_degree == 2) {
        (a_grid*a_grid)*sinh(AcReal*zeta);
    }

}

Preprocessed Vector
dgrid(Vector zeta, const int der_degree)
{
    return (Vector){
    grid_function(a_grid,(zeta.x - zeta_star.x), der_degree)/
    (grid_function(a_grid,(MAX_GRID_POINTS_X - zeta_star.x), 0) 
   + grid_function(a_grid,(zeta_star.x - MIN_GRID_POINTS_X), 0)), 

    grid_function(a_grid,(zeta.y - zeta_star.y), der_degree)/
    (grid_function(a_grid,(MAX_GRID_POINTS_Y - zeta_star.y), 0) 
   + grid_function(a_grid,(zeta_star.y - MIN_GRID_POINTS_Y), 0)),

    grid_function(a_grid,(zeta.z - zeta_star.z), der_degree)/
    (grid_function(a_grid,(MAX_GRID_POINTS_Z - zeta_star.z), 0) 
   + grid_function(a_grid,(zeta_star.z - MIN_GRID_POINTS_z), 0))
    };
}

Preprocessed Vector
dzeta1()
{

    Vector zeta = (Vector) { globalVertexIdx.x, globalVertexIdx.y, globalVertexIdx.z};

    Vector _dgrid = dgrid(zeta, 1);

    return  (Vector){Scalar(1.0)/_dgrid.x, Scalar(1.0)/_dgrid.y, Scalar(1.0)/_dgrid.z}; 
}

Preprocessed Vector
dzeta2()
{
    Vector zeta = (Vector) { globalVertexIdx.x, globalVertexIdx.y, globalVertexIdx.z};

    Vector _dgrid = dgrid(zeta, 1);
    Vector _ddgrid = dgrid(zeta, 2);

    return  (Vector){-_ddgrid.x/(_dgrid.x), -_ddgrid.y/(_dgrid.y), -_ddgrid.z/(_dgrid.z)}; 

}


//For nonuniform grid. 
Preprocessed Vector
gradient(in Scalar vertex)
{
    return (Vector){dzeta1().x * derx(vertexIdx, vertex),
                    dzeta1().y * dery(vertexIdx, vertex),
                    dzeta1().z * derz(vertexIdx, vertex)};
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
    //Cross derivatives not supported!
    hessian.row[0] = (Vector){dzeta1x_t_dzeta1x*derxx(vertexIdx, vertex) 
                            + dzeta2x_p_dzeta1x*derx(vertexIdx, vertex), Scalar(0.0), Scalar(0.0)};
    hessian.row[1] = (Vector){hessian.row[0].y, dzeta1y_t_dzeta1y*deryy(vertexIdx, vertex) 
                                              + dzeta2y_p_dzeta1y*dery(vertexIdx, vertex), Scalar(0.0)};
    hessian.row[2] = (Vector){hessian.row[0].z, hessian.row[1].z, dzeta1z_t_dzeta1z*derzz(vertexIdx, vertex)  
                                                                + dzeta2z_p_dzeta1z*derz(vertexIdx, vertex)};

    return hessian;
}
