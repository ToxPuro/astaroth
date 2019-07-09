Preprocessed Scalar
value(in Scalar vertex)
{
    return vertex[vertexIdx];
}

//
Preprocessed Vector
dzeta1()
{

 globalVertexIdx.x

    return  (Vector){ , , }; //Testing first with just one derection
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
