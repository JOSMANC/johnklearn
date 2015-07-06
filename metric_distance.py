def euclidean_distance(xu, xk):
    '''
    INPUT:
        - xu: 1d np array
        - xk: 2d np array
    OUTPUT:
        - 1d array of distances
    '''
    return np.linalg.norm((xk-xu), axis=1)


def cosine_distance(xu, xk):
    '''
    INPUT:
        - xu: 1d np array
        - xk: 2d np array
    OUTPUT:
        - 1d array of distances
    '''
    return 1.-(xk.dot(xu)/(np.linalg.norm(xu)*np.linalg.norm(xk, axis=1)))