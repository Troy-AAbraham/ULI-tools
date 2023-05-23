import numpy as np

def numerical_wave_drag_approx(area_dist, length, ref_area):
    
    # number of interior points
    n = len(area_dist[:,0]) - 2
    
    # non-dimensional x locations (0 > x < 1)
    i = np.arange(1,n+1)
    k = i/(n+1) 
    
    # set array sizes
    P = np.zeros((n,n))
    C = np.zeros((n))

    # nose area
    N = area_dist[0,1]
    # base area
    B = area_dist[-1,1]
    
    u = (1/np.pi)*(np.arccos(1.0 - 2.0*k) - 2*(1 - 2*k)*np.sqrt(k*(1 - k)))
    C = (area_dist[1:-1,1] - N) - (B - N)*u[:]
    C_mat1 = np.tile(C,(n,1))
    C_mat2 = np.tile(np.array([C]).T,(1,n))
    C_mat_mult = C_mat1*C_mat2
    
    # build matrix distributions of x and y values for use in p matrix
    x_mat = np.tile(k,(n,1))
    y_mat = np.tile(np.array([k]).T,(1,n))
    
    # follows variables defined by the Eminton algorithm
    C1 = -0.5*(x_mat - y_mat)*(x_mat - y_mat)
    C2 = x_mat + y_mat - 2*x_mat*y_mat
    C3 = 2*np.sqrt(x_mat*y_mat*(1 - x_mat)*(1 - y_mat))
    
    # sets error handling for expected zero division and multiplication on diagnals
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        P = C1*np.log((C2 - C3)/(C2 + C3)) + C2*C3
    
    # condition for the diagnol indicies
    P[np.diag_indices_from(P)] = C2[np.diag_indices_from(P)]*C3[np.diag_indices_from(P)]
    
    # inverts p matrix
    f = np.linalg.inv(P)
    
    sum_mat = C_mat_mult*f
    sum_i_j = np.sum(sum_mat)
    # print('Sum: ', sum_i_j)
    
    I = abs((4/np.pi)*(B - N)*(B - N) + np.pi*sum_i_j)
    # return (1/(length*length))*I*(0.5)*rho*U*U
    
    return (1/(length*length))*I/ref_area
    # return I/ref_area