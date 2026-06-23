import numpy as np
import math as mt


def dq_add(dq1, dq2):
    # esta suma no me da un cuaternion dual  que este dentro de SE(3), entocnes debo tomar enceunta 
    # las operaciones de inversa
    return (dq1 + dq2)

def q_conj(q):
    return np.array([ q[0], -q[1], -q[2], -q[3] ], dtype=float)

def q_normalize(q):
    """ Normaliza un cuaternion q = [w, x, y, z]. """
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-10 else np.array([1, 0, 0, 0])

def q_mul(a, b):
        w1,x1,y1,z1 = a
        w2,x2,y2,z2 = b
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 + y1*w2 + z1*x2 - x1*z2,
            w1*z2 + z1*w2 + x1*y2 - y1*x2
        ], dtype=float)

def dq_from_tr(t, r):
    # funcions para crear un cuaternion dual a partir de un vector de traslacion y un quaternion.
    dq = np.concatenate((r, 0.5 * q_mul(t, r)))
    return dq_unit_normalize(dq)


# def dq_normalize(dq):
#     """ Normaliza un cuaternion dual asegurando que la parte real es unitario. """
#     qr = q_normalize(dq[:4])
#     qd = dq[4:] / np.linalg.norm(qr) if np.linalg.norm(qr) > 1e-8 else dq[4:]
#     return np.hstack((qr, qd))

def dq_conjugate(dq):
    """
    Conjugado de un dq.
    conj(dq) = conj(qr) + eps*conj(qd).
    conj(qr) = [w, -x, -y, -z].
    """
    qr1, qd1 = dq[:4], dq[4:]        
    return np.concatenate([q_conj(qr1), q_conj(qd1)])

def dq_dot(dq1, dq2):
    """
    'Dot product' usual en R^8. No confundir con producto de cuaterniones.
    """
    return np.dot(dq1, dq2)
    

def dq_mul(dq1, dq2):
    """
    Producto de cuaterniones duales:
    dq1 = (qr1, qd1),  dq2 = (qr2, qd2).
    dq1*dq2 = (qr1*qr2,  qr1*qd2 + qd1*qr2).
    """
    qr1, qd1 = dq1[:4], dq1[4:]
    qr2, qd2 = dq2[:4], dq2[4:]    

    qr = q_mul(qr1, qr2)
    qd = q_mul(qr1, qd2) + q_mul(qd1, qr2)
    return np.concatenate([qr, qd])

def dq_inverse(dq):
    """
    dq^-1 = conj(dq) / (dq * conj(dq))
    Suponiendo que dq * conj(dq) != 0 (inversible).
    """
    conj_dq = (dq_conjugate(dq))
    denom = dq_mul(dq, conj_dq)  # = [s, 0,0,0,  something dual ]


    # en cuaterniones duales, dq*conj(dq) se reduce a un escalar real>0 + eps*(algo)
    # Lo importante es la parte real 's' = ||qr||^2.
    # Se asume que la parte real no es cero.
    s = denom[0]  # w
    if abs(s) < 1e-14:
        raise ValueError("No es invertible: dq * conj(dq) tiene parte real ~ 0")

    # Dividimos conj(dq) entre s. (ignoramos la parte dual del denominador
    # porque eps^2=0 => (1/s) - (dualPart)/s^2 no produce corrección de orden 2).

    return conj_dq /s

# def inv_vector

# def dq_inverse(dq):
#     """
#     dq^-1 = conj(dq) / (dq * conj(dq))
#     Suponiendo que dq * conj(dq) != 0 (inversible).
#     """
#     qr = dq[:4]
#     qd = dq[4:]
    

#     conj_dq = (dq_conjugate(dq))
#     denom = dq_mul(conj_dq, conj_dq)  # = [s, 0,0,0,  something dual ]
#     # denom = dq_unit_normalize(denom)
#     # print("denom:", denom)

#     # en cuaterniones duales, dq*conj(dq) se reduce a un escalar real>0 + eps*(algo)
#     # Lo importante es la parte real 's' = ||qr||^2.
#     # Se asume que la parte real no es cero.
#     s = denom[0]  # w
#     if abs(s) < 1e-14:
#         raise ValueError("No es invertible: dq * conj(dq) tiene parte real ~ 0")

#     # Dividimos conj(dq) entre s. (ignoramos la parte dual del denominador
#     # porque eps^2=0 => (1/s) - (dualPart)/s^2 no produce corrección de orden 2).


#     return conj_dq / (np.dot(dq,dq))

def q_norm_sqr(q):
    """
    Norma al cuadrado de un cuaternion q = [w, x, y, z].
    Retorna w^2 + x^2 + y^2 + z^2.
    """
    return np.dot(q, q)

def q_inverse(q):
    """
    Inversa de un cuaternion q. 
    q^{-1} = conj(q) / ||q||^2,
    asumiendo ||q|| != 0.
    """
    conj_q = q_conj(q)
    norm_q2 = q_norm_sqr(q)
    if abs(norm_q2) < 1e-15:
        raise ValueError("El cuaternion no es invertible (norma ~ 0).")
    return conj_q / norm_q2

# def dq_inverse(dq):
    """
    Inversa de d = a + eps*b en el anillo de cuaterniones duales,
    donde a, b ∈ R^4 (cuaterniones).
    
    Retorna (a_inv, b_inv).
    
    Fórmula:
       (a + eps*b)^-1 = a^-1 - eps * ( a^-1 b a^-1 ).
       
    Asumiendo a es invertible como cuaternion (||a|| != 0).

    """
    a = dq[:4]
    b = dq[4:]
    # a_inv = q_inverse(a)
    # # Parte dual invertida = - (a^-1 b a^-1)
    # b_inv = - q_mul( q_mul(a_inv, b), a_inv )

    qr = q_inverse(a)
    qd = -q_mul(q_mul(qr,b),qr)
    return (np.concatenate([qr, qd]))

##############################################################################
#               2) Del 'twist' a DQ puro (X) y viceversa                     #
##############################################################################

def twist_to_pureDQ(twist):
    """
    Dado un twist = (omega_x, omega_y, omega_z, v_x, v_y, v_z),
    construye el cuaternion dual 'puro':
      X = [0, omega_x, omega_y, omega_z,  0, v_x, v_y, v_z].
    """
    omega = twist[:3]
    v = twist[3:]
    return np.array([
        0.0, omega[0], omega[1], omega[2],
        0.0, v[0],     v[1],     v[2]
    ], dtype=float)

def pureDQ_to_twist(X):
    """
    Inverso de la anterior. Extrae (omega, v) de
    X = [0, omega_x, omega_y, omega_z, 0, v_x, v_y, v_z].
    """
    # Asumimos que X es 'puro' (parte escalar real=0).
    return np.concatenate([ X[1:4], X[5:8] ])

##############################################################################
#         3) Cayley map e inversa, en el anillo de Cuaterniones Duales       #
##############################################################################

def cayley_map(twist):
    """
    Transformada de Cayley (Selig) aplicada a un twist en R^6,
    pero *directamente* como dq:  M = (1 + X) * (1 - X)^-1
    donde X es el DQ puro asociado al twist.
    Retorna un dq unitario (representa un elemento de SE(3)).
    """
    X = twist_to_pureDQ(twist)
    # A = (1 + X)
    # B = (1 - X)
    one = np.array([1.0,0,0,0,0,0,0,0], dtype=float)

    A = dq_add(one, X)      # 1 + X
    B = dq_add(one, -X)     # 1 - X

    # Inversa de B en el anillo
    B_inv = dq_inverse(B)
    M = dq_mul(A, B_inv)

    # Normalmente este M debería ser 'unitario'. Pero podemos normalizarlo
    # por si hay pequeñas desviaciones numéricas:
    #  => la parte real qr de M debe tener norma 1.
    # Si deseas forzar unidad:
    M = dq_unit_normalize(M)

    #M = dq_normalize(M)

    return M

def cayley_map_inv(M):
    """
    Inversa de la transformada de Cayley:
       X = (M - 1) * (M + 1)^-1
    con todo en cuaterniones duales.
    Luego se traduce X a (omega, v).
    """
    one = np.array([1.0,0,0,0,0,0,0,0], dtype=float)
    # M - 1
    Mm = dq_add(M, - one)
    # M + 1
    Mp = dq_add(M, one)
    # (M + 1)^-1
    Mp_inv = dq_inverse(Mp)
    X = dq_mul( Mm, Mp_inv )  
    # Extraer twist
    twist = pureDQ_to_twist(X)
    
    return twist

def dq_unit_normalize(dq):
    """
    Normaliza un cuaternion dual
    """
    qr = dq[:4]
    norm_qr = np.linalg.norm(qr)
    if norm_qr < 1e-15:
        # mostraria el el cuaternion no puede normalizarse y no es valido
        print(dq, "no es un cuaternion dual")
        return dq
    return dq / norm_qr


##############################################################################
#                       Ejemplo de uso y verificación                         #
##############################################################################

if __name__ == "__main__":

    
    # Twist cualquiera
    twist_example = np.array([0.7, -0.4, 0.3,   1.2, 0.6, -0.8], dtype=float)
    print("Twist original =", twist_example)

    # 1) Cayley(twist) -> dq
    dq_cay = cayley_map(twist_example)
    print("\nDual Quaternion = Cayley_map(twist):\n", dq_cay)

    # 2) Inversa => recuperamos el twist
    twist_rec = cayley_map_inv(dq_cay)
    print("\nTwist recuperado via Cayley^-1(dq):\n", twist_rec)

    # 3) Diferencia
    diff = twist_rec - twist_example
    print("\nDiferencia (rec - original) =", diff)


    dq_cay = np.array([-0.6946, 0.2523, -0.6092, 0.2877, -0.2321, 0.1127, 0.1484, -0.3450])
    # dq_cay = dq_unit_normalize(dq_cay)

    t = [0, 0.0, -0.0, 0.0]
    r = [1, 0.0, -0.0, 0.0]

    dq_cay = dq_from_tr(t,r)
    print("dq_prueba:", dq_cay)



    twist_rec = cayley_map_inv(dq_cay)
    print("Cayley inverse: ",twist_rec)

    dq_cay_2 = cayley_map(twist_rec)
    print("Cayley map    : ",dq_cay_2)
    print("dq original   : ",dq_cay)

    diff = dq_cay_2-dq_cay
    print("Cayley diff   : ",diff)

    #dq_cay = np.array([0.68776371,  0.5907173,  -0.33755274,  0.25316456, -0.15189873,  1.01265823,   0.50632911, -0.67510549])
    
    #diff = dq_mul(dq_cay,dq_conjugate(dq_cay))
    #print("Cayley diff   : ",diff)



    dq_1 = np.array([-0.69459, 0.2523, -0.60919, 0.2877, -0.23201, 0.11274, 0.1484, -0.34496])
    dq_1 = dq_unit_normalize(dq_1)
    dq_2 = np.array([-0.68665, 0.27665, -0.59863, 0.30598, -0.22718, 0.11498, 0.13296, -0.35364])
    dq_2 = dq_unit_normalize(dq_2)

    dq_res = dq_mul(dq_2,dq_conjugate(dq_1))
    dq_res = dq_unit_normalize(dq_res)

    dq_res_twist = cayley_map_inv(dq_res)
    print("Cayley inverse: ",dq_res_twist)

    dq_remap = cayley_map(dq_res_twist)
    print("Cayley remap  : ",dq_remap)

    print("dq_res:       ",dq_res)
    print(dq_res - dq_remap)

