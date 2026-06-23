import numpy as np

########################################## Operaciones 
def quaternion_conjugate(q):
    """ Calcula el conjugado de un cuaternión q = [w, x, y, z]. """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def dq_inverse(dq):
    """
    dq^-1 = conj(dq) / (dq * conj(dq))
    Suponiendo que dq * conj(dq) != 0 (inversible).
    """
    conj_dq = dq_conjugate(dq)
    denom = dq_mul(dq, conj_dq)  # = [s, 0,0,0,  something dual ]
    #print("denom: ",denom)

    # en cuaterniones duales, dq*conj(dq) se reduce a un escalar real>0 + eps*(algo)
    # Lo importante es la parte real 's' = ||qr||^2.
    # Se asume que la parte real no es cero.
    s = denom[0]  # w
    if abs(s) < 1e-14:
        raise ValueError("No es invertible: dq * conj(dq) tiene parte real ~ 0")

    # Dividimos conj(dq) entre s. (ignoramos la parte dual del denominador
    # porque eps^2=0 => (1/s) - (dualPart)/s^2 no produce corrección de orden 2).
    return conj_dq / s

def dq_conjugate(q):
    """
    conjugado de un cauternion dual
    """
    return np.array([q[0], -q[1], -q[2], -q[3], q[4], -q[5], -q[6], -q[7]])

def normalize_quaternion(q):
    """ Normaliza un cuaternión q = [w, x, y, z]. """
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-10 else np.array([1, 0, 0, 0])

def normalize_dq(dq):
    """ Normaliza un cuaternión dual asegurando que la parte real es unitario. """
    qr = normalize_quaternion(dq[:4])
    qd = dq[4:] / np.linalg.norm(qr) if np.linalg.norm(qr) > 1e-8 else dq[4:]
    return np.hstack((qr, qd))

def quaternion_inverse(q):
    """ Calcula la inversa de un cuaternión unitario. """
    inv = quaternion_conjugate(q) / np.linalg.norm(q)
    return inv

def quaternion_multiply(q1, q2):
    """ Multiplicación de dos cuaterniones q1 y q2. """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    res = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    #res = normalize_quaternion(res)
    return res


def dual_quaternion_multiply(dq1, dq2):
    """
    Calcula el producto de dos cuaterniones duales dq1 y dq2.    
    """
    qr1, qd1 = dq1[:4], dq1[4:]
    qr2, qd2 = dq2[:4], dq2[4:]
    real_part = quaternion_multiply(qr1, qr2)
    dual_part = quaternion_multiply(qr1, qd2) + quaternion_multiply(qd1, qr2)
    return np.hstack((real_part, dual_part))
##############################################################


#########################MAPEO#####################################

def exp_cayley(phi):
    """
    Aplica el mapeo de Cayley a un vector 3D `phi` para obtener un cuaternión unitario q.
    
    Entrada:
        phi: Vector 3D (array de 3 elementos).
    
    Salida:
        q: Cuaternión unitario (array de 4 elementos) [q_s, q_v]
    """
    phi = phi[1:]
    norm_phi = np.linalg.norm(phi)  # ||phi||
    scale = 1.0 / np.sqrt(1 + norm_phi**2)  # 1 / sqrt(1 + ||phi||^2)
    
    q_s = scale  # Componente escalar del cuaternión
    q_v = scale * phi  # Componente vectorial del cuaternión
    
    return np.hstack((q_s, q_v))  # Retorna q = [q_s, q_v]

def log_cayley(q):
    """
    Aplica el mapeo de Cayley inverso a un cuaternión `q` para recuperar el vector 3D `phi`.
    
    Entrada:
        q: Cuaternión unitario (array de 4 elementos) [q_s, q_v].
    
    Salida:
        phi: Vector 3D (array de 3 elementos).
    """
    q_s = q[0]  # Parte escalar del cuaternión
    q_v = q[1:]  # Parte vectorial del cuaternión
    
    if np.abs(q_s) < 1e-8:  # Evitar divisiones por cero
        raise ValueError("El cuaternión no puede ser invertido: q_s es demasiado pequeño.")
    
    phi = q_v / q_s  # Aplicando la 
    return np.hstack((0.0,phi))

def log_cayley_dual(dq):
    """ Aplica la transformada inversa de Cayley a un cuaternión dual. """
    qr = normalize_quaternion(dq[:4])  # Solo normalizamos la parte real
    qd = (dq[4:])  # La parte dual se deja sin normalizar

 
    omega = log_cayley(qr)

    q_plus = qr + np.array([1.0, 0.0, 0.0, 0.0])  # q + 1

    q_plus_inv = quaternion_inverse(q_plus)  # (q + 1)^{-1}
  
    

    v =  2.0*quaternion_multiply(q_plus_inv, quaternion_multiply(qd, q_plus_inv))
    #v = normalize_quaternion(v)
    v [0] = 0

    print(omega)
    print(v)
    return np.hstack((omega, v))

def exp_cayley_dual(twist_dual):
    """
    Aplica el mapeo de Cayley a un twist dual.
    
    Entrada:
        twist_dual: Un array de 8 elementos que representa un twist dual [u, u']
                    - Los primeros 4 elementos son la parte rotacional u.
                    - Los últimos 4 elementos son la parte traslacional u'.
    
    Salida:
        cuaternión dual mapeado mediante Cayley [qr, qd]
    """
    u = twist_dual[:4]  # Parte rotacional
    u_prima = twist_dual[4:]  # Parte traslacional

    qr = exp_cayley(u)
    qr = normalize_quaternion(qr)
    
    qr = -qr   # esto no se porque debo ahcer para el ejemplo que me dio Daniel


    # Calculamos (1 + u)
    q_plus = np.array([1.0, 0.0, .0, .0]) + u
    # q_plus = normalize_quaternion(q_plus)
    # print ("q_plus: ", q_plus)

    # q_plus_inv = quaternion_inverse(q_plus)  # (1 + u)^{-1}

    # Calculamos la parte dual
    numerator = 2*quaternion_multiply(q_plus, quaternion_multiply(u_prima, q_plus))
    denominator = (1 + np.dot(u,u)) ** 2
    print(numerator)
    print(denominator)
    qd = normalize_quaternion(numerator) /denominator

    return np.hstack((qr, qd))



# ---------- PRUEBA ----------
dq = np.array([-0.6922, 0.2475, -0.6111, 0.2936, -0.1745, 0.1540, 0.0885, -0.3571])
#dq = np.array([0.666, 0.702, 0.0776, -0.2356, -0.1014, 0.1421, -0.0069, 0.1346])

dq = normalize_dq(dq)
print("DQ original : ", np.round(dq, 3))

dq_tan = log_cayley_dual(dq)
print("DQ log      :", np.round(dq_tan, 3))


dual_quat_re = exp_cayley_dual(dq_tan)
print("DQ recovered: ", np.round(dual_quat_re, 3))

dual_quat_re = normalize_dq(dual_quat_re)

# Comparación de error
error = np.abs(dq - dual_quat_re)
print("Error de reconstrucción: ", np.round(error, 3))

a = np.array([ 0.1975646, 0.1212384, -0.1503095, 0.9610809 ])
a = normalize_quaternion(a)

log_a = log_cayley(a)
exp_log_a = exp_cayley(log_a)

print("quatern  : ",np.round(a,4))
print("log_a    : ", np.round(log_a,4))
print("exp_log_a: ", np.round(exp_log_a,4))