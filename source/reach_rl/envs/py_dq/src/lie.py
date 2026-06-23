import torch
from .dq import *



def exp_parallel_transport(exp_fc, mul_operator, x: torch.Tensor, s: torch.Tensor):
    return mul_operator(x, exp_fc(s))


def log_parallel_transport(log_fc, diff_operator, x: torch.Tensor, q: torch.Tensor):

    # print("X: ", dq_translation(x))
    # print("Q: ", dq_translation(q))
    # print("DIFF: ", dq_translation(diff_operator(q, x)))
    # print("DIFF: ", dq_translation(diff_operator(x, q)))
    # print("\n\n\n")
    # raise

    return log_fc(diff_operator(x, q))





def convert_dq_to_Lab(x: torch.Tensor):
    assert x.shape[-1] == 8

    t = dq_translation(dq = x)
    r = x[:, :4]

    return torch.cat((t, r), dim = -1)



def identity_map(x: torch.Tensor):
    return x




def exp_bruno(dq_: torch.Tensor):

    assert dq_.shape[-1] == 6

    device = dq_.device

    dq_ = torch.cat((torch.zeros((dq_.shape[0], 1), device = device), dq_[:, :3], 
                     torch.zeros((dq_.shape[0], 1), device = device), dq_[:, 3:]), dim = -1)

    phi = torch.norm(dq_[:, :4], dim = -1)
    phi_idx = phi != 0

    prim = torch.cat((torch.ones(dq_.shape[0], 1), torch.zeros(dq_.shape[0], 3)), dim = -1).to(device)
    prim[phi_idx] = (torch.sin(phi[phi_idx]) / phi[phi_idx]).unsqueeze(-1) * dq_[phi_idx, :4]
    prim[phi_idx, 0] += torch.cos(phi[phi_idx])

    res = torch.cat((prim, q_mul(dq_[:, 4:], prim)), dim = -1)

    
    return res

def log_bruno(dq: torch.Tensor, debug = False):
    assert dq.shape[-1] == 8
    # assert torch.any(dq_is_norm(dq))

    primary = (q_angle(q = dq[:, :4]).unsqueeze(-1)*0.5)*q_axis(q = dq[:, :4])
    dual = dq_translation(dq = dq)*0.5

    if debug:
        print("P: ", primary)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("ANGLE: ", q_angle(q = dq[:, :4]), debug = True)
        print("AXIS: ", q_axis(q = dq[:, :4]), debug = True)

    return torch.cat((primary, dual), dim = -1)




def rotation_matrix_from_quaternion(q: torch.Tensor):
    assert q.shape[-1] == 4
    assert torch.any(q_is_norm(q = q))

    # First row of the rotation matrix
    r00 = 2 * (q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1]) - 1
    r01 = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    r02 = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])

    r1 = torch.cat((r00.unsqueeze(-1), r01.unsqueeze(-1), r02.unsqueeze(-1)), dim = -1)
     
    # Second row of the rotation matrix
    r10 = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    r11 = 2 * (q[:, 0] * q[:, 0] + q[:, 2] * q[:, 2]) - 1
    r12 = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    
    r2 = torch.cat((r10.unsqueeze(-1), r11.unsqueeze(-1), r12.unsqueeze(-1)), dim = -1)

    # Third row of the rotation matrix
    r20 = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    r21 = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    r22 = 2 * (q[:, 0] * q[:, 0] + q[:, 3] * q[:, 3]) - 1

    # First row of the rotation matrix
    r00 = 1 - 2*q[:, 2]**2 - 2*q[:, 3]**2
    r01 = 2*q[:, 1]*q[:, 2] - 2*q[:, 0]*q[:, 3]
    r02 = 2*q[:, 1]*q[:, 3] + 2*q[:, 0]*q[:, 2]

    r1 = torch.cat((r00.unsqueeze(-1), r01.unsqueeze(-1), r02.unsqueeze(-1)), dim = -1)
     
    # Second row of the rotation matrix
    r10 = 2*q[:, 1]*q[:, 2] + 2*q[:, 0]*q[:, 3]
    r11 = 1 - 2*q[:, 1]**2 - 2*q[:, 3]**2
    r12 = 2*q[:, 2]*q[:, 3] - 2*q[:, 0]*q[:, 1]
    
    r2 = torch.cat((r10.unsqueeze(-1), r11.unsqueeze(-1), r12.unsqueeze(-1)), dim = -1)

    # Third row of the rotation matrix
    r20 = 2*q[:, 1]*q[:, 3] - 2*q[:, 0]*q[:, 2]
    r21 = 2*q[:, 2]*q[:, 3] + 2*q[:, 0]*q[:, 1]
    r22 = 1 - 2*q[:, 1]**2 - 2*q[:, 2]**2

    r3 = torch.cat((r20.unsqueeze(-1), r21.unsqueeze(-1), r22.unsqueeze(-1)), dim = -1)

    return torch.cat((r1.unsqueeze(1), r2.unsqueeze(1), r3.unsqueeze(1)), dim = 1) 

def get_inv_V_matrix(w: torch.Tensor, theta: torch.Tensor):
    assert w.shape[-1] == 3
    assert theta.shape[-1] == 1

    return -0.5*w + (1 - (theta*torch.cos(theta/2) / (2*torch.sin(theta/2)))) / theta**2 * w**2

def get_V_matrix(w: torch.Tensor, theta: torch.Tensor):
    assert w.shape[-1] == 3
    assert theta.shape[-1] == 1

    return (1 - torch.cos(theta)) / (theta+1)**2 * w + (theta - torch.sin(theta)) / (theta+1)**3 * w**2

def exp_stereo_t(t: torch.Tensor, V: torch.Tensor):
    assert t.shape[-1] == 3
    assert V.shape[-1] == 3

    return V * t

def log_stereo_t(t: torch.Tensor, r:torch.Tensor):
    assert t.shape[-1] == 3
    assert r.shape[-1] == 4


    R = rotation_matrix_from_quaternion(q = r) 
    print("R: ", R)

    a = torch.cat(((R[:, 2,1] - R[:, 1, 2]), 
                   (R[:, 0,2] - R[:, 2, 0]), 
                   (R[:, 1,0] - R[:, 0, 1])), dim = 0).view(-1, t.shape[0]).transpose(0,1)


    theta = 2* torch.acos(r[:, 0]) # q_angle(q = r)
    print("THETA: ", theta)
    w = theta / (1+2 * torch.sin(theta)) * a
    print("W: ", w)
    print((1+2 * torch.sin(theta)))
    print(a)
    V = get_V_matrix(w = w, theta = theta)

    V_ = 1 / V # get_inv_V_matrix(w = w, theta = theta)
    print("V: ", V)
    print("V_: ", V_)

    return V_ * t, V

def exp_stereo_q(q_: torch.Tensor):
    assert q_.shape[-1] == 3
    div = (1 + torch.norm(q_, dim = -1)).unsqueeze(-1)

    re = (1 - torch.pow(torch.norm(q_, dim = -1), 2)).unsqueeze(-1) / div
    im = 2*q_ / div

    return torch.cat((re, im), dim = -1)

def log_stereo_q(q: torch.Tensor):
    assert q.shape[-1] == 4

    return q[:, 1:] / (1+q[:, 0]).unsqueeze(-1)

def exp_stereo(dq_:torch.Tensor):
    assert dq_.shape[-1] == 6
    # assert V.shape[-1] == 3

    q_p = exp_stereo_q(q_ = dq_[:, :3])    

    return dq_from_tr(t = dq_[:, 3:], r = q_p)

def log_stereo(dq: torch.Tensor):
    assert dq.shape[-1] == 8
    assert torch.any(dq_is_norm(dq))

    # t_, V = log_stereo_t(t = dq_translation(dq = dq), r = dq[:, :4])
    t_ = dq_translation(dq = dq)

    return torch.cat((log_stereo_q(q = dq[:, :4]), t_), dim = -1)



def log_cayley(dq: torch.Tensor):
    assert dq.shape[-1] == 8
    assert torch.any(dq_is_norm(dq = dq))

    identity = torch.cat((torch.ones(dq.shape[0], 1), torch.zeros(dq.shape[0], 3)), dim = -1)
    denom = q_norm(dq[:, :4] + identity).unsqueeze(-1)

    res = q_mul(q1 = dq[:, :4] - identity, q2 = q_conjugate(dq[:, :4] + identity) / (denom*denom) )

    return torch.cat((res[:, 1:4], dq_translation(dq = dq)), dim = -1)

def exp_cayley(dq_: torch.Tensor):
    assert dq_.shape[-1] == 6
    
    t = dq_[:, 3:]
    dq_ = torch.cat((torch.zeros(dq_.shape[0], 1), dq_[:, :3]), dim = -1)
    
    identity = torch.cat((torch.ones(dq_.shape[0], 1), torch.zeros(dq_.shape[0], 3)), dim = -1)
    denom = q_norm(identity - dq_[:, :4]).unsqueeze(-1)

    res = q_mul(q1 = (identity + dq_), q2 = q_conjugate(identity - dq_) / (denom*denom) )
    
    return dq_from_tr(t = t, r = res)




t1 = torch.tensor([[-4.9190e-01,  1.3330e-01,  4.8790e-01]])
r1 = torch.tensor([[3.3840e-06, -3.8268e-01, -9.2388e-01,  2.1003e-06]])

t2 = torch.tensor([[-0.1,  0.1348,  0.3480]])
r2 = torch.tensor([[0.2521, 0.0346, 0.9422,  -0.2179]])*1

# t2 = torch.tensor([[0.0, -0.0,  0.01,  0.0]]).repeat(2,1)
# r2 = torch.tensor([[0.4999, 0.0174, 0.0,  -0.0]]).repeat(2,1)


dq1 = dq_from_tr(t = t1, r = r1)
dq2 = dq_from_tr(t = t2, r = r1)

print("DQ1 translation: ", dq_translation(dq = dq1))
print("DQ1 translation: ", dq_translation(dq = dq2))

x = dq1
q = dq2
print("DQ: ", q)


# x_q = dq_mul(dq1 = dq_conjugate(x), dq2 = q)
# x_q = dq_normalize(x_q)

# l_ = log_stereo(dq = x_q)
l_ = log_parallel_transport(log_fc = log_bruno,
                            diff_operator = dq_diff,
                            x = x,
                            q = q)



# b = exp_stereo(dq_ = l_)

# b = dq_mul(x, b)
b = exp_parallel_transport(exp_fc = exp_bruno,
                           mul_operator = dq_mul,
                           x = x,
                           s = l_)
b = dq_normalize(b)

print("Logaritmico: ", l_.round(decimals = 6))
print("Exponencial: ", b.round(decimals = 6))

