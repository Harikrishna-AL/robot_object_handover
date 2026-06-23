import torch
from .lie import *

# === INTERPOLATION METHODS =======================================================================

def ScLERP(dq1: torch.Tensor, dq2: torch.Tensor, step: int = 0.1):

    assert dq1.shape[-1] == 8
    assert dq2.shape[-1] == 8
    assert torch.any(torch.logical_and(dq_is_norm(dq1), dq_is_norm(dq1)))
    
    device = dq1.device

    tau = torch.arange(0, 1 + step, step).to(device)
    dq_diff = dq_mul(dq1 = dq_conjugate(dq1), dq2 = dq2)

    log_diff = log_bruno(dq = dq_diff)

    dq_diff_tau = exp_bruno(dq_ = (tau*log_diff.unsqueeze(-1)).transpose(1,2).reshape(-1, 6))

    dq_tau = dq_mul(dq1 = dq1.repeat_interleave(tau.shape[0], dim = 0), dq2 = dq_diff_tau)

    return dq_normalize(dq = dq_tau).view(dq1.shape[0], -1, 8)



# t1 = torch.tensor([[0.0,     0.2576, -0.0686, 0.8289]])
# r1 = torch.tensor([[-0.6922, 0.2475, -0.6111, 0.2936]])

# t2 = torch.tensor([[0.0,     0.4397,  0.3643, 0.9340]])
# r2 = torch.tensor([[-0.5824, 0.4790, -0.4755, 0.4531]])

# dq1 = dq_from_tr(t = t1, r = r1)
# dq2 = dq_from_tr(t = t2, r = r2)

# # print(ScLERP(dq1, dq2))
# # print("------")
# # print(dq1)
# # print(dq2)

# print("-------")
# print("Traslacion original: ", dq_translation(dq2))
# print("Traslacion interpolada final: ", dq_translation(ScLERP(dq1, dq2)[:, -1]))
# print("ERROR en traslacion: ", dq_translation(dq2) - dq_translation(ScLERP(dq1, dq2)[:, -1]))