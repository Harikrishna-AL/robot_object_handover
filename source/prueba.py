import torch

actions_quat = torch.zeros((3, 2,(7+16)))

actions = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.1000, -0.2000, -0.3000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1000, 0.2000, 0.3000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.1000, -1.2000, -1.3000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.1000, 1.2000, 1.3000],
                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -2.1000, -2.2000, -2.3000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.1000, 2.2000, 2.3000]])

val = actions[:, 6:9]
val = torch.cat((val, actions[:, 6+9:9+9]), 0)
print(val)


# Índices para intercalar filas
indices = torch.arange(val.size(0)).view(2, -1).T.flatten()
print(torch.arange(val.size(0)))
print(torch.arange(val.size(0)).view(2, -1))
print(torch.arange(val.size(0)).view(2, -1).T)
print(torch.arange(val.size(0)).view(2, -1).T.flatten())


# Tensor con filas intercaladas
tensor_intercalado = val[indices]
print(tensor_intercalado)

actions_quat[:, :, 15:] = tensor_intercalado.view(3, 2, 3)[:, :, 1:].repeat_interleave(4, dim = -1)
print(actions_quat)

