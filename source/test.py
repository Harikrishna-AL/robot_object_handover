import torch

# Tensor creation
a = torch.tensor([0, 12, 4, 8, 
                  1, 13, 5, 9, 
                  2, 14, 6, 10, 
                  3, 15, 7, 11])+7
a = a.repeat(3, 1).double()
# a[0,0] = 8
# print(a[0,0])

# Operating
b = a.view(-1, 4,4)# .transpose(1,2)
res = torch.mean(b, 2, False)

c = torch.tensor([1,1,1])
d = torch.tensor([2,3,4])
# print(c*d)

a[[False, True, True]] = torch.zeros(16).double()
print(a)
