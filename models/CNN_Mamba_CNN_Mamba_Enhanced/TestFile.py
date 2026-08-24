import torch

x = torch.tensor([
    [
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    ]
])

print(x-x.mean(dim=[2,3],keepdim=True))