
import torch
import torch.nn as nn
import math

model = nn.Conv2d(3, 3, 3, 1, 1)

optim = torch.optim.AdamW(model.parameters(), lr=2e-4)

sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=540000, eta_min=1e-6)

for i in range(1, 540000 + 1):
    optim.step()
    sch.step()
    if i % 50 == 0 or i == 540000+1:
    # if i % 1052 == 0:
        print(i, optim.param_groups[0]['lr'])
    # optim.step()
    # sch.step()

# for i in range(1, int(math.sqrt(1052*1060))):
#     if (1052*1060)%i == 0:
#         print(i)