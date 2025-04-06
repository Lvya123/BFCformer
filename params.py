from ptflops import get_model_complexity_info
from models.model import SUNet
net = SUNet()
macs, params = get_model_complexity_info(net, (3,256,256), as_strings=True,print_per_layer_stat = True, verbose=True)
p = 0
for par in net.parameters():
    p += par.numel()
print(p)