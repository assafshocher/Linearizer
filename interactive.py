import pickle
from lin_diff import LinearDiffusion
import torch

load_dir = './results/mnist_20250224_095459/ckpts/'
with open(load_dir + 'conf.pkl', 'rb') as f:
    conf = pickle.load(f)
model = LinearDiffusion(conf).to(conf.device)
model.load_checkpoint("e100.pth")
model.eval()

# Iterative sampling code
b_sz = 1
T = conf.T
A = model.A()[0]  # .bool()
x = torch.randn(b_sz, *conf.im_shape, device=model.a.device)
a = model.a_forward
b = model.b_forward
sigma = model.Sigma
print(f"Sampling iteratively with T={T}")
g_x = model.g(x)
g_epses = torch.randn(T, b_sz, *conf.im_shape, device=model.a.device)
gmin = []
# tmp = []
# tmp2 = []
for t, g_eps in zip(range(T - 2, 0, -1), g_epses):
    gmin.append(g_x.min().item())
    print("g_min: ", g_x.min().item(), "g_max: ", g_x.max().item(), "g_abs_min: ", g_x.abs().min().item())
    print("at: ", a[t].item(), "bt: ", b[t].item(), "sigma t: ", sigma[t].item())
    print("g_eps_min: ", g_eps.min().item(), "g_eps_max: ", g_eps.max().item(), "g_eps_abs_min: ", g_eps.abs().min())
    print("Unclear calculation: ", (a[t] + b[t] * A))
    # tmp.append(self.g.inverse(g_x.view_as(x))[0, :].norm().item())
    # print("New calculation: ", self.g.inverse(g_x.view_as(x))[0, :].norm().item())
    # tmp2.append(g_x[0, :].norm().item())
    # import pdb; pdb.set_trace()
    g_x = (a[t] + b[t] * A) * g_x.view(g_x.shape[0], -1) + sigma[t] * model.g(g_eps).view(g_eps.shape[0], -1)
# if self.conf.wandb:
#     plt.plot(range(T-2, -1, -1), [a[t].item() for t in range(T-2, -1, -1)], range(T-2, -1, -1), [b[t].item() for t in range(T-2, -1, -1)])
#     plt.legend(["at", "bt"])
#     wandb.log({"at_bt": wandb.Image(plt)})
#     plt.plot(range(T-2, -1, -1), gmin)
#     wandb.log({"gmin": wandb.Image(plt)})
x_0 = model.g.inverse(g_x.view_as(x))