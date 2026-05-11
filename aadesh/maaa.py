import torch
from tvm import relay

class BlackmanModel(torch.nn.Module):
    def __init__(self, n, periodic):
        super().__init__()
        self.n = n
        self.periodic = periodic

    def forward(self):
        return torch.blackman_window(self.n, periodic=self.periodic)


def main():
    n = 16
    periodic = False
    model = BlackmanModel(n, periodic).eval()
    scripted = torch.jit.trace(model, ())
    mod, params = relay.frontend.from_pytorch(scripted, [])

    print("===== Relay IR =====")
    print(mod.astext(show_meta_data=False))


if __name__ == "__main__":
    main()
    