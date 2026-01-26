import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

class MultiBranchDeformConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_sizes=[3], stride=1, dilation=1, bias=True, offset_groups=1, with_mask=False):
        super(MultiBranchDeformConv2d, self).__init__()
        assert len(kernel_sizes) <= 4, "Only up to 4 branches supported"

        self.kernel_sizes = kernel_sizes
        self.with_mask = with_mask
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.dilation = dilation
        self.bias_enabled = bias
        self.offset_groups = offset_groups

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.paddings = []
        self.param_generators = nn.ModuleList()

        self.branch_out_dim = out_dim // len(kernel_sizes)
        for k in kernel_sizes:
            padding = k // 2
            self.paddings.append(padding)

            weight = nn.Parameter(torch.empty(self.branch_out_dim, in_dim, k, k))
            self.weights.append(weight)

            if bias:
                b = nn.Parameter(torch.empty(self.branch_out_dim))
                self.biases.append(b)
            else:
                self.biases.append(None)

            num_params = 3 if with_mask else 2
            param_generator = nn.Conv2d(in_dim, num_params * offset_groups * k * k, kernel_size=3, padding=1, stride=1)
            self.param_generators.append(param_generator)

        self.merge_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=0)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
        for bias in self.biases:
            if bias is not None:
                nn.init.constant_(bias, 0)
        for pg in self.param_generators:
            nn.init.constant_(pg.weight, 0)
            nn.init.constant_(pg.bias, 0)
        nn.init.kaiming_uniform_(self.merge_conv.weight, a=5 ** 0.5)
        nn.init.constant_(self.merge_conv.bias, 0)

    def forward(self, x):
        outputs = []
        for i, k in enumerate(self.kernel_sizes):
            weight = self.weights[i]
            bias = self.biases[i]
            padding = self.paddings[i]
            param_gen = self.param_generators[i]

            offset_mask = param_gen(x)
            if self.with_mask:
                o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
                offset = torch.cat([o1, o2], dim=1)
                mask = mask.sigmoid()
            else:
                offset = offset_mask
                mask = None

            out = deform_conv2d(
                x,
                offset=offset,
                weight=weight,
                bias=bias,
                stride=(self.stride, self.stride),
                padding=(padding, padding),
                dilation=(self.dilation, self.dilation),
                mask=mask
            )
            outputs.append(out)

        merged = torch.cat(outputs, dim=1)
        return self.merge_conv(merged)

if __name__ == "__main__":
    model = MultiBranchDeformConv2d(
        in_dim=3,
        out_dim=12,
        kernel_sizes=[3, 5, 7],
        stride=1,
        dilation=1,
        bias=True,
        offset_groups=1,
        with_mask=True
    )
    print(model)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
