"""CSPNeXt-lite: a compact conv backbone (pure PyTorch, MPS-friendly ops only).

Five stages, each halving spatial resolution, channels scaled by `width`.
Stem stride 2 + 4 stages stride 2 -> /32 total. Output is the last feature map.
"""
import torch.nn as nn


def _ch(base, width):
    return max(8, int(round(base * width / 8) * 8))


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = ConvBNAct(c, c, 3)
        self.c2 = ConvBNAct(c, c, 3)

    def forward(self, x):
        return x + self.c2(self.c1(x))


class CSPNeXtLite(nn.Module):
    def __init__(self, width=0.33, depth=0.33):
        super().__init__()
        base = [64, 128, 256, 512, 1024]
        nblocks = max(1, int(round(3 * depth)))
        chs = [_ch(b, width) for b in base]
        self.stem = ConvBNAct(3, chs[0], 3, s=2)
        stages = []
        cin = chs[0]
        for cout in chs[1:]:
            layers = [ConvBNAct(cin, cout, 3, s=2)]
            layers += [Bottleneck(cout) for _ in range(nblocks)]
            stages.append(nn.Sequential(*layers))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.out_channels = chs[-1]

    def forward(self, x):
        x = self.stem(x)
        for st in self.stages:
            x = st(x)
        return x  # (B, out_channels, H/32, W/32)
