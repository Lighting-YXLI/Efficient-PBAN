import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from GMDC import MultiBranchDeformConv2d
from SubEC import SE_Block


class myBlock(nn.Module):
    def __init__(self, channel, upchannel=False):
        super(myBlock, self).__init__()
        self.q1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.q2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.k1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.br1 = MultiBranchDeformConv2d(in_dim=channel, out_dim=channel, kernel_sizes=[3,5,7,9])
        self.k2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.br2 = MultiBranchDeformConv2d(in_dim=channel, out_dim=channel, kernel_sizes=[3,5,7,9])
        self.v1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.v2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.SE = SE_Block(inchannel=channel)
        self.upchannel = upchannel
        self.after_block1 = nn.Sequential(
            nn.Conv2d(channel, 2 * channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(2 * channel),
            nn.ReLU(inplace=True)
        )
        self.after_block2 = nn.Sequential(
            nn.Conv2d(channel, 2 * channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(2 * channel),
            nn.ReLU(inplace=True)
        )
        self.after_block3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.after_block4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=-1)  # 用dim=-1，适配1D注意力

    def forward(self, x1, x2):
        b, c, h, w = x1.size()

        q1 = self.q1(x1)
        q2 = self.q2(x2)
        k1 = self.k1(x1)
        q1 = self.br1(q1)
        k2 = self.k2(x2)
        q2 = self.br2(q2)
        v1 = self.v1(x1)
        v2 = self.v2(x2)

        # 解耦空间注意力计算：先计算H方向注意力，再计算W方向注意力

        # --- H方向注意力 ---
        # reshape为 [B, C, H, W] -> [B, C, H, W]
        # 对每一列计算注意力，令W为序列长度
        q1_h = q1.permute(0, 2, 1, 3).contiguous().view(b*h, c, w)   # (B*H, C, W)
        k2_h = k2.permute(0, 2, 1, 3).contiguous().view(b*h, c, w)   # (B*H, C, W)
        v1_h = v1.permute(0, 2, 1, 3).contiguous().view(b*h, c, w)   # (B*H, C, W)

        att_h = torch.bmm(q1_h.transpose(1,2), k2_h)  # (B*H, W, W)
        att_h = att_h / (att_h.std() + 1e-6)
        att_h = self.softmax(att_h)                    # (B*H, W, W)

        out_h = torch.bmm(v1_h, att_h)                 # (B*H, C, W)
        out_h = out_h.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()  # (B, C, H, W)

        # --- W方向注意力 ---
        # 对每一行计算注意力，令H为序列长度
        q1_w = q1.permute(0, 3, 1, 2).contiguous().view(b*w, c, h)   # (B*W, C, H)
        k2_w = k2.permute(0, 3, 1, 2).contiguous().view(b*w, c, h)   # (B*W, C, H)
        v1_w = v1.permute(0, 3, 1, 2).contiguous().view(b*w, c, h)   # (B*W, C, H)

        att_w = torch.bmm(q1_w.transpose(1,2), k2_w)  # (B*W, H, H)
        att_w = att_w / (att_w.std() + 1e-6)
        att_w = self.softmax(att_w)                    # (B*W, H, H)

        out_w = torch.bmm(v1_w, att_w)                 # (B*W, C, H)
        out_w = out_w.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()  # (B, C, H, W)

        # 融合H和W方向结果
        out1 = (out_h + out_w) * 0.5

        # 同理计算另一边的注意力（q2, k1, v2）
        # --- H方向 ---
        q2_h = q2.permute(0, 2, 1, 3).contiguous().view(b*h, c, w)
        k1_h = k1.permute(0, 2, 1, 3).contiguous().view(b*h, c, w)
        v2_h = v2.permute(0, 2, 1, 3).contiguous().view(b*h, c, w)

        att2_h = torch.bmm(q2_h.transpose(1,2), k1_h)
        att2_h = att2_h / (att2_h.std() + 1e-6)
        att2_h = self.softmax(att2_h)
        out2_h = torch.bmm(v2_h, att2_h)
        out2_h = out2_h.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()

        # --- W方向 ---
        q2_w = q2.permute(0, 3, 1, 2).contiguous().view(b*w, c, h)
        k1_w = k1.permute(0, 3, 1, 2).contiguous().view(b*w, c, h)
        v2_w = v2.permute(0, 3, 1, 2).contiguous().view(b*w, c, h)

        att2_w = torch.bmm(q2_w.transpose(1,2), k1_w)
        att2_w = att2_w / (att2_w.std() + 1e-6)
        att2_w = self.softmax(att2_w)
        out2_w = torch.bmm(v2_w, att2_w)
        out2_w = out2_w.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()

        out2 = (out2_h + out2_w) * 0.5

        out1 = self.SE(out1) + x1
        out2 = self.SE(out2) + x2

        if self.upchannel:
            out1 = self.after_block1(out1)
            out2 = self.after_block2(out2)
        else:
            out1 = self.after_block3(out1)
            out2 = self.after_block4(out2)
        return out1, out2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet18 = models.resnet34(pretrained=True)
        for param in resnet18.conv1.parameters():
            param.requires_grad = False
        for param in resnet18.bn1.parameters():
            param.requires_grad = False
        for param in resnet18.layer1.parameters():
            param.requires_grad = False
        # 不冻结 layer2（参与训练）
        for param in resnet18.layer2.parameters():
            param.requires_grad = True
        for param in resnet18.layer3.parameters():
            param.requires_grad = True
        for param in resnet18.layer4.parameters():
            param.requires_grad = True

        # 两个stem仍是layer1输出，64通道，160x160
        self.stem = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2
        )

        # block1, block2输入64通道，不放大通道
        #self.block1 = myBlock(channel=64, upchannel=False)
        self.block2 = myBlock(channel=256, upchannel=True)

        # 接着加resnet的layer2，通道变128，空间减半160->80
        self.layer2_1 = resnet18.layer3
        self.layer2_2 = resnet18.layer3

        # 这里对应输入变为128通道
        self.pool1 = nn.AdaptiveAvgPool2d((2,2))
        self.pool2 = nn.AdaptiveAvgPool2d((2,2))

        self.fc11 = nn.Linear(512*2*2, 512)
        self.fc12 = nn.Linear(512*2*2, 512)
        self.fc21 = nn.Linear(512, 128)
        self.fc22 = nn.Linear(512, 128)
        self.drop11 = nn.Dropout(0.3)
        self.drop12 = nn.Dropout(0.3)
        self.drop21 = nn.Dropout(0.2)
        self.drop22 = nn.Dropout(0.2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, x1, x2):
        x1 = self.stem(x1)  # [B,64,160,160]
        x2 = self.stem(x2)

        #x1, x2 = self.block1(x1, x2)  # 64通道，160x160

        x1 = self.layer2_1(x1)   # 128通道，80x80
        x2 = self.layer2_2(x2)

        x1, x2 = self.block2(x1, x2)

        x1 = self.pool1(x1)      # 128x4x4
        x2 = self.pool2(x2)

        x1 = self.drop11(F.relu(self.fc11(self.flat(x1))))
        x2 = self.drop21(F.relu(self.fc12(self.flat(x2))))
        x1 = self.drop21(F.relu(self.fc21(x1)))
        x2 = self.drop22(F.relu(self.fc22(x2)))

        x = torch.cat((x1, x2), dim=1)
        x = torch.squeeze(self.out(self.fc(x)), dim=1)

        return x


if __name__ == "__main__":
    x1 = torch.randn(2, 3, 1280, 1280)
    x2 = torch.randn(2, 3, 1280, 1280)
    model = Net()
    out = model(x1, x2)
    print(out.size())  # 预期输出 [2]
