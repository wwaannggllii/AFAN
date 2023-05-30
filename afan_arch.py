import torch
import torch.nn as nn
import torch.nn.functional as F



class ACRB(nn.Module):
    def __init__(self, wn, channel, reduction = 4):
        super(ACRB, self).__init__()
        self.conv1_1 = nn.Sequential(wn(nn.Conv2d(channel,channel//reduction,kernel_size=(1,3),padding=(0,1),groups=1,bias=False)),
                                     wn(nn.Conv2d(channel//reduction,channel//reduction,kernel_size=(3,1),padding=(1,0),groups=1,bias=False)),
                                     wn(nn.Conv2d(channel//reduction,channel,kernel_size=(3,3),padding=(1,1),groups=1,bias=False)))
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1_1(x) #torch.Size([2, 64, 1, 1])
        y = self.ReLU(y)
        y = y + x
        return y

class DAAB(nn.Module):
    def __init__(self, wn, n_channels, reduction = 4):
        super(LAAB, self).__init__()
        self.GN = nn.GroupNorm(n_channels, n_channels)
        self.conv_du = nn.Sequential(

            wn(nn.Conv2d(1, n_channels // reduction, kernel_size=3, stride=1, padding=3 // 2, dilation=1, groups=1,
                         bias=False)),
            nn.ReLU(True),
            wn(nn.Conv2d(n_channels // reduction, 1, kernel_size=1, stride=1, padding=1 // 2, dilation=1, groups=1,
                         bias=False)),
            nn.Sigmoid()
        )
        self.act= nn.Sigmoid()
    def forward(self, x):
        x =self.GN(x)
        second_c = torch.mean(x, 1).unsqueeze(1)  # ([2, 1, 32, 32])  #平均池化
        second_c = self.conv_du(second_c)  # ([2, 1, 32, 32])
        second_c = second_c * x   # ([2, 64, 32, 32])

        xh = x.permute(0, 2, 1, 3)
        second_h = torch.mean(xh, 1).unsqueeze(1)  
        second_h = self.conv_du(second_h)  
        second_h = (second_h * xh).permute(0, 2, 1, 3)


        xw = x.permute(0, 3, 2, 1)
        second_w = torch.mean(xw, 1).unsqueeze(1)  
        second_w = self.conv_du(second_w)  
        second_w = (second_w * xw).permute(0, 3, 2, 1)

        y = second_c + second_h + second_w + x
        y = self.act(y) 
        return y



class SAAB(nn.Module):
    def __init__(self,  wn, num_features, act=nn.ReLU(True),upscale_factor = 1,reduction=1):
        super(SAAB, self).__init__()

        self.shared_MLP0 = nn.Sequential(
            nn.GroupNorm(num_features, num_features),
            wn(nn.Conv2d(num_features, num_features// reduction, kernel_size=3, padding=3 // 2,bias=False)),
            nn.ReLU(True),
            wn(nn.Conv2d(num_features // reduction, num_features, kernel_size=1, padding=1 // 2, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, x):
        avgout0 = self.shared_MLP0(x)
        return avgout0


class UPSample(nn.Module):
    def __init__(self,  upscale_factor, num_features,out_channels, reduction):
        super(UPSample, self).__init__()
        self.scale = upscale_factor
        self.upconv1 = nn.Conv2d(num_features, num_features//reduction, 3, 1, 1, bias=True)
        self.HRconv1 = nn.Conv2d(num_features//reduction, num_features//reduction, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_features//reduction, out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        if self.scale == 2 or self.scale == 3 or self.scale == 8:
            fea = self.upconv1(F.interpolate(x, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(x, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.HRconv2(fea))
        out = self.conv_last(fea)
        return out

class SCAM(nn.Module):
    def __init__(self, wn, channel):
        super(SCAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
        self.conv_du = nn.Sequential(
                wn(nn.Conv2d(channel, channel, kernel_size=3,stride=1, padding=3//2, dilation=1, bias=True)),
                nn.Sigmoid()
        )
    def forward(self, x):

        y = self.avg_pool(x)

        y = self.conv_du(y) 
        out = x * y + x
        return out.expand_as(x)

class AFAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_recurs, upscale_factor, norm_type=None,
                 act_type='prelu',reduction = 1):
        super(AFAN, self).__init__()

        act = nn.ReLU(True)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor([0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        head = []
        head.append(nn.Conv2d(in_channels, num_features, 3, padding=3 // 2))

        self.daab0 = DAAB(wn, num_features, reduction=1)
        self.daab1 = DAAB(wn, num_features, reduction=1)
        self.daab2 = DAAB(wn, num_features, reduction=1)
        self.daab3 = DAAB(wn, num_features, reduction=1)
        self.daab4 = DAAB(wn, num_features, reduction=1)
        self.daab5 = DAAB(wn, num_features, reduction=1)

        self.acrb0 = ACRB(wn, num_features, reduction=1)
        self.acrb1 = ACRB(wn, num_features, reduction=1)
        self.acrb2 = ACRB(wn, num_features, reduction=1)
        self.acrb3 = ACRB(wn, num_features, reduction=1)
        self.acrb4 = ACRB(wn, num_features, reduction=1)
        self.acrb5 = ACRB(wn, num_features, reduction=1)

        self.saab0 = SAAB(wn, num_features, act, upscale_factor, reduction=1)
        self.saab1 = SAAB(wn, num_features, act, upscale_factor, reduction=1)
        self.saab2 = SAAB(wn, num_features, act, upscale_factor, reduction=1)
        self.saab3 = SAAB(wn, num_features, act, upscale_factor, reduction=1)
        self.saab4 = SAAB(wn, num_features, act, upscale_factor, reduction=1)
        self.saab5 = SAAB(wn, num_features, act, upscale_factor, reduction=1)

        self.scam = SCAM(wn, num_features)

        self.head = nn.Sequential(*head)

        self.conv_1x1_output0 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output1 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output2 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output3 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output4 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output5 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))

        self.conv_1x1_output01 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output11 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output21 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output31 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output41 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))
        self.conv_1x1_output51 = wn(nn.Conv2d(num_features, num_features, 3, 1, 1, groups=1))


        self.conv_output0 = wn(nn.Conv2d(num_features*3, num_features, 1, 1, 0, groups=1))
        self.conv_output1 = wn(nn.Conv2d(num_features * 3, num_features, 1, 1, 0, groups=1))
        self.conv_output2 = wn(nn.Conv2d(num_features * 3, num_features, 1, 1, 0, groups=1))
        self.conv_output3 = wn(nn.Conv2d(num_features * 3, num_features, 1, 1, 0, groups=1))
        self.conv_output4 = wn(nn.Conv2d(num_features * 3, num_features, 1, 1, 0, groups=1))
        self.conv_output5 = wn(nn.Conv2d(num_features * 3, num_features, 1, 1, 0, groups=1))


        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.up = UPSample(upscale_factor,num_features,out_channels,reduction=2)

    def forward(self, x):
        x = (x - self.rgb_mean.cuda() * 255) / 127.5
  
        x = self.head(x)

        g0 = self.saab0(x)
        b0 = self.daab0(x)
        r0 = self.acrb0(x)
        gg0 = self.lrelu(self.conv_1x1_output0(torch.mul(g0, r0))) + x
        bb0 = self.lrelu(self.conv_1x1_output01(torch.mul(b0,r0))) + x

        out0 = torch.cat([gg0, bb0, r0], dim=1)
        out0 = self.conv_output0(out0)

        g1 = self.saab1(out0)
        b1 = self.daab1(out0)
        r1 = self.acrb1(out0)
        gg1 = self.lrelu(self.conv_1x1_output1(torch.mul(g1, r1))) + out0
        bb1 = self.lrelu(self.conv_1x1_output11(torch.mul(b1, r1))) + out0
        out1 = torch.cat([gg1, bb1, r1], dim=1)
        out1 = self.conv_output1(out1)

        g2 = self.saab2(out1)
        b2 = self.daab2(out1)
        r2 = self.acrb2(out1)
        gg2 = self.lrelu(self.conv_1x1_output2(torch.mul(g2, r2))) + out1
        bb2 = self.lrelu(self.conv_1x1_output21(torch.mul(b2, r2))) + out1
        out2 = torch.cat([gg2, bb2, r2], dim=1)
        out2 = self.conv_output2(out2)

        g3 = self.saab3(out2)
        b3 = self.daab3(out2)
        r3 = self.acrb3(out2)
        gg3 = self.lrelu(self.conv_1x1_output3(torch.mul(g3, r3))) + out2
        bb3 = self.lrelu(self.conv_1x1_output31(torch.mul(b3, r3))) + out2
        out3 = torch.cat([gg3, bb3, r3], dim=1)
        out3 = self.conv_output3(out3)

        g4 = self.saab4(out3)
        b4 = self.daab4(out3)
        r4 = self.acrb4(out3)
        gg4 = self.lrelu(self.conv_1x1_output4(torch.mul(g4, r4))) + out3
        bb4 = self.lrelu(self.conv_1x1_output41(torch.mul(b4, r4))) + out3
        out4 = torch.cat([gg4, bb4, r4], dim=1)
        out4 = self.conv_output4(out4)

        g5 = self.saab5(out4)
        b5 = self.daab5(out4)
        r5 = self.acrb5(out4)
        gg5 = self.lrelu(self.conv_1x1_output5(torch.mul(g5, r5))) + out4
        bb5 = self.lrelu(self.conv_1x1_output51(torch.mul(b5, r5))) + out4
        out5 = torch.cat([gg5, bb5, r5], dim=1)
        out5 = self.conv_output5(out5)


        out = self.scam(out5)

        out = out + x
  
        x = self.up(out)

        out = x * 127.5 + self.rgb_mean.cuda() * 255
        return out

