import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.channel_1 = 24
        self.channel_2 = 32
        self.channel_4 = 48

        self.convLayer_1 = nn.Sequential(
            nn.Conv2d(3, self.channel_1, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.convLayer_2 = nn.Sequential(
            nn.Conv2d(3, self.channel_2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.convLayer_4 = nn.Sequential(
            nn.Conv2d(3, self.channel_4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img):
        b, c, h, w = img.shape[0], img.shape[1], img.shape[2], img.shape[3]

        img1 = img
        img2 = F.interpolate(img, size=(h // 2, w // 2), mode='bilinear')
        img4 = F.interpolate(img, size=(h // 4, w // 4), mode='bilinear')

        fea1 = self.convLayer_1(img1)
        fea2 = self.convLayer_2(img2)
        fea4 = self.convLayer_4(img4)

        return fea1, fea2, fea4

class Tail(nn.Module):
    def __init__(self):
        super(Tail, self).__init__()

        self.channel_1 = 24
        self.channel_2 = 32
        self.channel_4 = 48

        self.catLayer = nn.Sequential(
            nn.Conv2d((self.channel_1 + self.channel_2 + self.channel_4), 3, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, fea1, fea2, fea4):
        b, c, h, w = fea1.shape[0], fea1.shape[1], fea1.shape[2], fea1.shape[3]
        fea2_up = F.interpolate(fea2, size=(h, w), mode='bilinear')
        fea4_up = F.interpolate(fea4, size=(h, w), mode='bilinear')
        merge = torch.cat((fea1, fea2_up, fea4_up), 1)
        fake_S = self.catLayer(merge)

        return fake_S

# TODO: IMDB, Inner_Scale_Connection_Block
# TODO: ISCB
class ISCB(nn.Module):
    def __init__(self, channel):
        super(ISCB, self).__init__()
        self.channel = channel
        self.scale_num = 4
        self.conv_num = 4

        self.scale1 = nn.ModuleList()
        self.scale2 = nn.ModuleList()
        self.scale4 = nn.ModuleList()
        self.scale8 = nn.ModuleList()

        for i in range(self.conv_num):
            self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.scale2.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.scale4.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.scale8.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))

        self.fusion84 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.fusion42 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))

        self.pooling8 = nn.MaxPool2d(8, 8)
        self.pooling4 = nn.MaxPool2d(4, 4)
        self.pooling2 = nn.MaxPool2d(2, 2)

        self.fusion_all = nn.Sequential(nn.Conv2d(4 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        feature8 = self.pooling8(x)
        b8, c8, h8, w8 = feature8.size()

        feature4 = self.pooling4(x)
        b4, c4, h4, w4 = feature4.size()

        feature2 = self.pooling2(x)
        b2, c2, h2, w2 = feature2.size()

        feature1 = x
        b1, c1, h1, w1 = feature1.size()

        for i in range(self.conv_num):
            feature8 = self.scale8[i](feature8)
        scale8 = feature8
        feature4 = self.fusion84(torch.cat([feature4, F.upsample(scale8, [h4, w4])], dim=1))

        for i in range(self.conv_num):
            feature4 = self.scale4[i](feature4)
        scale4 = feature4
        feature2 = self.fusion42(torch.cat([feature2, F.upsample(scale4, [h2, w2])], dim=1))

        for i in range(self.conv_num):
            feature2 = self.scale2[i](feature2)

        scale2 = feature2
        feature1 = self.fusion21(torch.cat([feature1, F.upsample(scale2, [h1, w1])], dim=1))

        for i in range(self.conv_num):
            feature1 = self.scale1[i](feature1)
        scale1 = feature1
        fusion_all = self.fusion_all(torch.cat([scale1, F.upsample(scale2, [h1, w1]), F.upsample(scale4, [h1, w1]), F.upsample(scale8, [h1, w1])], dim=1))

        return fusion_all + x

# TODO: MSFB
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class MSFB(nn.Module):
    def __init__(self, channel, conv=default_conv):
        super(MSFB, self).__init__()
        self.channel = channel
        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(self.channel, self.channel, kernel_size_1)
        self.conv_3_2 = conv(self.channel * 2, self.channel * 2, kernel_size_1)
        self.conv_5_1 = conv(self.channel, self.channel, kernel_size_2)
        self.conv_5_2 = conv(self.channel * 2, self.channel * 2, kernel_size_2)
        self.confusion = nn.Conv2d(self.channel * 4, self.channel, 1, padding=0, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x

        return output

class UnitBlock(nn.Module):
    def __init__(self, channel):
        super(UnitBlock, self).__init__()
        self.channel = channel

        self.unit = nn.Sequential(
            ISCB(self.channel),
            MSFB(self.channel),
        )

    def forward(self, fea):
        output = self.unit(fea)
        return output

# TODO: DenseConnection
class DenseConnection(nn.Module):
    def __init__(self, channel, unit, unit_num):
        super(DenseConnection, self).__init__()

        self.unit_num = unit_num
        self.channel = channel
        self.units = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()

        for i in range(self.unit_num):
            self.units.append(unit)
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2, inplace=True)))

    def forward(self, x, catList=None):
        cat = []
        cat.append(x)
        out = x

        if catList == None:
            for i in range(self.unit_num):
                tmp = self.units[i](out)
                cat.append(tmp)
                out = self.conv1x1[i](torch.cat(cat, dim=1))
        else:
            for i in range(self.unit_num):
                tmp = self.units[i](torch.add(out, catList[i + 1]))
                cat.append(tmp)
                out = self.conv1x1[i](torch.cat(cat, dim=1))

        return cat, out

# TODO: ResBlock
class ResBlock(nn.Module):
    def __init__(self, channel, ksize, stride, padding):
        super(ResBlock, self).__init__()

        self.channel = channel
        self.ksize = ksize
        self.stride = stride
        self.padding = padding

        self.convLayer = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, self.ksize, self.stride, self.padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel, self.channel, 3, 1, 1),
        )

        self.active = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, fea):
        residual = fea
        output = self.convLayer(fea)
        output = torch.add(output, residual)
        final = self.active(output)

        return final

class Encoder_Decoder_1(nn.Module):
    def __init__(self):
        super(Encoder_Decoder_1, self).__init__()

        self.channel_1 = 24
        self.channel_2 = 32
        self.channel_4 = 48

        self.unit_num_1 = 10
        self.unit_num_2 = 4
        self.unit_num_4 = 2

        self.enLayer_1 = DenseConnection(self.channel_1, UnitBlock(self.channel_1), self.unit_num_1)
        self.enLayer_2 = DenseConnection(self.channel_2, UnitBlock(self.channel_2), self.unit_num_2)
        self.enLayer_4 = DenseConnection(self.channel_4, UnitBlock(self.channel_4), self.unit_num_4)
        self.deLayer_2 = DenseConnection(self.channel_2, UnitBlock(self.channel_2), self.unit_num_2)
        self.deLayer_1 = DenseConnection(self.channel_1, UnitBlock(self.channel_1), self.unit_num_1)

        # TODO: catConv
        self.catLayer_1 = nn.Sequential(
            nn.Conv2d(self.channel_1 * self.unit_num_1, self.channel_2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.catLayer_2 = nn.Sequential(
            nn.Conv2d(self.channel_2 * self.unit_num_2, self.channel_4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.catLayer_3 = nn.Sequential(
            nn.Conv2d(self.channel_4 * self.unit_num_4, self.channel_2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.catLayer_4 = nn.Sequential(
            nn.Conv2d(self.channel_2 * self.unit_num_2, self.channel_1, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # TODO: active
        self.active = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, fea1, en2_cat2, de2_cat2, en2_cat4):
        b, c, h, w = fea1.shape[0], fea1.shape[1], fea1.shape[2], fea1.shape[3]

        # TODO: encoder1
        en1_input = fea1 # [6, 24, 128, 128]
        en1_cat, en1_output = self.enLayer_1(en1_input)
        en1_pool_list = []

        for i in range(self.unit_num_1):
            fea_pool = F.interpolate(en1_cat[i + 1], size=(h // 2, w // 2), mode='bilinear')
            en1_pool_list.append(fea_pool)

        en1_merge = torch.cat(en1_pool_list, 1)

        # TODO: encoder2
        en2_input = self.catLayer_1(en1_merge)
        en2_cat, en2_output = self.enLayer_2(en2_input, en2_cat2)
        en2_pool_list = []

        for i in range(self.unit_num_2):
            fea_pool = F.interpolate(en2_cat[i + 1], size=(h // 4, w // 4), mode='bilinear')
            en2_pool_list.append(fea_pool)

        en2_merge = torch.cat(en2_pool_list, 1)

        # TODO: encoder4
        en4_input = self.catLayer_2(en2_merge)
        en4_cat, en4_output = self.enLayer_4(en4_input, en2_cat4)

        # TODO: code
        code = en4_cat[-1]

        en4_up_list = []

        for i in range(self.unit_num_4):
            fea_up = F.interpolate(en4_cat[i + 1], size=(h // 2, w // 2), mode='bilinear')
            en4_up_list.append(fea_up)

        en4_merge = torch.cat(en4_up_list, 1)

        # TODO: decoder2
        de2_input = self.catLayer_3(en4_merge)
        de2_cat, de2_output = self.deLayer_2(de2_input, de2_cat2)
        # de2_output_merge = self.active(torch.add(de2_output, en2_input))
        de2_output_merge = torch.add(de2_output, en2_input)
        de2_up_list = []

        for i in range(self.unit_num_2):
            fea_up = F.interpolate(de2_cat[i + 1], size=(h, w), mode='bilinear')
            if i == self.unit_num_2 - 1:
                fea_up = F.interpolate(de2_output_merge, size=(h, w), mode='bilinear')
            de2_up_list.append(fea_up)

        de2_merge = torch.cat(de2_up_list, 1)

        # TODO: decoder1
        de1_input = self.catLayer_4(de2_merge)
        de1_cat, de1_output = self.deLayer_1(de1_input, en1_cat)
        # de1_output = self.active(torch.add(de1_output, fea1))
        de1_output = torch.add(de1_output, fea1)

        return de1_output, code

class Encoder_Decoder_2(nn.Module):
    def __init__(self):
        super(Encoder_Decoder_2, self).__init__()
        self.channel_2 = 32
        self.channel_4 =48

        self.unit_num_2 = 4
        self.unit_num_4 = 2

        self.enLayer_2 = DenseConnection(self.channel_2, UnitBlock(self.channel_2), self.unit_num_2)
        self.enLayer_4 = DenseConnection(self.channel_4, UnitBlock(self.channel_4), self.unit_num_4)
        self.deLayer_2 = DenseConnection(self.channel_2, UnitBlock(self.channel_2), self.unit_num_2)

        # TODO: catConv
        self.catLayer_2 = nn.Sequential(
            nn.Conv2d(self.channel_2 * self.unit_num_2, self.channel_4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.catLayer_3 = nn.Sequential(
            nn.Conv2d(self.channel_4 * self.unit_num_4, self.channel_2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # TODO: active
        self.active = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, fea2, en4_cat4):
        b, c, h, w = fea2.shape[0], fea2.shape[1], fea2.shape[2], fea2.shape[3]

        # TODO: encoder2
        en2_cat, en2_output = self.enLayer_2(fea2)
        en2_pool_list = []

        for i in range(self.unit_num_2):
            fea_pool = F.interpolate(en2_cat[i + 1], size=(h // 2, w // 2), mode='bilinear')
            en2_pool_list.append(fea_pool)

        en2_merge = torch.cat(en2_pool_list, 1)

        # TODO: encoder4
        en4_input = self.catLayer_2(en2_merge)
        en4_cat, en4_output = self.enLayer_4(en4_input, en4_cat4)
        en4_up_list = []

        for i in range(self.unit_num_4):
            fea_up = F.interpolate(en4_cat[i + 1], size=(h, w), mode='bilinear')
            en4_up_list.append(fea_up)

        en4_merge = torch.cat(en4_up_list, 1)

        # TODO: decoder2
        de2_input = self.catLayer_3(en4_merge)
        de2_cat, de2_output = self.deLayer_2(de2_input, en2_cat)
        # de2_output = self.active(torch.add(de2_output, fea2))
        de2_output = torch.add(de2_output, fea2)

        return de2_output, en2_cat, de2_cat, en4_cat

class Encoder_Decoder_4(nn.Module):
    def __init__(self):
        super(Encoder_Decoder_4, self).__init__()
        self.channel_4 = 48
        self.unit_num_4 = 2

        self.enLayer_4 = DenseConnection(self.channel_4, UnitBlock(self.channel_4), self.unit_num_4)

        # TODO: active
        self.active = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, fea4):
        en4_cat, en4_output = self.enLayer_4(fea4)
        # de4_output = self.active(torch.add(fea4, en4_output))
        de4_output = torch.add(fea4, en4_output)
        return de4_output, en4_cat

class ODE_DerainNet(nn.Module):
    def __init__(self):
        super(ODE_DerainNet, self).__init__()
        self.head = Head()
        self.encoder_decoder_1 = Encoder_Decoder_1()
        self.encoder_decoder_2 = Encoder_Decoder_2()
        self.encoder_decoder_4 = Encoder_Decoder_4()
        self.tail = Tail()

    def forward(self, img):
        fea1, fea2, fea4 = self.head(img)
        de4_output, en4_cat4 = self.encoder_decoder_4(fea4) # [6, 48, 30, 30]
        de2_output, en2_cat2, de2_cat2, en2_cat4 = self.encoder_decoder_2(fea2, en4_cat4) # [6, 32, 60, 60]
        de1_output, code = self.encoder_decoder_1(fea1, en2_cat2, de2_cat2, en2_cat4) # [6, 24, 120, 120]
        fake_S = self.tail(de1_output, de2_output, de4_output) # [6, 3, 120, 120]
        derain = img - fake_S # [6, 3, 120, 120]

        return derain, fake_S, code
