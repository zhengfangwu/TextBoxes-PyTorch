import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import L2Norm, PriorBoxLayer

class Net(torch.nn.Module):

    def __init__(self, min_size, max_size, aspect_ratios, clip=True, use_cuda=True):
        super(Net, self).__init__()

        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios
        self.clip = clip
        self.use_cuda = use_cuda

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=6, padding=6)
        self.fc7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        
        self.conv6_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv8_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.conv4_3_norm = L2Norm(in_channels=512, scale_init=20.0)

        self.conv4_3_norm_mbox_conf = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 5), stride=1, padding=(0, 2))
        self.conv4_3_norm_mbox_loc = nn.Conv2d(in_channels=512, out_channels=48, kernel_size=(1, 5), stride=1, padding=(0, 2))
        self.fc7_mbox_conf = nn.Conv2d(in_channels=1024, out_channels=28, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.fc7_mbox_loc = nn.Conv2d(in_channels=1024, out_channels=56, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv6_2_mbox_conf = nn.Conv2d(in_channels=512, out_channels=28, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv6_2_mbox_loc = nn.Conv2d(in_channels=512, out_channels=56, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv7_2_mbox_conf = nn.Conv2d(in_channels=256, out_channels=28, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv7_2_mbox_loc = nn.Conv2d(in_channels=256, out_channels=56, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv8_2_mbox_conf = nn.Conv2d(in_channels=256, out_channels=28, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv8_2_mbox_loc = nn.Conv2d(in_channels=256, out_channels=56, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.pool6_mbox_conf = nn.Conv2d(in_channels=256, out_channels=28, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.pool6_mbox_loc = nn.Conv2d(in_channels=256, out_channels=56, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))


        self.conv4_3_norm_mbox_priorbox = PriorBoxLayer(min_size[0], max_size[0], aspect_ratios, clip, use_cuda)
        self.fc7_mbox_priorbox = PriorBoxLayer(min_size[1], max_size[1], aspect_ratios, clip, use_cuda)
        self.conv6_2_mbox_priorbox = PriorBoxLayer(min_size[2], max_size[2], aspect_ratios, clip, use_cuda)
        self.conv7_2_mbox_priorbox = PriorBoxLayer(min_size[3], max_size[3], aspect_ratios, clip, use_cuda)
        self.conv8_2_mbox_priorbox = PriorBoxLayer(min_size[4], max_size[4], aspect_ratios, clip, use_cuda)
        self.pool6_mbox_priorbox = PriorBoxLayer(min_size[5], max_size[5], aspect_ratios, clip, use_cuda)

        if use_cuda:
            self.cuda()

    def forward(self, input):
        batch_size = input.size(0)

        conv1_1 = self.conv1_1(input)
        relu1_1 = F.relu(conv1_1, inplace=True)
        conv1_2 = self.conv1_2(relu1_1)
        relu1_2 = F.relu(conv1_2, inplace=True)
        pool1 = self.pool1(relu1_2)
        conv2_1 = self.conv2_1(pool1)
        relu2_1 = F.relu(conv2_1, inplace=True)
        conv2_2 = self.conv2_2(relu2_1)
        relu2_2 = F.relu(conv2_2, inplace=True)
        pool2 = self.pool2(relu2_2)
        conv3_1 = self.conv3_1(pool2)
        relu3_1 = F.relu(conv3_1, inplace=True)
        conv3_2 = self.conv3_2(relu3_1)
        relu3_2 = F.relu(conv3_2, inplace=True)
        conv3_3 = self.conv3_3(relu3_2)
        relu3_3 = F.relu(conv3_3, inplace=True)
        pool3 = self.pool3(relu3_3)
        conv4_1 = self.conv4_1(pool3)
        relu4_1 = F.relu(conv4_1, inplace=True)
        conv4_2 = self.conv4_2(relu4_1)
        relu4_2 = F.relu(conv4_2, inplace=True)
        conv4_3 = self.conv4_3(relu4_2)
        relu4_3 = F.relu(conv4_3, inplace=True)
        conv4_3_norm = self.conv4_3_norm(relu4_3)       # 1st feature map
        pool4 = self.pool4(relu4_3)
        conv5_1 = self.conv5_1(pool4)
        relu5_1 = F.relu(conv5_1, inplace=True)
        conv5_2 = self.conv5_2(relu5_1)
        relu5_2 = F.relu(conv5_2, inplace=True)
        conv5_3 = self.conv5_3(relu5_2)
        pool5 = self.pool5(conv5_3)
        fc6 = self.fc6(pool5)
        relu6 = F.relu(fc6, inplace=True)
        fc7 = self.fc7(relu6)
        relu7 = F.relu(fc7)                             # 2nd feature map
        conv6_1 = self.conv6_1(relu7)
        conv6_1_relu = F.relu(conv6_1, inplace=True)
        conv6_2 = self.conv6_2(conv6_1_relu)
        conv6_2_relu = F.relu(conv6_2, inplace=True)    # 3rd feature map
        conv7_1 = self.conv7_1(conv6_2_relu)
        conv7_1_relu = F.relu(conv7_1, inplace=True)
        conv7_2 = self.conv7_2(conv7_1_relu)
        conv7_2_relu = F.relu(conv7_2, inplace=True)    # 4th feature map
        conv8_1 = self.conv8_1(conv7_2_relu)
        conv8_1_relu = F.relu(conv8_1, inplace=True)
        conv8_2 = self.conv8_2(conv8_1_relu)
        conv8_2_relu = F.relu(conv8_2, inplace=True)    # 5th feature map

        # global avg. pooling
        kH = conv8_2_relu.size(2)
        kW = conv8_2_relu.size(3)
        pool6 = F.avg_pool2d(conv8_2_relu, kernel_size=(kH, kW))    # 6th feature map

        conv4_3_norm_mbox_conf_flat = self.conv4_3_norm_mbox_conf(conv4_3_norm).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        fc7_mbox_conf_flat = self.fc7_mbox_conf(relu7).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        conv6_2_mbox_conf_flat = self.conv6_2_mbox_conf(conv6_2_relu).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        conv7_2_mbox_conf_flat = self.conv7_2_mbox_conf(conv7_2_relu).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        conv8_2_mbox_conf_flat = self.conv8_2_mbox_conf(conv8_2_relu).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        pool6_mbox_conf_flat = self.pool6_mbox_conf(pool6).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        mbox_conf = torch.cat([conv4_3_norm_mbox_conf_flat, fc7_mbox_conf_flat, conv6_2_mbox_conf_flat, conv7_2_mbox_conf_flat, conv8_2_mbox_conf_flat, pool6_mbox_conf_flat], 1)
        mbox_conf_flatten = F.softmax(mbox_conf.view(batch_size, -1, 2), dim=2).view(batch_size, -1)

        conv4_3_norm_mbox_loc_flat = self.conv4_3_norm_mbox_loc(conv4_3_norm).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        fc7_mbox_loc_flat = self.fc7_mbox_loc(relu7).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        conv6_2_mbox_loc_flat = self.conv6_2_mbox_loc(conv6_2_relu).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        conv7_2_mbox_loc_flat = self.conv7_2_mbox_loc(conv7_2_relu).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        conv8_2_mbox_loc_flat = self.conv8_2_mbox_loc(conv8_2_relu).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        pool6_mbox_loc_flat = self.pool6_mbox_loc(pool6).permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        mbox_loc = torch.cat([conv4_3_norm_mbox_loc_flat, fc7_mbox_loc_flat, conv6_2_mbox_loc_flat, conv7_2_mbox_loc_flat, conv8_2_mbox_loc_flat, pool6_mbox_loc_flat], 1)
        mbox_loc_flatten = F.softmax(mbox_loc.view(batch_size, -1, 2), dim=2).view(batch_size, -1)

        priors = []
        priors.append(self.conv4_3_norm_mbox_priorbox(input, conv4_3_norm))
        priors.append(self.fc7_mbox_priorbox(input, relu7))
        priors.append(self.conv6_2_mbox_priorbox(input, conv6_2_relu))
        priors.append(self.conv7_2_mbox_priorbox(input, conv7_2_relu))
        priors.append(self.conv8_2_mbox_priorbox(input, conv8_2_relu))
        priors.append(self.pool6_mbox_priorbox(input, pool6))
        print(conv4_3_norm.size())
        print(fc7.size())
        print(conv6_2_relu.size())
        print(conv7_2_relu.size())
        print(conv8_2_relu.size())
        print(pool6.size())

        return mbox_conf_flatten, mbox_loc_flatten, priors

if __name__ == "__main__":
    img_h = 300
    img_w = 300
    # 20% ~ 95% of image size
    min_size = [30, 60, 114, 168, 222, 276] 
    max_size = [None, 114, 167, 222, 276, 330]
    aspect_ratios = [2, 3, 5, 7, 10]
    net = Net(min_size, max_size, aspect_ratios)
    input = torch.Tensor(1, 3, 300, 300).cuda()
    output = net(input)
    print(output[0].size())
    print(output[1].size())
    for i in range(6):
        print(output[2][i].size())