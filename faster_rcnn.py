# -*- coding: utf-8 -*-
from torch import nn
from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from lib.creator_tool import ProposalCreator
from lib.generate_anchors import generate_anchor_base
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils import data as data_
from lib.array_tool import scalar
from data.dataset import Dataset,TestDataset
from lib.eval_tool import eval_detection_voc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")  #torch.device代表将torch.Tensor分配到的设备的对象
def decom_vgg16():
    
    vgg = models.vgg16(pretrained=True, progress=True)
    
    extractor = nn.Sequential(*list(vgg.features)[:-1])
    classifier = vgg.classifier
    
    del classifier[6]
    
    
    for i,layer in enumerate(extractor):
        if i>=10:
            break
        if isinstance(layer,torch.nn.modules.conv.Conv2d):
            for p in layer.parameters():
                p.requires_grad = False
    #print(extractor)
    #print(classifier)
    return extractor, classifier
def normal_init(m, mean, stddev, truncated=False):
    
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)#
    shift = np.stack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel()), axis=1)#  .ravel() 将数组展平输出沿y轴拼接的数组
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

class RegionProposalNetwork(nn.Module):
    

    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],anchor_scales=[8, 16, 32], feat_stride=16,
                 proposal_creator_params=dict(),):
        super(RegionProposalNetwork, self).__init__()#等同于运行dengnn.Module.__init__(self)父类
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)# 调用generate_anchor_base（）函数，
        #生成左上角9个anchor_base
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)#为Fast-RCNN也即检测网络提供2000个训练样本,极大值抑制
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
    def forward(self, x, img_size, scale=1.):
        
        n, _, hh, ww = x.shape
        #n是batchsize大小
        # print self.anchor_base
        # print self.anchor_base.shape
         # 所有特征图上9种锚点的坐标
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base),self.feat_stride, hh, ww)
        # print anchor动态获取Faster RCNN首先是支持输入任意大小的图片的，
        #进入网络之前对图片进行了规整化尺度的设定，如可设定图像短边不超过600，图像长边不超过1000，
        #我们可以假定M*N=1000*600（如果图片少于该尺寸，可以边缘补0，即图像会有黑色边缘）
        #print anchor.shape
        n_anchor = anchor.shape[0] // (hh * ww)#一个中心点anchor的数量，return (K*A, 4) ， K = hh*ww  ，K约为20000
        #anchor = anchor.reshape((K * A, 4)).astype(np.float32)
        h = F.relu(self.conv1(x))
        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes，contiguous保持内存连续可以加快执行速度，reshape函数似乎可以更好地完成这个工作
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)#返回具有相同数据但大小不同的新张量
        #permute：将tensor的维度换位，
        #有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，
        #这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        #对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1，输出与输入相同尺寸和形状的张量器
        #dim:指明维度，dim=0表示按列计算；dim=1表示按行计算,4表示维度按照
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)
        # 经过nms(极大值抑制)获得的roi
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
        # 将list转为numpy格式，
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        #printf(roi.shape)
        crois = np.zeros((rois.shape[0],rois.shape[1]+1), dtype = np.float32)
        crois[:,0] = roi_indices
        crois[:,1:] = rois
        
        return rpn_locs, rpn_scores, crois, anchor
    
from torchvision.ops import RoIAlign#插值改进

def totensor(data, cuda=True):
    
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.to(device)
    return tensor

class VGG16RoIHead(nn.Module):
    

    def __init__(self, n_class, roi_size, spatial_scale,classifier):

        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIAlign(self.roi_size, self.spatial_scale, 1)#池化操作 output (Tensor[K, C, output_size[0], output_size[1]])
        #def __init__(self, output_size, spatial_scale, sampling_ratio):

    def forward(self, x, rois):
       
        # 避免假使 roi_indices is  ndarray
        rois = totensor(rois).float()        
        pool = self.roi(x, rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores
from lib.array_tool import totensor, tonumpy
from lib.bbox_tool import loc2bbox
from torchvision.ops import nms

def nograd(f):
   
    def new_f(*args,**kwargs):
        with torch.no_grad():
            return f(*args,**kwargs)
    return new_f

class FasterRCNN(nn.Module):
    
    def __init__(self, extractor, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.nms_thresh = 0.3
        self.score_thresh = 0.05
        
    @property
    def n_class(self):
        # 所有的类别总数，包含背景
        return self.head.n_class
    
    def forward(self, x, scale=1.):
        
        img_size = x.shape[2:]
        h = self.extractor(x)
        
        rpn_loss, rpn_scores, crois, anchors = self.rpn(h, img_size, scale)
        
        roi_cls_locs, roi_scores = self.head(h, crois)
        
        return roi_cls_locs, roi_scores, crois
        
    def _suppress(self, raw_cls_bbox, raw_prob):
       
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            
            keep = nms(totensor(cls_bbox_l), totensor(prob_l), self.nms_thresh)
            keep = tonumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score
    
    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        
        self.eval()
        prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = totensor(rois[:,1:]) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")  #torch.device代表将torch.Tensor分配到的设备的对象
            mean = torch.Tensor(self.loc_normalize_mean).to(device).repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).to(device).repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(tonumpy(roi).reshape((-1, 4)),tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[1])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[0])

            prob = tonumpy(F.softmax(totensor(roi_score), dim=1))

            raw_cls_bbox = tonumpy(cls_bbox)
            raw_prob = tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        
        self.train()   
        
        return bboxes, labels, scores
        
    def get_optimizer(self):
        lr = 1e-3
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        
        return self.optimizer
    
    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
class FasterRCNNVGG16(FasterRCNN):
    

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = decom_vgg16()#基础网络

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )#RPN模块

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=(7,7),
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )#ROIHead部分

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )#继承父类也就是相当于执行FasterRCNN.__init__()
from lib.creator_tool import AnchorTargetCreator, ProposalTargetCreator

class FasterRCNNTrainer(nn.Module):
    

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 3.#loss公式的参数
        self.roi_sigma = 1.#

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

    def forward(self, imgs, bboxes, labels, scale):
       
        n = bboxes.shape[0]#batchsize数量
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, anchor = self.faster_rcnn.rpn(features, img_size, scale)#rpn_locs的维度（hh*ww*9，4），
        #rpn_scores维度为（hh*ww*9，2）， rois的维度为（2000,4），roi_indices用不到，anchor的维度为（hh*ww*9，4），H和W是经过数据预处理后的。
        #计算（H/16）x(W/16)x9(大概20000)个anchor属于前景的概率，取前12000个并经过NMS得到2000个近似目标框G^的坐标。roi的维度为(2000,4)

        # 程序限定N=1，把批维度去掉方便操作
        bbox = bboxes[0] #bbox维度(N, R, 4)
        label = labels[0] #labels维度为（N，R）
        rpn_score = rpn_scores[0] #（hh*ww*9，4）
        rpn_loc = rpn_locs[0] #hh*ww*9
        roi = rois #(2000,4)


        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            tonumpy(bbox),
            tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        
        #因为ProposalTargetCreator的设计问题，此处需要插入一列idx
        sample_roi_index = np.zeros(len(sample_roi)) 
        sample_roi = np.insert(sample_roi, 0, values=sample_roi_index, axis=1)
        
        roi_cls_loc, roi_score = self.faster_rcnn.head(features, sample_roi)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(tonumpy(bbox), anchor, img_size)
        
        gt_rpn_label = totensor(gt_rpn_label).long()
        gt_rpn_loc = totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...索引默认值
        
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.to(device), ignore_index=-1)

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().to(device), totensor(gt_roi_label).long()]
        
        gt_roi_label = totensor(gt_roi_label).long()
        gt_roi_loc = totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)
        #迷惑行为  self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.to(device))

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return losses
    
    
    def save(self, save_path):
        save_dict= self.faster_rcnn.state_dict()
        torch.save(save_dict, save_path)
        return save_path
    
    def load(self, path):
        state_dict = torch.load(path)
        return self

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses[-1].backward()
        self.optimizer.step()

        return losses

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).to(device)
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 与源代码有出入
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).to(device)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: 
            break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, use_07_metric=True)
    
    return result
def train(**kwargs):
    #opt._parse(kwargs)#将调用函数时候附加的参数用，config.py文件里面的opt._parse()进行解释，然后获取其数据存储的路径，之后放到Dataset里面！
    path_data='/home/yxm218/data/datasets/VOC/VOCdevkit/VOC2012/'
    dataset = Dataset(path_data)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=8) #Dataset完成的任务见第一篇博客数据预处理部分，这里简单解释一下，就是用VOCBboxDataset作为数据读取库，然后依次从样例数据库中读取图片出来，还调用了Transform(object)函数，完成图像的调整和随机反转工作！
    path_test='/home/yxm218/data/datasets/VOC/VOCdevkit/VOC2012'
    testset = TestDataset(path_test)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=8,
                                       shuffle=False, \
                                       pin_memory=True
                                       ) #将数据装载到dataloader中，shuffle=True允许数据打乱排序，num_workers是设置数据分为几批处理，同样的将测试数据集也进行同样的处理，然后装载到test_dataloader中！
    faster_rcnn = FasterRCNNVGG16()#接下来定义faster_rcnn=FasterRCNNVGG16()定义好模型
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).to(device) #设置trainer = FasterRCNNTrainer(faster_rcnn).cuda()将FasterRCNNVGG16作为fasterrcnn的模型送入到FasterRCNNTrainer中并设置好GPU加速
    best_map = 0
    lr_ = 1e-3
    running_loss = 0.
    for epoch in range(14): #之后用一个for循环开始训练过程，而训练迭代的次数opt.epoch=14也在config.py文件中都预先定义好，属于超参数
        #trainer.reset_meters() #首先在可视化界面重设所有数据
       # for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
        for ii, (img, bbox_, label_, scale) in enumerate(dataloader):
            scale = scalar(scale)
            img, bbox, label = img.to(device).float(), bbox_.to(device), label_.to(device) #然后从训练数据中枚举dataloader,设置好缩放范围，
            #将img,bbox,label,scale全部设置为可gpu加速
            losses = trainer.train_step(img, bbox, label, scale)
            running_loss += losses[-1].item()
            if (ii + 1) % 40 == 0:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, ii + 1, running_loss / 40))
                running_loss = 0.0
                
        eval_result = eval(test_dataloader, faster_rcnn, test_num=1000)
        

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save("./save/fasterRcnn.pth") #用if判断语句永远保存效果最好的map
            print("epoch:" ,epoch, "map:",eval_result['map'])
       # if epoch == 9:
          #  trainer.load(best_path)
           # trainer.faster_rcnn.scale_lr(opt.lr_decay)
           # lr_ = lr_ * opt.lr_decay #if判断语句如果学习的epoch达到了9就将学习率*0.1变成原来的十分之一
        if epoch == 9:
            trainer.faster_rcnn.scale_lr(0.1)
            lr_ = lr_ * 0.1
        if epoch == 13: 
            break #判断epoch==13结束训练验证过程

