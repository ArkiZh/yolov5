# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, pred, targets):  # predictions, targets
        loss_cls = torch.zeros(1, device=self.device)  # class loss
        loss_box = torch.zeros(1, device=self.device)  # box loss
        loss_obj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(pred, targets)  # targets
        # 上面返回的是每个检测层中：锚框与真实框组合过滤后再与当前特征图中的grid组合
        # Losses
        for i, layer_pred in enumerate(pred):  # layer index, layer predictions  Shape: [B,A,H,W,C]
            img_id, anchor_id, grid_y, grid_x = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(layer_pred.shape[:4], dtype=layer_pred.dtype, device=self.device)  # target obj

            n = img_id.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = layer_pred[img_id, anchor_id, grid_y, grid_x].split((2, 2, 1, self.nc), dim=1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                loss_box += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    img_id, anchor_id, grid_y, grid_x, iou = img_id[j], anchor_id[j], grid_y[j], grid_x[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[img_id, anchor_id, grid_y, grid_x] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    loss_cls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(layer_pred[..., 4], tobj)
            loss_obj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        loss_box *= self.hyp['box']
        loss_obj *= self.hyp['obj']
        loss_cls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (loss_box + loss_obj + loss_cls) * bs, torch.cat((loss_box, loss_obj, loss_cls)).detach()

    def build_targets(self, pred, targets):
        # Build targets for compute_loss(), input targets(image index inside current batch, class index, x, y, w, h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)   Shape: na,nt
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        # targets形状从 [nt, 6] 变成了 [na, nt, 7], 第0维为na个anchor，第1维为nt个真实框，第2维为真实框信息：SampleIndex,ClassIndex,x,y,w,h,AnchorIndex

        g = 0.5  # bias
        offset = torch.tensor(
            [
                [0, 0],  # gxy 不偏移              无论何时都分配到当前grid
                [1, 0],  # gxy x方向偏移           到当前grid左边线不足g，分配到左边grid
                [0, 1],  # gxy y方向偏移           到当前grid上边线不足g，分配到上面grid
                [-1, 0], # gxy inverse x方向偏移   到当前grid右边线不足g，分配到右边grid
                [0, -1], # gxy inverse y方向偏移   到当前grid下边线不足g，分配到下面grid
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], pred[i].shape
            batch_size, feature_na, feature_h, feature_w, feature_c = pred[i].shape
            gain[2:6] = torch.tensor([feature_w, feature_h, feature_w, feature_h])  # xyxy gain  获取当前检测层特征图的宽高，对

            # Match targets to anchors
            t = targets * gain  # shape(na,nt,7)
            if nt:
                # Matches  为每个真实框找可以形状匹配的锚框
                r = t[..., 4:6] / anchors[:, None]  # w,h ratio  Shape: [na, nt, 2] 对多有真实框，计算与nt个锚框的宽度比例、高度比例
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare 筛选出宽度比例与高度比例小于阈值的 Shape: [na, nt], 每一行代表与当前锚框形状接近的真实框
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter 筛选出根据形状阈值匹配到的真实框锚框组合 Shape: [M, 7]  M为匹配到的组合数量  TODO 可能出现真实框没有找到形状接近的锚框情况，怎么处理？
                M = t.shape[0]
                
                # Offsets
                truth_xy = t[:, 2:4]  # grid xy 真实框在当前特征图中的中心点xy坐标（距离左上角点的距离）  Shape: [M, 2]
                gxy_inverse = gain[[2, 3]] - truth_xy  # inverse  翻转成距离右下角点的距离
                hit_x, hit_y = ((truth_xy % 1 < g) & (truth_xy > 1)).T  # 对xy坐标分别判断是否：大于1且模1得到的小数部分小于g。 hit_x:满足条件的x坐标；hit_y：满足条件的y坐标。 [M,2] --> [2,M]
                hit_inverse_x, hit_inverse_y = ((gxy_inverse % 1 < g) & (gxy_inverse > 1)).T
                offset_hit = torch.stack([torch.ones_like(hit_x), hit_x, hit_y, hit_inverse_x, hit_inverse_y])  # Shape: [5, M]
                t = t.repeat((5, 1, 1))[offset_hit]   # Shape: [M, 7] --> [5, M, 7] --> [Q, 7]  Q为offset_hit中True的个数
                offsets = offset[:,None].repeat(1,M,1)[offset_hit]  # Shape: [5, 2] --> [5,1,2] --> [5,M,2] --> [Q, 2]  Q为offset_hit中True的个数
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, truth_xy, truth_wh, anchor_id = t.chunk(chunks=4, dim=1)  # (imageId, classId), 真实框xy, 真实框wh, anchors
            anchor_id, (img_id, class_id) = anchor_id.long().view(-1), bc.long().T  # anchors, image, class
            grid_xy = (truth_xy - offsets).long()
            grid_x, grid_y = grid_xy[:,0], grid_xy[:, 1]  # grid indices 

            # Append  不需要对grid坐标clip了，找grid时边缘一格只用当前grid，不偏移。 grid_y.clamp_(0, shape[2] - 1), grid_x.clamp_(0, shape[3] - 1)
            indices.append((img_id, anchor_id, grid_y, grid_x))  # imageId, anchorId, grid_y, grid_x 的shape一样：[Q]
            tbox.append(torch.cat((truth_xy - grid_xy, truth_wh), 1))  # 每个grid的回归目标：真实中心相对当前grid的xy偏移，真实wh。 Shape: [Q, 4]
            anch.append(anchors[anchor_id])  # anchors  Shape: [Q, 4]
            tcls.append(class_id)  # class  Shape: [Q]

        return tcls, tbox, indices, anch
