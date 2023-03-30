import itertools
import os
import PIL
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
import pytorch_lightning as pl
from torchvision import transforms
from ldm.lr_scheduler import LambdaLinearScheduler
from torch.optim.lr_scheduler import LambdaLR

# from utils.lr_scheduler import LambdaLinearScheduler
from ldm.util import instantiate_from_config
from utils.SCD_metrics import RunningMetrics
from utils.scdloss import SCDLoss
from utils.helpers import DeNormalize
import torch.optim.lr_scheduler as lrs
from ldm.data import Index2Color, Color2Index, TensorIndex2Color

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
from torchvision import models
class FCN(nn.Module):
    def __init__(self, in_channels=3,  pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()        
        self.FCN = FCN(in_channels,pretrained=pretrained)
    
    def base_forward(self, x):
       
        x = self.FCN.layer0(x) #size:1/2
        x = self.FCN.maxpool(x) #size:1/4
        x = self.FCN.layer1(x) #size:1/4
        x = self.FCN.layer2(x) #size:1/8
        x = self.FCN.layer3(x) #size:1/8
        x = self.FCN.layer4(x) #size:1/8
        return x

    def forward(self, x1, x2):
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        return [x1, x2]


class SCDHead(nn.Module):
    def __init__(self, in_channel,num_classes=7) -> None:
        super().__init__()
        self.classifier1 = nn.Conv2d(in_channel, num_classes, kernel_size=1)
        
        self.classifier2 = nn.Conv2d(in_channel, num_classes, kernel_size=1)
        
        self.classifierCD = nn.Sequential(nn.Conv2d(in_channel, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
    
    def forward(self, x1, x2, change):
        change = self.classifierCD(change)
        x1 = self.classifier1(x1)
        x2 = self.classifier2(x2)       
        return x1, x2, change

class Neck(nn.Module):
    def __init__(self,embed_dim, mid_dim=128) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(embed_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(mid_dim), nn.ReLU())
        self.resCD = self._make_layer(ResBlock, mid_dim*2, mid_dim, 3, stride=1)
        
        
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )
        
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    def base_forward(self, x1, x2):
        x1 = self.head(x1)
        x2 = self.head(x2)
        change = [x1, x2]
        change = torch.cat(change, 1)
        change = self.resCD(change)
        return x1, x2, change

    def forward(self, x1, x2):
         return self.base_forward(x1, x2)
  

class SCDDecoder(nn.Module):
    def __init__(self, num_classes=7, mid_dim=128, x_size=(512,512)) -> None:
        super().__init__()
        self.neck = Neck(embed_dim=512, mid_dim=mid_dim)
        self.head = SCDHead(mid_dim, num_classes)   
        self.x_size = x_size    
       
    def forward(self, x, perturbation = None, o_l = None):     
        if perturbation is not None:
            x = torch.cat(x, dim=1)
            x = perturbation(x, o_l)
            x1, x2 = torch.chunk(x, dim=1,chunks=2) 
        else:
            x1, x2 = x
        x1, x2, change = self.neck(x1, x2)
        x1, x2, change = self.head(x1, x2, change)
        return [F.interpolate(x1, self.x_size, mode='bilinear', align_corners=True), 
                F.interpolate(x2, self.x_size, mode='bilinear', align_corners=True), 
                F.interpolate(change, self.x_size, mode='bilinear', align_corners=True)]

class SCDNet(pl.LightningModule):
    def __init__(self,in_channels=3, monitor=None,
                num_classes=7, mid_dim=128,
                scheduler_config=None) -> None:
        super().__init__()  
        if monitor is not None:
            self.monitor = monitor 
        self.encoder = Encoder(in_channels)
        self.decoder = SCDDecoder(num_classes=num_classes, mid_dim=mid_dim)
        self.loss_func = SCDLoss()
        self.running_meters = RunningMetrics(num_classes=7)
        self.scheduler_config = scheduler_config
        
    def encode(self, x1, x2):
        return self.encoder(x1, x2)

    def decode(self, x, perturbation = None, o_l = None):
        return self.decoder(x, perturbation, o_l)
        
    def forward(self, x1, x2):
        x = self.encode(x1, x2)
        return self.decode(x)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    # 这里写反向传播过程
    def training_step(self, batch):
        x1 = batch['image_A']
        x2 = batch['image_B']
        bs = x1.size(0)
        target_A = batch['target_A'].long()
        target_B = batch['target_B'].long()
        x1, x2, change = self(x1, x2)
        target_bn = (target_B>0).float()
        loss = self.loss_func(x1, x2, change, target_bn, target_A, target_B)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log('train_lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True, batch_size=bs)
        return loss
     
    
    # 这里配置优化器
    def configure_optimizers(self):
        opt = torch.optim.SGD(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=self.learning_rate, weight_decay=1e-4, momentum=0.9)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                    {
                        'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                        'interval': 'step',
                        'frequency': 1
                    }
                    ]
            return [opt], scheduler
        return [opt], []
        
    
    def get_prediction(self, preds, domain='sup'):
        x1, x2, change = preds
        change_mask = torch.sigmoid(change).detach()>0.5
        p1 = torch.argmax(x1, dim=1)
        p2 = torch.argmax(x2, dim=1)
        p1 = p1 * change_mask.squeeze().long()
        p2 = p2 * change_mask.squeeze().long()
        del change_mask, x1, x2
        return {
            f'{domain}_pred_A': p1, 
            f'{domain}_pred_B': p2,
            # f'{domain}_change': change_mask.long(),
            }
    
    # 这里写一个epoch后的验证过程
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x1 = batch['image_A']
        x2 = batch['image_B']
        target_A = batch['target_A']
        target_B = batch['target_B']
        x1, x2, change = self(x1, x2)
        preds = dict()
        preds.update(self.get_prediction((x1, x2, change), domain='val'))
        
        bs = x1.size(0)
        self.running_meters.update(target_A.detach().cpu().numpy(), preds['val_pred_A'].detach().cpu().numpy())
        self.running_meters.update(target_B.detach().cpu().numpy(), preds['val_pred_B'].detach().cpu().numpy())
        scores = self.running_meters.get_scores()
        self.log('val/F_scd', scores['F_scd'], on_epoch=True,  logger=True, batch_size=bs)
        self.log('val/miou', scores['Mean_IoU'], on_epoch=True, logger=True, batch_size=bs)
        self.log('val/OA', scores['OA'], on_epoch=True,  logger=True, batch_size=bs)
        self.log('val/SeK', scores['Sek'], on_epoch=True, logger=True, batch_size=bs)
        
        return preds  
          
    def on_validataion_epoch_end(self):
        
        self.running_meters.reset()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x1, x2 = batch['image_A'], batch['image_B']
        bs = x1.size(0)
        image_id = batch['image_id'][0]
        target_A, target_B = batch['target_A'], batch['target_B']
        x1, x2, change = self(x1, x2)
        preds = dict()
        preds.update(self.get_prediction((x1, x2, change), domain='test'))
        self.running_meters.update(target_A.detach().cpu().numpy(), preds['test_pred_A'].detach().cpu().numpy())
        self.running_meters.update(target_B.detach().cpu().numpy(), preds['test_pred_B'].detach().cpu().numpy())
           
        preds['test_pred_A'] = Index2Color(preds['test_pred_A'].detach().cpu().numpy().squeeze())
        preds['test_pred_B'] = Index2Color(preds['test_pred_B'].detach().cpu().numpy().squeeze())
        # save images
        save_dir = os.path.join(self.trainer.log_dir, os.pardir, os.pardir)
        save_dir = os.path.join(save_dir, 'test_results')
        
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        pred_A_L_show = PIL.Image.fromarray(preds['test_pred_A'].astype(np.uint8)).convert('RGB')
        pred_B_L_show = PIL.Image.fromarray(preds['test_pred_B'].astype(np.uint8)).convert('RGB')
        pred_A_L_show.save(os.path.join(save_dir, image_id+'_pred_A_L_show.png'))
        pred_B_L_show.save(os.path.join(save_dir, image_id+'_pred_B_L_show.png'))
        
        # 记录测试精度 
        # scores = self.running_meters.get_scores()
        # self.log('test/F_scd', scores['F_scd'], on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        # self.log('test/miou', scores['Mean_IoU'], on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        # self.log('test/OA', scores['OA'], on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        # self.log('test/SeK', scores['Sek'], on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        
        return preds
    def on_test_end(self) -> None:
        scores = self.running_meters.get_scores()
        message = 'Test: performance: F_scd: {:.4f}, miou: {:.4f}, OA: {:.4f}, Sek: {:.4f}'.format(
            scores['F_scd'], scores['Mean_IoU'], scores['OA'], scores['Sek'])
        print(message)
        self.running_meters.reset()
        
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        x1 = batch['image_A'], x2 = batch['image_B']
        image_id = batch['image_id']
        x1, x2, change = self(x1, x2)
        preds = {'image_id':image_id}
        preds.update(self.model.get_prediction((x1, x2, change), domain='test'))
        
        preds['test_pred_A'] = TensorIndex2Color(preds['test_pred_A']).detach().cpu().numpy().squeeze()
        preds['test_pred_B'] = TensorIndex2Color(preds['test_pred_B']).detach().cpu().numpy().squeeze()
        
        return preds
    
    @torch.no_grad()
    def log_images(self, batch, outputs, split='val', **kwargs):
        log = dict()
        A = batch['image_A']
        B = batch['image_B']
        denormalize = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        A = denormalize(A)# b,c,h,w
        B = denormalize(B)
        log['imageA'] = A
        log['imageB'] = B
        A_L = batch['target_A']# b,h,w,c
        B_L = batch['target_B']
        A_L = TensorIndex2Color(A_L)
        B_L = TensorIndex2Color(B_L)    
        log['target_A'] = A_L.permute(0,3,1,2)
        log['target_B'] = B_L.permute(0,3,1,2)
        log['pred_A'] = outputs[f'{split}_pred_A'].unsqueeze(1)
        log['pred_B'] = outputs[f'{split}_pred_B'].unsqueeze(1)
        return log

        
class SemiSCDNet(pl.LightningModule):
    def __init__(self,in_channels=3, monitor=None,
                num_classes=7, mid_dim=128,
                scheduler_config=None) -> None:
        super().__init__(in_channels=in_channels, monitor=monitor,
                         num_classes=num_classes, mid_dim=mid_dim,
                         scheduler_config=scheduler_config)  