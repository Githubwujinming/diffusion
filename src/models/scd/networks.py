from contextlib import contextmanager
import logging

from src.modules.ema import LitEma
from utils.metrics import weighted_BCE_logits
from .components import *
class SCDNet(pl.LightningModule):
    def __init__(self,in_channels=3, monitor=None,
                num_classes=7, mid_dim=128,
                scheduler_config=None, ckpt_path=None) -> None:
        super().__init__()  
        self.save_hyperparameters(ignore=['loss_func', 'running_meters', 'scheduler_config'])
        if monitor is not None:
            self.monitor = monitor 
        self.encoder = Encoder(in_channels)
        self.decoder = SCDDecoder(embed_dim=512, num_classes=num_classes, mid_dim=mid_dim)
        self.loss_func = SCDLoss()
        self.train_running_meters = RunningMetrics(num_classes=num_classes)
        self.running_meters = RunningMetrics(num_classes=num_classes)
        self.scheduler_config = scheduler_config
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        
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
        x1 = batch['image_A'].clone()
        x2 = batch['image_B'].clone()
        bs = x1.size(0)
        target_A = batch['target_A'].long()
        target_B = batch['target_B'].long()
        x1, x2, change = self(x1, x2)
        target_bn = (target_B>0)
        loss = self.loss_func(x1, x2, change, target_bn.float(), target_A, target_B)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True, batch_size=bs)
        
        sup_pred = self.get_prediction((x1, x2, change), domain='sup')
        self.train_running_meters.update(target_A.detach().cpu().numpy(), sup_pred['sup_pred_A'].detach().cpu().numpy())
        self.train_running_meters.update(target_B.detach().cpu().numpy(), sup_pred['sup_pred_B'].detach().cpu().numpy())
        return loss
    
    def on_train_epoch_end(self) -> None:
        # log metric for sup/unsup data
        sup_score = self.train_running_meters.get_scores()

        for k, v in sup_score.items():
            self.log(f'train/sup_{k}', v, on_step=False, on_epoch=True, logger=True)
        
        # logging in file
        train_logger = logging.getLogger('train')
        message = '[Training SCD (epoch %d summary)]: Fscd=%.5f \n' %\
                      (self.current_epoch, sup_score['F_scd'])
        for k, v in sup_score.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        train_logger.info(message)
        # reset mertric for next epoch
        self.running_meters.reset()
    
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
        
        self.running_meters.update(target_A.detach().cpu().numpy(), preds['val_pred_A'].detach().cpu().numpy())
        self.running_meters.update(target_B.detach().cpu().numpy(), preds['val_pred_B'].detach().cpu().numpy())
    
        return preds  
    
    def on_fit_start(self) -> None:
        val_logger = logging.getLogger('val')
        self.running_meters.reset()
        val_logger.info('\n###### EVALUATION ######')
        
    @torch.no_grad()
    def on_validation_epoch_end(self):
        # log the validation metrics
        val_logger = logging.getLogger('val')
        scores = self.running_meters.get_scores()
        self.log('val/F_scd', scores['F_scd'], on_epoch=True,  logger=True)
        self.log('val/miou', scores['Mean_IoU'], on_epoch=True, logger=True)
        self.log('val/OA', scores['OA'], on_epoch=True,  logger=True)
        self.log('val/SeK', scores['Sek'], on_epoch=True, logger=True)
        
        message  = f'Validation Summary Epoch: [{self.current_epoch}]\n'
        for k, v in scores.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        val_logger.info(message)
        self.running_meters.reset()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x1, x2 = batch['image_A'], batch['image_B']
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
        save_dir = os.path.join(self.trainer.log_dir)
        save_dir = os.path.join(save_dir, 'test_results')
        
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        pred_A_L_show = PIL.Image.fromarray(preds['test_pred_A'].astype(np.uint8)).convert('RGB')
        pred_B_L_show = PIL.Image.fromarray(preds['test_pred_B'].astype(np.uint8)).convert('RGB')
        pred_A_L_show.save(os.path.join(save_dir, image_id+'_pred_A_L_show.png'))
        pred_B_L_show.save(os.path.join(save_dir, image_id+'_pred_B_L_show.png'))
        
        return preds
    def on_test_end(self) -> None:
        scores = self.running_meters.get_scores()
        message = '=========Test: performance=========\n'
        for k, v in scores.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        val_logger = logging.getLogger('val')
        val_logger.info(message)
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

        
class SemiSCDNet(SCDNet):
    def __init__(self,in_channels=3, monitor=None, num_epochs=0,
                num_classes=7, mid_dim=128,iters_per_epoch=0,
                scheduler_config=None, rc_config=None, **kwargs) -> None:
        super().__init__(in_channels=in_channels, monitor=monitor,
                         num_classes=num_classes, mid_dim=mid_dim,
                         scheduler_config=scheduler_config)  
        self.sup_running_metric = RunningMetrics(num_classes)
        self.unsup_running_metric = RunningMetrics(num_classes)
        self.unsuper_loss = softmax_mse_loss
        self.unsup_loss_w = consistency_weight(final_w=rc_config['unsup_final_w'], iters_per_epoch=iters_per_epoch,
                                               rampup_ends=int(0.1*num_epochs))
        
        # confidence masking 
        self.confidence_th      = rc_config['confidence_th']
        self.confidence_masking = rc_config['confidence_masking']
        
        self.pertubs = self._make_perturbs(rc_config)
        
        
    def _make_perturbs(self, opt):
        assert opt is not None, "perturebation config is None"
        vat     = [VAT(xi=opt['xi'], eps=opt['eps']) for _ in range(opt['vat'])]
        drop    = [DropOut(drop_rate=opt['drop_rate'], spatial_dropout=opt['spatial'])
                                    for _ in range(opt['drop'])]
        cut     = [CutOut(erase=opt['erase']) for _ in range(opt['cutout'])]
        context_m = [ContextMasking() for _ in range(opt['context_masking'])]
        object_masking  = [ObjectMasking() for _ in range(opt['object_masking'])]
        feature_drop    = [FeatureDrop() for _ in range(opt['feature_drop'])]
        feature_noise   = [FeatureNoise(uniform_range=opt['uniform_range'])
                                    for _ in range(opt['feature_noise'])]
        return nn.ModuleList([*vat, *drop, *cut,
                                *context_m, *object_masking, *feature_drop, *feature_noise])
        # 半监督学习的step
    def training_step(self, batch, batch_idx):
        sup_data, unsup_data = batch
        # supervise loss
        x1 = sup_data['image_A']
        x2 = sup_data['image_B']
        target_A = sup_data['target_A'].long()
        target_B = sup_data['target_B'].long()
        x1, x2, change = self(x1, x2)
        target_bn = (target_B>0).float()
        loss_sup = self.loss_func(x1, x2, change, target_bn, target_A, target_B)
        
        A_ul, B_ul, target_uA, target_uB, _ = [v for v in unsup_data.values()]
        bs = A_ul.shape[0]
        # Get main prediction
        x_ul      = self.encoder(A_ul, B_ul)
        output_ul = self.decoder(x_ul)
        
        # Get auxiliary predictions
        outputs_ul = [self.decoder(x_ul, perturbation=pertub, o_l = output_ul[-1].detach()) for pertub in self.pertubs]
        targets = [F.softmax(o_ul.detach(), dim=1) for o_ul in output_ul]
        # Compute unsupervised loss
        loss_unsup = sum([0.5*(self.unsuper_loss(inputs=u[0], targets=targets[0], \
                        conf_mask=self.confidence_masking, 
                        threshold=self.confidence_th, 
                        use_softmax=False)+(self.unsuper_loss(inputs=u[1], targets=targets[1], \
                        conf_mask=self.confidence_masking, 
                        threshold=self.confidence_th, 
                        use_softmax=False)))
                        for u in outputs_ul])
        loss_unsup = (loss_unsup / len(outputs_ul))
        # Compute the unsupervised loss
        weight_u    = self.unsup_loss_w(self.global_step)
        loss_unsup  = loss_unsup * weight_u
        total_loss  = loss_unsup  + loss_sup 
        
        # update mertic
        sup_pred = self.get_prediction((x1, x2, change), domain='sup')
        unsup_pred = self.get_prediction(output_ul, domain='unsup')
        del output_ul, x_ul
        self.sup_running_metric.update(target_A.detach().cpu().numpy(), sup_pred['sup_pred_A'].detach().cpu().numpy())
        self.sup_running_metric.update(target_B.detach().cpu().numpy(), sup_pred['sup_pred_B'].detach().cpu().numpy())
        self.unsup_running_metric.update(target_uA.detach().cpu().numpy(), unsup_pred['unsup_pred_A'].detach().cpu().numpy())
        self.unsup_running_metric.update(target_uB.detach().cpu().numpy(), unsup_pred['unsup_pred_B'].detach().cpu().numpy())
        
        # log loss
        self.log('train/loss_sup', loss_sup, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True, batch_size=bs)
        self.log('train/loss_unsup', loss_unsup, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        
        return total_loss
        
    def on_train_epoch_end(self) -> None:
        # log metric for sup/unsup data
        sup_score = self.sup_running_metric.get_scores()
        unsup_score = self.unsup_running_metric.get_scores()

        for k, v in sup_score.items():
            self.log(f'train/sup_{k}', v, on_step=False, on_epoch=True, logger=True)
        for k, v in unsup_score.items():
            self.log(f'train/unsup_{k}', v, on_step=False, on_epoch=True, logger=True)
        
        # logging in file
        train_logger = logging.getLogger('train')
        message = '[Training SemiSCD (epoch %d summary)]: Fscd_sup=%.5f,  Fscd_unsup=%.5f \n' %\
                      (self.current_epoch, sup_score['F_scd'], unsup_score['F_scd'])
        for k, v in sup_score.items():
            message += 'sup_{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        for k, v in unsup_score.items():
            message += 'unsup_{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        train_logger.info(message)
        # reset mertric for next epoch
        self.sup_running_metric.reset()
        self.unsup_running_metric.reset()
    
class SemiMTSCD(SemiSCDNet):
    def __init__(self, use_ema=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ema = use_ema
        if self.use_ema:
            self.decoder_ema = LitEma(self.decoder)
            print(f"Keeping EMAs of {len(list(self.decoder_ema.buffers()))}.")
    
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.decoder_ema.store(self.decoder.parameters())
            self.decoder_ema.copy_to(self.decoder)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.decoder_ema.restore(self.decoder.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
                    
    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        if self.use_ema:
            self.decoder_ema(self.decoder)
            
            
        # 半监督学习的step
    def training_step(self, batch, batch_idx):
        sup_data, unsup_data = batch
        # supervise loss
        x1 = sup_data['image_A']
        x2 = sup_data['image_B']
        target_A = sup_data['target_A'].long()
        target_B = sup_data['target_B'].long()
        x1, x2, change = self(x1, x2)
        target_bn = (target_B>0).float()
        loss_sup = self.loss_func(x1, x2, change, target_bn, target_A, target_B)
        
        A_ul, B_ul, target_uA, target_uB, _ = [v for v in unsup_data.values()]
        bs = A_ul.shape[0]
        # Get main prediction
        x_ul = self.encoder(A_ul, B_ul)
        with self.ema_scope():
            output_ul = self.decoder(x_ul)
        
        # Get auxiliary predictions
        outputs_ul = [self.decoder(x_ul, perturbation=pertub, o_l = output_ul[-1].detach()) for pertub in self.pertubs]
        targets = [F.softmax(o_ul.detach(), dim=1) for o_ul in output_ul]
        # Compute unsupervised loss
        loss_unsup = sum([0.5*(self.unsuper_loss(inputs=u[0], targets=targets[0], \
                        conf_mask=self.confidence_masking, 
                        threshold=self.confidence_th, 
                        use_softmax=False)+(self.unsuper_loss(inputs=u[1], targets=targets[1], \
                        conf_mask=self.confidence_masking, 
                        threshold=self.confidence_th, 
                        use_softmax=False)))
                        for u in outputs_ul])
        loss_unsup = (loss_unsup / len(outputs_ul))
        # Compute the unsupervised loss
        weight_u    = self.unsup_loss_w(self.global_step)
        loss_unsup  = loss_unsup * weight_u
        total_loss  = loss_unsup  + loss_sup 
        
        # update mertic
        sup_pred = self.get_prediction((x1, x2, change), domain='sup')
        unsup_pred = self.get_prediction(output_ul, domain='unsup')
        del output_ul, x_ul
        self.sup_running_metric.update(target_A.detach().cpu().numpy(), sup_pred['sup_pred_A'].detach().cpu().numpy())
        self.sup_running_metric.update(target_B.detach().cpu().numpy(), sup_pred['sup_pred_B'].detach().cpu().numpy())
        self.unsup_running_metric.update(target_uA.detach().cpu().numpy(), unsup_pred['unsup_pred_A'].detach().cpu().numpy())
        self.unsup_running_metric.update(target_uB.detach().cpu().numpy(), unsup_pred['unsup_pred_B'].detach().cpu().numpy())
        
        # log loss
        self.log('train/loss_sup', loss_sup, on_step=True, on_epoch=True,  logger=True, batch_size=bs)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True, batch_size=bs)
        self.log('train/loss_unsup', loss_unsup, on_step=True, on_epoch=True, logger=True, batch_size=bs)
        
        return total_loss
    
    
# class SemiMTDiffSCD(SemiMTSCD):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         mid_dim = kwargs['mid_dim'] if hasattr(kwargs, 'mid_dim') else 128
        
#         # detect change from diff image
#         self.diff_decoder = nn.Sequential(
#             self._make_layer(ResBlock, 512, mid_dim, 3, stride=1),
#             nn.Conv2d(mid_dim, 64, kernel_size=1), 
#             nn.BatchNorm2d(64), 
#             nn.ReLU(), 
#             nn.Conv2d(64, 1, kernel_size=1)
#         )
#         if self.use_ema:
#             self.diff_decoder_ema = LitEma(self.diff_decoder)
#             print(f"Keeping EMAs of {len(list(self.decoder_ema.buffers()))}.")
    
        
#     @contextmanager
#     def ema_diff_scope(self, context=None):
#         if self.use_ema:
#             self.diff_decoder_ema.store(self.diff_decoder.parameters())
#             self.diff_decoder_ema.copy_to(self.diff_decoder)
#             if context is not None:
#                 print(f"{context}: Switched to EMA weights")
#         try:
#             yield None
#         finally:
#             if self.use_ema:
#                 self.diff_decoder_ema.restore(self.diff_decoder.parameters())
#                 if context is not None:
#                     print(f"{context}: Restored training weights")
                    
#     def on_train_batch_end(self, *args, **kwargs):
#         super().on_train_batch_end(*args, **kwargs)
#         if self.use_ema:
#             self.diff_decoder_ema(self.diff_decoder)
            
        
#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 conv1x1(inplanes, planes, stride),
#                 nn.BatchNorm2d(planes) )
        
#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
            
#         return nn.Sequential(*layers)
#     # 半监督学习的step
#     def training_step(self, batch, batch_idx):
#         sup_data, unsup_data = batch
#         # supervise loss
#         x1 = sup_data['image_A']
#         x2 = sup_data['image_B']
#         diff = torch.abs(x1-x2)
#         target_A = sup_data['target_A'].long()
#         target_B = sup_data['target_B'].long()
#         x1, x2, change = self(x1, x2)
#         target_bn = (target_B>0).float()
#         loss_sup = self.loss_func(x1, x2, change, target_bn, target_A, target_B)
        
#         diff_l = self.encoder.base_forward(diff)
#         with self.ema_diff_scope():
#             diff_l = self.diff_decoder(diff_l)
#         if diff_l.shape[2] != target_bn.shape[2]:
#             diff_l = F.interpolate(diff_l, size=target_bn.shape[2], mode='bilinear', align_corners=True)
#         # diff branch sup loss
#         loss_sup += weighted_BCE_logits(diff_l, target_bn)
        
#         # for unlabeled data
#         A_ul, B_ul, target_uA, target_uB, _ = [v for v in unsup_data.values()]
#         bs = A_ul.shape[0]
#         # Get main prediction
#         x_ul = self.encoder(A_ul, B_ul)
#         diff_ux =  torch.abs(A_ul-B_ul)
#         diff_ul = self.encoder.base_forward(diff_ux)    
#         with self.ema_scope():
#             output_ul = self.decoder(x_ul)
#         with self.ema_diff_scope():
#             diff_ul = self.diff_decoder(diff_ul)
#         if diff_ul.shape[2] != target_uA.shape[2]:
#             diff_ul = F.interpolate(diff_ul, size=target_uA.shape[2], mode='bilinear', align_corners=True)
#         # Get auxiliary predictions
#         outputs_ul = [self.decoder(x_ul, perturbation=pertub, o_l = output_ul[-1].detach()) for pertub in self.pertubs]
#         targets = [F.softmax(o_ul.detach(), dim=1) for o_ul in output_ul]
#         # Compute unsupervised loss
#         loss_unsup = sum([0.5*(self.unsuper_loss(inputs=u[0], 
#                         targets=targets[0], \
#                         conf_mask=self.confidence_masking, 
#                         threshold=self.confidence_th, 
#                         use_softmax=False)+(self.unsuper_loss(inputs=u[1], targets=targets[1], \
#                         conf_mask=self.confidence_masking, 
#                         threshold=self.confidence_th, 
#                         use_softmax=False))) + 0.01*self.unsuper_loss(inputs=u[-1], targets=diff_ul.detach(),
#                         conf_mask=self.confidence_masking, # compute diff loss
#                         threshold=self.confidence_th, 
#                         use_softmax=False)
#                         for u in outputs_ul])
#         loss_unsup = (loss_unsup / len(outputs_ul))
#         # Compute the unsupervised loss
#         weight_u    = self.unsup_loss_w(self.global_step)
#         loss_unsup  = loss_unsup * weight_u
#         total_loss  = loss_unsup  + loss_sup 
        
#         # update mertic
#         sup_pred = self.get_prediction((x1, x2, change), domain='sup')
#         unsup_pred = self.get_prediction(output_ul, domain='unsup')
#         del output_ul, x_ul
#         self.sup_running_metric.update(target_A.detach().cpu().numpy(), sup_pred['sup_pred_A'].detach().cpu().numpy())
#         self.sup_running_metric.update(target_B.detach().cpu().numpy(), sup_pred['sup_pred_B'].detach().cpu().numpy())
#         self.unsup_running_metric.update(target_uA.detach().cpu().numpy(), unsup_pred['unsup_pred_A'].detach().cpu().numpy())
#         self.unsup_running_metric.update(target_uB.detach().cpu().numpy(), unsup_pred['unsup_pred_B'].detach().cpu().numpy())
        
#         # log loss
#         self.log('train/loss_sup', loss_sup, on_step=True, on_epoch=True,  logger=True, batch_size=bs)
#         self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True, batch_size=bs)
#         self.log('train/loss_unsup', loss_unsup, on_step=True, on_epoch=True, logger=True, batch_size=bs)
        
#         return total_loss