from contextlib import contextmanager
import logging

from src.modules.ema import LitEma
from utils.metrics import weighted_BCE_logits
from .components import *
from .networks import SCDNet

   
class RCR(SCDNet):
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
        targets = torch.sigmoid(output_ul[-1]).detach()>0.5
        # Compute unsupervised loss
        loss_unsup = sum([self.unsuper_loss(inputs=u[-1], targets=targets)
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