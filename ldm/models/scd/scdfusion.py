
import logging
import os
import PIL
import torch
from ldm.data.SCDDataloader import Index2Color

from ldm.models.diffusion.ddpm_scd import LatentDiffusionSCD
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR

class SCDfusion(LatentDiffusionSCD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.loss_func = SCDLoss()
    
    # disable training of denoise & encoder
    def instantiate_first_stage(self, config):
        # disable training of encoder
        self.first_stage_model = instantiate_from_config(config)
        # self.first_stage_model.encoder.eval()
        # self.first_stage_model.encoder.train = lambda *args:None
        # for param in self.first_stage_model.encoder.parameters():
        #     param.requires_grad = False
        # disable training of denoise network
        self.model.eval()
        self.model.train = lambda *args:None
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def diffusion_step(self, x, c, noise=None):
        t = torch.randint(0, 1, (x.shape[0],), device=self.device).long()
        # 随机生成噪声
        noise = default(noise, lambda: torch.randn_like(x))
        # 扩散破坏原始图像
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        # 使用去噪模型预测噪声/原始图像
        model_output = self.apply_model(x_noisy, t, c)
        return model_output
    
    def scdforword(self, batch, batch_idx):
        x, c = self.get_input(batch, self.first_stage_key)
        x = self.diffusion_step(x, c)
        x = x.chunk(2, dim=1)
        x1, x2, change = self.first_stage_model.decode(x)
        return x1, x2, change
    
    def training_step(self, batch, batch_idx):
        # forward
        target_A = batch['target_A'].long()
        target_B = batch['target_B'].long()
        bs = target_B.size(0)
        x = self.scdforword(batch, batch_idx)
        target_bn = (target_B>0).float()
        
        loss = self.first_stage_model.loss_func(*x, target_bn, target_A, target_B)
        self.log('train\loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log('train\lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True, batch_size=bs)
        
        sup_pred = self.first_stage_model.get_prediction(x, domain='sup')
        self.first_stage_model.train_running_meters.update(target_A.detach().cpu().numpy(), sup_pred['sup_pred_A'].detach().cpu().numpy())
        self.first_stage_model.train_running_meters.update(target_B.detach().cpu().numpy(), sup_pred['sup_pred_B'].detach().cpu().numpy())
        return loss
        
    def on_train_epoch_end(self) -> None:
        # log metric for sup/unsup data
        sup_score = self.first_stage_model.train_running_meters.get_scores()

        for k, v in sup_score.items():
            self.log(f'train\sup_{k}', v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # logging in file
        train_logger = logging.getLogger('train')
        message = '[Training SCD (epoch %d summary)]: Fscd=%.5f \n' %\
                      (self.current_epoch, sup_score['F_scd'])
        for k, v in sup_score.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        train_logger.info(message)
        # reset mertric for next epoch
        self.first_stage_model.running_meters.reset()
        
    def configure_optimizers(self):
        opt = torch.optim.SGD(list(self.first_stage_model.decoder.parameters()),
                                  lr=self.learning_rate, weight_decay=1e-4, momentum=0.9)
        if self.first_stage_model.scheduler_config is not None:
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
    
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        target_A = batch['target_A']
        target_B = batch['target_B']
        x1, x2, change = self.scdforword(batch, batch_idx)
        preds = dict()
        preds.update(self.first_stage_model.get_prediction((x1, x2, change), domain='val'))
        
        bs = x1.size(0)
        self.first_stage_model.running_meters.update(target_A.detach().cpu().numpy(), preds['val_pred_A'].detach().cpu().numpy())
        self.first_stage_model.running_meters.update(target_B.detach().cpu().numpy(), preds['val_pred_B'].detach().cpu().numpy())
        scores = self.first_stage_model.running_meters.get_scores()
        self.log('val\F_scd', scores['F_scd'], on_epoch=True,  logger=True, batch_size=bs)
        self.log('val\miou', scores['Mean_IoU'], on_epoch=True, logger=True, batch_size=bs)
        self.log('val\OA', scores['OA'], on_epoch=True,  logger=True, batch_size=bs)
        self.log('val\SeK', scores['Sek'], on_epoch=True, logger=True, batch_size=bs)
        
        return preds  
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        # log the validation metrics
        val_logger = logging.getLogger('val')
        scores = self.first_stage_model.running_meters.get_scores()
        message  = f'Validation Summary Epoch: [{self.current_epoch}]\n'
        for k, v in scores.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        val_logger.info(message)
        self.first_stage_model.running_meters.reset()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        target_A = batch['target_A']
        target_B = batch['target_B']
        image_id = batch['image_id'][0]
        x1, x2, change = self.scdforword(batch, batch_idx)
        preds = dict()
        preds.update(self.get_prediction((x1, x2, change), domain='test'))
        self.running_meters.update(target_A.detach().cpu().numpy(), preds['test_pred_A'].detach().cpu().numpy())
        self.running_meters.update(target_B.detach().cpu().numpy(), preds['test_pred_B'].detach().cpu().numpy())
           
        preds['test_pred_A'] = Index2Color(preds['test_pred_A'].detach().cpu().numpy().squeeze())
        preds['test_pred_B'] = Index2Color(preds['test_pred_B'].detach().cpu().numpy().squeeze())
        # save images
        save_dir = os.path.join(self.trainer.log_dir, os.pardir)
        save_dir = os.path.join(save_dir, 'test_results')
        
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        pred_A_L_show = PIL.Image.fromarray(preds['test_pred_A'].astype(np.uint8)).convert('RGB')
        pred_B_L_show = PIL.Image.fromarray(preds['test_pred_B'].astype(np.uint8)).convert('RGB')
        pred_A_L_show.save(os.path.join(save_dir, image_id+'_pred_A_L_show.png'))
        pred_B_L_show.save(os.path.join(save_dir, image_id+'_pred_B_L_show.png'))
        
        return preds