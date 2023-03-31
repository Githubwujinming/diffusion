import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import DDPM, LatentDiffusion

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

class LatentDiffusionSCD(LatentDiffusion):
    '''
    first_stage_key: ['image_A','image_B']
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # first_stage_key ['image_A','image_B']
        # TODO: 参数尺寸需要检查
        self.learnable_cond = torch.nn.Parameter(torch.randn(1, 128, 32, 32), requires_grad=True)
        
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch 每个epoch的第一个batch的处理函数
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            z, _ = self.get_input(batch, self.first_stage_key)
            encoder_posterior = torch.cat(z, dim=1)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")
            
    # 解码器对去噪网络预测的特征解码，生成新图片，samples为扩散模型采样的数据
    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        # for zd in tqdm(samples, desc=desc):
        #     denoise_row.append(self.decode_first_stage(zd.to(self.device),
        #                                                     force_not_quantize=force_no_decoder_quantization))
        # n_imgs_per_row = len(denoise_row)
        # denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        # denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        # denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        # denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        # return denoise_grid
        raise NotImplementedError
    
    '''
    从数据中返回编码后的输入和条件（提示词，位置编码等）
    '''
    @torch.no_grad()
    def get_input(self, batch, ks, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        def get_input(batch, k):
            x = batch[k]
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.to(memory_format=torch.contiguous_format).float()
            return x
        x1 = get_input(batch, ks[0])# 根据key获取输入，batch是一个字典
        x2 = get_input(batch, ks[1])
        if bs is not None:
            x1, x2 = x1[:bs], x2[:bs]
        # x1 = x1.to(self.device)
        # x2 = x2.to(self.device)
        # 对输入（图像）编码, z为编码后的结果
        encoder_posterior = self.encode_first_stage(x1, x2)
        z = torch.cat(encoder_posterior, dim=1)
        z = self.get_first_stage_encoding(z).detach()
        # 从数据中获取条件，这个应该是和图像文本对的数据相关的，自定义的数据可以修改这里
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key in ['diff', 'diff_abs']:
                xc = torch.abs(x1, x2)# 使用差分图像作为条件
            else:
                xc = self.learnable_cond
            # 如果条件编码器不可训练，或者强制编码，就设置为编码后的结果，否则输入条件数据
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            # if self.use_positional_encodings:
            #     pos_x, pos_y = self.compute_latent_shifts(batch)
            #     ckey = __conditioning_keys__[self.model.conditioning_key]
            #     c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            # 如果是无条件生成，c可以为位置编码或者空
            c = None
            xc = None
            # # 如果使用位置编码，计算位置编码
            # if self.use_positional_encodings:
            #     pos_x, pos_y = self.compute_latent_shifts(batch)
            #     c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]# z是隐变量，c是条件编码
        return out# 
    # 与encoder_first_stage流程一致，对编码后的隐变量解码。
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        # 变化检测中predict_cids设置为False
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        # 变化检测时split_input_params不设置
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                # 变化检测时，直接调用first_stage_model的decode方法
                return self.first_stage_model.decode(z)
    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)
    @torch.no_grad()
    def encode_first_stage(self, x1, x2):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x1.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                # 将输入展开
                z1 = unfold(x1)  # (bn, nc * prod(**ks), L)
                z2 = unfold(x2)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z1 = z1.view((z1.shape[0], -1, ks[0], ks[1], z1.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
                # 每个小块分别编码
                z2 = z2.view((z2.shape[0], -1, ks[0], ks[1], z2.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
                
                output_list = [self.first_stage_model.encode(z1[:, :, :, :, i], z2[:, :, :, :, i])
                               for i in range(z1.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                # 将小块合并。
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x1, x2)
        else:
            return self.first_stage_model.encode(x1, x2)