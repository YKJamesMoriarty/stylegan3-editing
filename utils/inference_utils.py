from pathlib import Path
from typing import Optional

import dataclasses
import torch
from torchvision import transforms

from configs.paths_config import model_paths
from inversion.models.e4e3 import e4e
from inversion.models.psp3 import pSp
from inversion.options.e4e_train_options import e4eTrainOptions
from inversion.options.test_options import TestOptions
from inversion.options.train_options import TrainOptions
from models.stylegan3.model import SG3Generator
from utils.model_utils import ENCODER_TYPES

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

FULL_IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

'''作用：
加载编码器模型（如 pSp 或 e4e），并根据提供的检查点路径进行初始化。
如果提供了 StyleGAN3 生成器路径，还会替换模型中的生成器。
返回模型和其配置'''
def load_encoder(checkpoint_path: Path, test_opts: Optional[TestOptions] = None, generator_path: Optional[Path] = None):
    #加载模型检查点：torch.load() 用于加载 .pt 或 .pth 模型文件。
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    #opts 包含训练时的超参数和配置。
    opts = ckpt['opts']
    #opts["checkpoint_path"] 将模型的路径保存在选项中，方便后续使用
    opts["checkpoint_path"] = checkpoint_path
    #修正 StyleGAN3 权重路径：项目中可能有不同格式的权重文件，这段代码根据不同情况调整路径，以确保加载正确的权重
    if opts['stylegan_weights'] == Path(model_paths["stylegan3_ffhq"]):
        opts['stylegan_weights'] = Path(model_paths["stylegan3_ffhq_pt"])
    if opts['stylegan_weights'] == Path(model_paths["stylegan3_ffhq_unaligned"]):
        opts['stylegan_weights'] = Path(model_paths["stylegan3_ffhq_unaligned_pt"])
    '''
    选择编码器类型：根据 opts["encoder_type"]，判断使用 pSp 还是 e4e 编码器。
    TrainOptions 和 e4eTrainOptions 是分别用于 pSp 和 e4e 的训练配置。
    pSp 和 e4e 是两种不同的风格编码器网络。
    test_opts 包含测试时的额外选项，如果有提供，会更新训练选项。
    '''
    if opts["encoder_type"] in ENCODER_TYPES['pSp']:
        opts = TrainOptions(**opts)
        if test_opts is not None:
            opts.update(dataclasses.asdict(test_opts))
        net = pSp(opts)
    else:
        opts = e4eTrainOptions(**opts)
        if test_opts is not None:
            opts.update(dataclasses.asdict(test_opts))
        net = e4e(opts)
    #更新 StyleGAN3 生成器：如果提供了生成器路径，则加载并替换模型的解码器部分。
    print('Model successfully loaded!')
    if generator_path is not None:
        print(f"Updating SG3 generator with generator from path: {generator_path}")
        net.decoder = SG3Generator(checkpoint_path=generator_path).decoder

    net.eval()#设置为评估模式：net.eval() 表示模型处于推理模式，不进行梯度更新。
    net.cuda()#将模型加载到 GPU：net.cuda() 将模型转移到 GPU 上。
    return net, opts

'''作用：用于获取模型中的平均潜变量生成的图像，通常作为参考图像，供后续图像编辑使用。'''
def get_average_image(net):
    '''获取平均潜变量：net.latent_avg 是模型中的平均潜变量，它代表了潜变量空间中一个典型
    的点。latent_avg.repeat(16, 1) 将平均潜变量重复 16 次，形成一个批次的输入张量。'''
    '''生成平均图像：将潜变量输入模型生成对应的平均图像，input_code=True 表示输入的是
    潜变量而不是图像，return_latents=False 只返回图像而不返回潜变量。'''
    avg_image = net(net.latent_avg.repeat(16, 1).unsqueeze(0).cuda(),
                    input_code=True,
                    return_latents=False)[0]
    
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image

'''这个函数是核心，处理图像反演和生成图像'''
def run_on_batch(inputs: torch.tensor, net, opts: TrainOptions, avg_image: torch.tensor,
                 landmarks_transform: Optional[torch.tensor] = None):
    '''初始化结果存储：results_batch 和 results_latent 是两个字典，
    用于保存每个样本在每次迭代中的图像和潜变量。'''
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    y_hat, latent = None, None
    #检查选项：如果 opts 中没有 resize_outputs 选项，则默认设置为 False。
    if "resize_outputs" not in dataclasses.asdict(opts):
        opts.resize_outputs = False
    '''逐步编辑迭代：循环 opts.n_iters_per_batch 次，代表对每个批次图像进行多次迭代编辑'''
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            #第一次迭代：将输入图像和平均图像拼接成输入，传给网络。
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            #后续迭代：将输入图像和上一轮网络的输出 图像进行y_hat 拼接，继续输入模型。
            '''
             inputs 和 y_hat 沿着通道维度拼接（dim=1），拼接后的 x_input 形状为 (batch_size
             , 2 * channels, height, width)。拼接的结果包含了当前输入图像和上一轮的输出图像的特征。
            '''
            x_input = torch.cat([inputs, y_hat], dim=1)
          
        #判断是否是最后一次迭代：如果是最后一次迭代，编辑完成后不会进行进一步迭代。
        is_last_iteration = iter == opts.n_iters_per_batch - 1
        #网络前向传播：将输入图像和潜变量输入到网络中
        '''landmarks_transform：可选的关键点变换，用于处理人脸对齐
        aligned_and_unaligned=True：返回对齐和未对齐的输出图像，这在人脸编辑任务中特别有用
        return_latents=True：除了生成的图像，还返回编辑后的潜变量，用于追踪每次迭代的潜变量变化
        resize=opts.resize_outputs：是否调整输出图像的尺寸，基于用户的设置。
        '''
        res = net.forward(x_input,
                          latent=latent,
                          landmarks_transform=landmarks_transform,
                          return_aligned_and_unaligned=True,
                          return_latents=True,
                          resize=opts.resize_outputs)

        '''处理无 landmark 变换的情况：如果没有关键点变换，只返回对齐后的输出
        图像 y_hat 和对应的潜变量 latent'''
        if landmarks_transform is None:
            y_hat, latent = res
        #处理 landmark 变换的情况(如果有关键点变换)
        #在最后一次迭代时，返回未对齐的最终输出。
        #其他迭代中，返回对齐的中间输出。
        else:
            # note: res = images, unaligned_images, codes
            if is_last_iteration:
                _, y_hat, latent = res
            else:
                y_hat, _, latent = res

        '''存储中间结果：将每次迭代的图像 y_hat 和潜变量 latent 存储在 
        results_batch 和 results_latent 中，便于后续查看每次编辑的效果。'''
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            results_latent[idx].append(latent[idx].cpu().numpy())

        # 在输入下一次迭代前，将输出图像进行调整，确保 y_hat 的通道数在合理范围内。
        y_hat = net.face_pool(y_hat)
    #返回最终结果：函数返回一个字典，包含每个样本在每次迭代中的图像结果和潜变量变化
    return results_batch, results_latent
