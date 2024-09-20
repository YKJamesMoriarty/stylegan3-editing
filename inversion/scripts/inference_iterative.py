import sys
import time

import numpy as np
import pyrallis #用于管理命令行参数配置
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
'''加载自定义模块'''
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from inversion.options.test_options import TestOptions
from inversion.datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im
from utils.inference_utils import get_average_image, run_on_batch, load_encoder


@pyrallis.wrap()#允许从命令行传递参数，这些参数会被解析为 TestOptions 对象
def run(test_opts: TestOptions):
    '''1.创建输出目录
    创建两个输出目录：一个用于保存每一步编辑结果（inference_results），
    另一个用于保存逐步拼接后的图像（inference_coupled）。'''
    out_path_results = test_opts.output_path / 'inference_results'
    out_path_coupled = test_opts.output_path / 'inference_coupled'
    out_path_results.mkdir(exist_ok=True, parents=True)
    out_path_coupled.mkdir(exist_ok=True, parents=True)

    '''2.加载预训练模型和数据集'''
    #load_encoder 加载预训练的图像编码器及其训练时使用的参数.
    net, opts = load_encoder(checkpoint_path=test_opts.checkpoint_path, test_opts=test_opts)
    #加载数据集并应用相应的图像变换，构造数据加载器 dataloader。
    print(f'Loading dataset for {opts.dataset_type}')
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    #InferenceDataset 会根据给定的路径data_path加载输入图像并进行transform变换
    dataset = InferenceDataset(root=opts.data_path,
                               landmarks_transforms_path=opts.landmarks_transforms_path,
                               transform=transforms_dict['transform_inference'])
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    # get_average_image 获取与网络相关的“平均图像”，在某些生成模型中用作参考图像
    avg_image = get_average_image(net)
    #resize_amount 确定输出图像的尺寸，取决于用户是否要求调整输出大小。
    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
    '''3.图像编辑的主循环'''
    global_i = 0
    global_time = []
    all_latents = {}
    #使用 tqdm 显示进度条，遍历数据加载器中的每个批次图像。
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        
        with torch.no_grad():
            input_batch, landmarks_transform = input_batch
            tic = time.time()
            '''run_on_batch 函数对每个图像批次应用编辑操作，生成编辑后的
            结果（result_batch）以及相应的潜变量（result_latents）
            '''
            result_batch, result_latents = run_on_batch(inputs=input_batch.cuda().float(),
                                                        net=net,
                                                        opts=opts,
                                                        avg_image=avg_image,
                                                        landmarks_transform=landmarks_transform.cuda().float())
            toc = time.time()
            #记录每个批次的处理时间以进行性能评估
            global_time.append(toc - tic)
        '''4.保存编辑结果'''
        #对每张图片进行处理，迭代生成编辑结果，保存每一步的编辑图像
        for i in range(input_batch.shape[0]):
            # tensor2im 将 PyTorch 张量转换为可视化的图像格式。
            results = [tensor2im(result_batch[i][iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
            im_path = dataset.paths[global_i]

            # save individual step results
            #将每一步的结果保存到 inference_results 目录
            for idx, result in enumerate(results):
                save_dir = out_path_results / str(idx)
                save_dir.mkdir(exist_ok=True, parents=True)
                result.resize(resize_amount).save(save_dir / im_path.name)

            # save step-by-step results side-by-side
            # 拼接后的图像保存到 inference_coupled 目录。
            input_im = tensor2im(input_batch[i])
            res = np.array(results[0].resize(resize_amount))
            for idx, result in enumerate(results[1:]):
                res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
            res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)
            Image.fromarray(res).save(out_path_coupled / im_path.name)

            # store all latents with dict pairs (image_name, latents)
            all_latents[im_path.name] = result_latents[i]

            global_i += 1
    #计算平均运行时间和标准差，输出到 stats.txt 文件
    stats_path = opts.output_path / 'stats.txt'
    result_str = f'Runtime {np.mean(global_time):.4f}+-{np.std(global_time):.4f}'
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)

    # save all latents as npy file
    np.save(test_opts.output_path / 'latents.npy', all_latents)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    run()
