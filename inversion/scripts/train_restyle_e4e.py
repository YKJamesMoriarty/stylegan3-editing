import json
import pprint
import sys
from plistlib import Dict
from typing import Any

import dataclasses
import pyrallis
import torch

'''
该脚本的作用：
负责训练e4e模型，支持从头开始训练或从之前的检查点继续训练。
它允许通过渐进式解锁StyleGAN的风格层来实现逐步优化。
'''


'''添加搜索路径
将当前目录和上一级目录添加到Python模块搜索路径中，确保可以导入项目中的其他模块'''
sys.path.append(".")
sys.path.append("..")
'''导入项目模块
从inversion模块中导入训练选项e4eTrainOptions和训练控制器Coach。
    e4eTrainOptions：定义了训练e4e模型时所需的选项和参数。
    Coach：负责实际的训练过程。'''
from inversion.options.e4e_train_options import e4eTrainOptions
from inversion.training.coach_restyle_e4e import Coach


@pyrallis.wrap()#@pyrallis.wrap()将命令行参数解析为e4eTrainOptions类型的对象 opts
def main(opts: e4eTrainOptions):
	previous_train_ckpt = None
	#首先检查是否有之前的检查点
	if opts.resume_training_from_ckpt:
		#如果有，则恢复之前的训练
		opts, previous_train_ckpt = load_train_checkpoint(opts)
	else:
		#如果没有，则设置渐进式训练步骤，并创建实验的初始目录
		setup_progressive_steps(opts)
		create_initial_experiment_dir(opts)
	#初始化Coach对象，传入训练选项和检查点，然后调用train方法启动训练过程。
	coach = Coach(opts, previous_train_ckpt)
	coach.train()

'''作用：从指定的检查点恢复训练。'''
def load_train_checkpoint(opts: e4eTrainOptions):
	train_ckpt_path = opts.resume_training_from_ckpt
	previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location='cpu')
	new_opts_dict = dataclasses.asdict(opts)
	opts = previous_train_ckpt['opts']
	opts['resume_training_from_ckpt'] = train_ckpt_path
	update_new_configs(opts, new_opts_dict)
	pprint.pprint(opts)
	opts = e4eTrainOptions(**opts)
	if opts.sub_exp_dir is not None:
		sub_exp_dir = opts.sub_exp_dir
		opts.exp_dir = opts.exp_dir / sub_exp_dir
	create_initial_experiment_dir(opts)
	return opts, previous_train_ckpt

'''作用：为渐进式训练设置步骤。
渐进式训练是通过逐步解锁StyleGAN的风格层（style layers）进行的。这段代码根据用户提供的选项
progressive_start和progressive_step_every来生成一个包含风格层解锁时间的进度表。'''
def setup_progressive_steps(opts: e4eTrainOptions):
	num_style_layers = 16
	num_deltas = num_style_layers - 1
	if opts.progressive_start is not None:  # If progressive delta training
		opts.progressive_steps = [0]
		next_progressive_step = opts.progressive_start
		for i in range(num_deltas):
			opts.progressive_steps.append(next_progressive_step)
			next_progressive_step += opts.progressive_step_every

	assert opts.progressive_steps is None or is_valid_progressive_steps(opts, num_style_layers), \
		"Invalid progressive training input"

'''作用：检查训练的渐进步骤是否合法。确保总的训练步骤数与风格层的数量一致，并且第一个步骤从第0层开始。'''
def is_valid_progressive_steps(opts: e4eTrainOptions, num_style_layers: int):
	return len(opts.progressive_steps) == num_style_layers and opts.progressive_steps[0] == 0

'''作用：创建一个实验目录来存放训练结果，并将当前训练选项保存为opt.json文件，
便于以后检查训练设置。'''
def create_initial_experiment_dir(opts: e4eTrainOptions):
	opts.exp_dir.mkdir(exist_ok=True, parents=True)
	opts_dict = dataclasses.asdict(opts)
	pprint.pprint(opts_dict)
	with open(opts.exp_dir / 'opt.json', 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True, default=str)

'''作用：将新选项（new_opts）更新到之前的检查点选项（ckpt_opts）中。这样做可以确保用户的最新配置
被应用到检查点的恢复训练过程中。'''
def update_new_configs(ckpt_opts: Dict[str, Any], new_opts: Dict[str, Any]):
	for k, v in new_opts.items():
		ckpt_opts[k] = v


if __name__ == '__main__':
	main()
