import math
import torch
import random
import os.path
import logging
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main():

    # 1) Prepare options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # Distributed settings
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # Update opt
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

    # Save opt to '../option.json' file
    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    # Configure logger
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)

    # Seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2) Create dataloaders
    test_loaders = list()
    for phase, dataset_opt in opt['datasets'].items():
        if 'test' in phase:
            test_set = define_Dataset(dataset_opt)
            test_loaders.append(DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True))
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    # 3) Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = define_Model(opt)
    model.load()
    model.netG.eval()

    if opt['rank'] == 0:
        logger.info(model.info_network())
        #logger.info(model.info_params())

    # 4) Main Test
    for test_loader, name in test_loaders:
        avg_psnr = 0.0
        avg_ssim = 0.0
        counter = 0

        logger.info(f"Testset: {name}")

        for test_data in test_loader:
            counter += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join(opt['path']['images'], img_name)
            util.mkdir(img_dir)

            model.feed_data(test_data)
            model.netG_forward()

            visuals = model.current_visuals()
            E_img = util.tensor2uint(visuals['E'])
            H_img = util.tensor2uint(visuals['H'])

            
            # Save estimated image E
            save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
            util.imsave(E_img, save_img_path)

            # Calculate PSNR
            E_y = util.rgb2ycbcr(E_img)
            H_y = util.rgb2ycbcr(H_img)
            
            current_psnr = util.calculate_psnr(E_y, H_y, border=border)
            current_ssim = util.calculate_ssim(E_y, H_y, border=border)

            avg_psnr += current_psnr
            avg_ssim += current_ssim

        avg_psnr = avg_psnr / counter
        avg_ssim = avg_ssim / counter

        # Log
        logger.info('Average PSNR : {:<.4f}dB, Average SSIM : {:<.4f}\n'.format(avg_psnr, avg_ssim))


if __name__ == '__main__':
    main()
    