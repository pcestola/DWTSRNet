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
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif 'test' in phase:
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
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        #logger.info(model.info_params())

    
    # 4) Main training
    for epoch in range(100):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for train_data in train_loader:

            current_step += 1

            # Feed patch pairs
            model.feed_data(train_data)

            # Optimize parameters
            model.optimize_parameters(current_step)

            # update learning rate
            model.update_learning_rate()

            # Training information
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

        # Testing
        if (epoch+1) % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
            
            for test_loader in test_loaders:
                avg_psnr = 0.0
                avg_ssim = 0.0
                counter = 0

                for test_data in test_loader:
                    counter += 1

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

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
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.4f}dB, Average SSIM : {:<.4f}\n'.format(epoch, current_step, avg_psnr, avg_ssim))

        # Saving the model
        logger.info('Saving the model.')
        model.save(epoch)


if __name__ == '__main__':
    main()
    