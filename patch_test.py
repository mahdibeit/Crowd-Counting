################################################################################
#patching

import torch
import os
import numpy as np
from datasets.crowd_sh import Crowd
from models.vgg import vgg19
import argparse
import torch, torchvision.models

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data_dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save_dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []
    epoch_minus1 = []
    filename = []
    gt = []
    pred = []
    temp_minus = 0
    tmp = 0
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        # print(name[0])
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            filename.append(name[0])
            gt.append(count[0].item())
            pred.append(torch.sum(outputs).item())
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)

    for i in range(int(len(epoch_minus)/4)):
      sum_ = epoch_minus[4*i] + epoch_minus[4*i+1] + epoch_minus[4*i+2] + epoch_minus[4*i+3]
      epoch_minus1.append(sum_)

    print('epoch_minus1: ',epoch_minus1)
    print('filename: ',filename)
    # print(len(filename))
    print('epoch_minus: ',epoch_minus)
    # print(len(epoch_minus))
    print('gt: ',gt)
    print('pred: ',pred)

    epoch_minus = np.array(epoch_minus1)
    mse = np.sqrt(np.mean(np.square(epoch_minus1)))
    mae = np.mean(np.abs(epoch_minus1))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
