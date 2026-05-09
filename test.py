import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch

from torch.autograd import Variable
from torch.nn import functional as F
import time
import os

import cv2
import daf

import argparse


def make_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def test(model, args):
    test_root = args.data_root
    if args.test_lst is not None:
        with open(osp.join(test_root, args.test_lst), 'r') as f:
            test_lst = f.readlines()
        test_lst = [x.strip() for x in test_lst]
        if ' ' in test_lst[0]:
            test_lst = [x.split(' ')[0] for x in test_lst]
    else:
        test_lst = os.listdir(test_root)
    # print(test_lst[0])
    count=len(test_lst)-1
    tem=1
    save_sideouts = 0
    if save_sideouts:
        save_dir = args.res_dir
        k = 1
        for j in range(11):
            make_dir(os.path.join(save_dir, 's2d_' + str(k)))
            make_dir(os.path.join(save_dir, 'd2s_' + str(k)))
            k += 1
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    save_dir = args.res_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.cuda:
        model.cuda()
    model.eval()
    start_time = time.time()
    all_t = 0

    for nm in test_lst:
        if ".db"  in nm :
            continue
        print(f"{tem}/{count}")
        tem += 1
        print(test_root + '/' + nm)
        data = cv2.imread(test_root + '/' + nm)
        data = np.array(data, np.float32)
        data -= mean_bgr

        data = data.transpose((2, 0, 1))


        data = torch.from_numpy(data).float().unsqueeze(0)
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        t1 = time.time()

        out = model(data)

        if '/' in nm:
            nm = nm.split('/')[-1]
        if save_sideouts:
            out = [F.sigmoid(x).cpu().data.numpy()[0, 0, :, :] for x in out]
            # print(out[1])
            print(f"len(out){len(out)}")
            k = 1

        else:
            out = [F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]

        if not os.path.exists(os.path.join(save_dir, 'fuse')):
            os.mkdir(os.path.join(save_dir, 'fuse'))
        cv2.imwrite(os.path.join(save_dir, 'fuse/%s.png' % nm.split('/')[-1].split('.')[0]), 255 * out[-1])
        all_t += time.time() - t1
    print(all_t)
    print('Overall Time use: ', time.time() - start_time)


def main():
    import time
    print(time.localtime())
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = daf.daf()
    model.load_state_dict(torch.load('%s' % (args.model)))
    test(model, args)

path= "../../../param_all/daf_15000.pth"





def parse_args():
    parser = argparse.ArgumentParser('test daf')
    parser.add_argument('-c', '--cuda', action='store_true', default=True,
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='1',
                        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default=path,
                        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='./my_bsds500Test1',
                        help='the dir to store result')
    parser.add_argument('--data-root', type=str, default='./images/test', help='the data dir to test')
    parser.add_argument('--test-lst', type=str, default=None)
    return parser.parse_args()





if __name__ == '__main__':
    main()

