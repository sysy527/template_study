import sys
import os

# src 디렉토리의 절대 경로를 계산하여 sys.path에 추가
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src_path)
from models.VGG16 import *

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
sys.path.append(data_path)
from dataset import *

import torch
import torch.nn as nn
import yaml

from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

class Train:
    def __init__(self, args, config_path=None):
        # YAML 파일에서 설정값 불러오기
        if config_path is None:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cfg/default.yml'))
            
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # YAML 값과 argparse 값 결합
        config.update(vars(args))
        for key, value in config.items():
            setattr(self, key, value)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # 프로젝트 최상위 디렉토리 설정
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        # 로그 및 체크포인트 디렉토리 설정
        self.dir_chck = os.path.join(self.base_dir, self.dir_checkpoint, self.scope, self.name_data)
        self.dir_log = os.path.join(self.base_dir, self.dir_log, self.scope)
        self.dir_data = os.path.join(self.base_dir, self.dir_data, self.name_data)


    def save(self, dir_chck, net, optim, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                    '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, net, optim=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            net.load_state_dict(dict_net['net'])
            optim.load_state_dict(dict_net['optim'])

            return net, optim, epoch

        elif mode == 'test':
            net.load_state_dict(dict_net['net'])

            return net, epoch

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr = self.lr

        batch_size = self.batch_size
        device = self.device

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ny_in = self.ny_in
        nx_in = self.nx_in
    

        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,)), transforms.Resize(32)])

        dataset_train = datasets.MNIST(root=os.path.join(self.base_dir, 'data'), train=True, download=True, transform=transform_train)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=0)

        num_train = len(loader_train.dataset)
        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        net = VGG16(nch_in, nch_out, nch_ker).to(device)

        ## setup loss & optimization
        fn = nn.CrossEntropyLoss().to(device)

        params = net.parameters()

        optim = torch.optim.Adam(params, lr=lr)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            net, optim, st_epoch = self.load(dir_chck, net, optim, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=self.dir_log)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            net.train()

            loss_train = []
            pred_train = []

            # for i, data in enumerate(loader_train, 1):
            for i, (input, label) in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input = input.to(device)
                label = label.to(device)
 
                output = net(input)
                pred = torch.softmax(output, dim=1).max(dim=1, keepdim=True)[1]

                # backward netD
                optim.zero_grad()

                loss = fn(output, label)
                loss.backward()
                optim.step()

                # get losses
                loss_train += [loss.item()]
                pred_train += [((pred == label.view_as(pred)).type(torch.float)).mean(dim=0).item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f ACC: %.4f' %
                      (epoch, i, num_batch_train, np.mean(loss_train), 100 * np.mean(pred_train)))

        
            writer_train.add_scalar('loss', np.mean(loss_train), epoch)

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(self.dir_chck, net, optim, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr = self.lr

        batch_size = self.batch_size
        device = self.device

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ny_in = self.ny_in
        nx_in = self.nx_in

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data)

        dir_data_test = os.path.join(self.dir_data, name_data, 'test')

        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,)), transforms.Resize(32)])

        dataset_test = datasets.MNIST(root='.', train=False, download=True, transform=transform_test)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        num_test = len(loader_test.dataset)
        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        net = VGG16(nch_in, nch_out, nch_ker).to(device)
        ## setup loss & optimization
        fn = nn.CrossEntropyLoss().to(device)

        ## load from checkpoints
        st_epoch = 0

        net, st_epoch = self.load(self.dir_chck, net, mode=mode)

        ## test phase
        with torch.no_grad():
            net.eval()

            loss_test = []
            pred_test = []

            # for i, data in enumerate(loader_test, 1):
            for i, (input, label) in enumerate(loader_test, 1):

                input = input.to(device)
                label = label.to(device)
                # input = data['input'].to(device)
                # label = data['label'].to(device)

                output = net(input)
                pred = torch.softmax(output, dim=1).max(dim=1, keepdim=True)[1]

                loss = fn(output, label)

                # get losses
                loss_test += [loss.item()]
                pred_test += [((pred == label.view_as(pred)).type(torch.float)).mean(dim=0).item()]

                print('TEST: BATCH %04d/%04d: LOSS: %.4f ACC: %.4f' % (i, num_batch_test, np.mean(loss_test), 100 * np.mean(pred_test)))

            print('TEST: AVERAGE LOSS: %.6f' % (np.mean(loss_test)))
            print('TEST: AVERAGE ACC: %.6f' % (100 * np.mean(pred_test)))
