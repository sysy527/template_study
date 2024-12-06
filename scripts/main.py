import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

import argparse
import torch.backends.cudnn as cudnn
from utils.utils import Parser, Logger
from utils.train import Train
import yaml

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True


# YAML 파일 불러오기
config_path = os.path.join(os.path.dirname(__file__), '../cfg/default.yml')
with open(config_path, "r") as file:
    default_config = yaml.safe_load(file)

#***********************************************************************************************************
# ArgumentParser 객체 생성
parser = argparse.ArgumentParser(description='Train the MNIST classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#***********************************************************************************************************
## 명령행 인자 정의 parser.add_argument
# bash에 python3 main.py --mode train --scope mnist --dir_log ./log
for key, value in default_config.items():
    parser.add_argument(f'--{key}', default=value, type=type(value), dest=key)
    
#***********************************************************************************************************
# utils.py에서 정의한 Parser 클래스 사용해 명령행 인자처리 "객체" 생성
PARSER = Parser(parser)

#***********************************************************************************************************
# main 함수
def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test()

if __name__ == '__main__':
    main()