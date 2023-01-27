# coding=utf-8
# @Project  ：SAFECount 
# @FileName ：data_split.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/1/27 18:49
import argparse
import json
import os
import os.path as osp
import sys
sys.path.append('./')
sys.path.append('../')
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser(description='SAFECount data split')
    parser.add_argument('--input_dir', required=True, help='input directory with json files')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--xml_dir', required=True, help='where the test xml file saved')


    parser.add_argument('--test_suffix', type=str, default='xml', help='If the file has xml annotation, it is a test set')
    args = parser.parse_args()
    print(args)
    return args

def json_generator(files, root_dir, target_path):
    total_f = open(target_path, 'w', encoding='utf-8')
    for file in tqdm(files, desc=target_path):
        filename = osp.join(root_dir, file)
        with open(filename, 'r', encoding='utf-8') as f:
            content = json.load(f)
            content = json.dumps(content, indent=4)
            total_f.write(content + '\n')
    total_f.close()

if __name__ == '__main__':
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)

    train_items, test_items = [], []
    for root, _, filenames in os.walk(args.input_dir):
        for filename in filenames:
            if filename.endswith('json') and not filename.startswith('.'):
                name = osp.splitext(filename)[0]
                # has xml file, this image can be test image
                if osp.exists(osp.join(args.xml_dir, name+'.xml')):
                    test_items.append(filename)
                else:
                    train_items.append(filename)

    json_generator(train_items, args.input_dir, osp.join(args.output_dir, 'train.json'))
    json_generator(test_items, args.input_dir, osp.join(args.output_dir, 'test.json'))
