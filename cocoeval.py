# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/12/18 20:46
  @version V1.0
"""
import argparse
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOValidator:

    def __init__(self, args):
        self.args = args

    def _get_task(self, opt_str):
        if opt_str == 'detect':
            return 'bbox'

        if opt_str == 'segment':
            return 'segm'
        if opt_str == 'keypoints':
            return 'keypoints'

    def eval(self):

        cocoGt = COCO(self.args.anno_json)
        cocoDt = cocoGt.loadRes(self.args.pred_json)

        cocoEval = COCOeval(cocoGt, cocoDt, self._get_task(self.args.task))

        # 执行评估
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        if self.args.save:
            # project_path = os.path.dirname(os.getcwd())
            # abs_save_path = os.path.join(project_path, self.args.save_path)
            if not os.path.exists(self.args.save_folder_path):
                os.makedirs(self.args.save_folder_path)

            with open(os.path.join(self.args.save_folder_path, f"{self.args.name}.txt"), 'w') as f:
                f.write(str(cocoEval.stats))
            if self.args.log:
                with open(os.path.join(self.args.save_folder_path, f"{self.args.name}-log.txt"), 'w') as f:
                    print(cocoEval.stats, file=f)

        # 打印结果
        return cocoEval.stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-json', default='', type=str,
                        help="Path of gt json,coco format, such as data/anno.json")
    parser.add_argument('--pred-json', default='', type=str,
                        help="Path of pred json ,generated by yolov8 validator,  such as runs/predictions.json")
    parser.add_argument('--save-folder-path', type=str, default='run/cocoeval/',
                        help="Folder path path of result")
    parser.add_argument('--name', type=str, default='cocoeval',
                        help="File Name of result")
    parser.add_argument('--save', action='store_true',
                        help="save or not ")
    parser.add_argument('--log', action='store_true',
                        help="save log or not ")
    parser.add_argument('--task', type=str, default='detect',
                        help="[detect, segment, keypoints] ")

    args = parser.parse_args()

    validator = COCOValidator(args)
    validator.eval()
