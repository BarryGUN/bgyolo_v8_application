# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/12/18 20:46
  @version V1.0
"""
import argparse
import os

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOValidator:

    def __init__(self, args):
        self.args = args
        self.cocoGt = COCO(self.args.anno_json)
        self.cocoDt = self.cocoGt.loadRes(self.args.pred_json)

    def _get_task(self, opt_str):
        if opt_str == 'detect':
            return 'bbox'

        if opt_str == 'segment':
            return 'segm'
        if opt_str == 'keypoints':
            return 'keypoints'

    def save(self, stats, folder, name):

        indexes = [
            ('AP', 'area=all', 'IoU=50:95'), ('AP', 'area=all', 'IoU=50'),
            ('AP', 'area=all', 'IoU=75'), ('AP', 'area=small', 'IoU=50:95'),
            ('AP', 'area=medium', 'IoU=50:95'), ('AP', 'area=large', 'IoU=50:95'),
            ('AR', 'area=all', 'IoU=50:95'), ('AR', 'area=all', 'IoU=50'),
            ('AR', 'area=all', 'IoU=75'), ('AR', 'area=small', 'IoU=50:95'),
            ('AR', 'area=medium', 'IoU=50:95'), ('AR', 'area=large', 'IoU=50:95'),
        ]

        values = [
            round(stats[0], 3), round(stats[1], 3), round(stats[2], 3),
            round(stats[3], 3), round(stats[4], 3), round(stats[5], 3),
            round(stats[6], 3), round(stats[7], 3), round(stats[8], 3),
            round(stats[9], 3), round(stats[10], 3),round(stats[11], 3),

                 ]
        indexes = pd.MultiIndex.from_tuples(indexes)
        pd.DataFrame(data=values, index=indexes).T.to_csv(os.path.join(folder, f'{name}.csv'))

    def eval(self):

        cocoEval = COCOeval(self.cocoGt, self.cocoDt, self._get_task(self.args.task))

        # eval
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        if self.args.save:
            self.save(cocoEval.stats, self.args.save_folder_path, self.args.name)
        # result
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
    parser.add_argument('--task', type=str, default='detect',
                        help="[detect, segment, keypoints] ")

    args = parser.parse_args()

    validator = COCOValidator(args)
    validator.eval()
