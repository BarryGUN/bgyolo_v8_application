# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/12/18 20:46
  @version V1.0
"""
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
	pred_json = 'w/dense-eiou/predictions.json'
	anno_json = 'data/val.json'

	# 使用COCO API加载预测结果和标注
	cocoGt = COCO(anno_json)
	# cocoDt = cocoGt.loadRes(pred_json)

	with open(pred_json, encoding='utf-8') as f:
		preds = json.load(f)

	predsw = [pred['image_id'] for pred in preds]
	gt = cocoGt.getImgIds()
	#
	print(predsw)
	print('-----------------------------------------')
	# a = set(predsw) & set(gt)
	print(gt)
	# print(a)
	# print(set(predsw))
	# # 创建COCOeval对象
	# cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
	#
	# # 执行评估
	# cocoEval.evaluate()
	# cocoEval.accumulate()
	# cocoEval.summarize()
	#
	# # 保存结果
	# with open('output/coco_eval.txt', 'w') as f:
	# 	f.write(str(cocoEval.stats))
	#
	# # 打印结果
	# print(cocoEval.stats)

