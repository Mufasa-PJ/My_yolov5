import os
import torch
import sys

# 解决python weight_only 参数不兼容问题
# 关键：添加YOLOv5模型类到PyTorch安全全局列表，适配2.6+版本


# 调用和使用模型参数
def t0():
    # torch.hub --> 主要作用是支持直接从github上下载代码以及模型文件，加载恢复进行预测-->只要求你记载我的模型文件时又hubconf文件
    model = torch.hub.load(
        repo_or_dir="ultralytics/yolov5:v7.0" , # 给定github上的项目名称或者本地文件夹路径
        model = "yolov5s", # 给定模型文件，其实就是hubconf.py中的方法名
        source= 'github', # 加载的代码/模型来源：可选：github. local
    )  # 需要yolov5.pt 放到和hubconf同一层 不然会报错

    # 模型预测
    img = r"../data/images/bus.jpg"

    # Inference 模型的推理预测
    results = model(img)

    # result 结果展示
    print(results)
    results.show()




def t1():
    # torch.hub-->主要作用是支持直接从github上下载代码以及模型文件，加载恢复进行预测-->只要求你加载的模型文件有hubconf.py文件
    print(os.path.abspath(".."))
    model = torch.hub.load(
        repo_or_dir= "..", # 给定github上的项目名称或者本地文件夹路径
        model="yolov5", # 给定模型文件，其实就是hubconf.py中的方法名
        source='local' # 加载的代码/模型来源：可选：github,local
    )

def t2():
    print(sys.path)

    from hubconf import yolov5s
    from models.common import Detections

    model = yolov5s()

    # 模型来源
    img = r"../data/images/bus.jpg"

    # Inference 模型的推理预测
    results: Detections = model(img)

    # Results 结果展示
    results.print()
    results.show()


'''
总结：
1.其实就是用3种方式找到模型的位置，1.可以直接下载，2.可以通过hubconf 和torch来导入代码。个人建议第三种
2.准备数据集，要求跟coco128.yaml的格式一样，可以参考datas文件夹中的数据格式。
3.训练的时候用模型训练中的终端训练方式训练。
附加：模型训练：python train.py --data ./data/coco128-copy.yaml --epochs 3 --weights  ./study/yolov5s.pt --batch-size 8
--workers 0 --name coco128 --device cpu
    模型推理；python detect.py --weights yolov5s.pt --source 0    
'''

if __name__ == "__main__":
    t2()

