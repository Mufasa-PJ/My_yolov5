import  torch
from models.yolo import  Model
from pathlib import Path


def t00():
    cfg_path = '../models/yolov5s_copy_00.yaml' # 给定模型配置文件路径
    import yaml # for torch hub
    yaml_file = Path(cfg_path).name
    with open(cfg_path,encoding='ascii',errors='ignore') as f:
        yaml = yaml.safe_load(f) # model dict
    print(yaml)
    print(type(yaml))


# 根据给定的模型配置文件创建模型
def t0():
    cfg_path = '../models/yolov5s_copy_00.yaml'  # 给定模型配置文件路径
    net = Model(
        cfg = cfg_path, # 给定模型配置文件或者dict字典
        ch = 3, # 输入的通道数目
        nc = None, # 类别数目，不给定的时候，直接cfg里面的nc
        anchors = None # 给定出事的先验框尺度数了

    )
    print(net)

    x = torch.randn(2,3,608,608)
    r = net(x)
    print(r)
    print(type(r))
    print(r[0].shape)
    print(r[1].shape)
    print(r[2].shape)

    torch.onnx.export(
        model=net,
        args=(x,),
        f = 'yolov5s_copy_00.onnx',
        input_names=['image'] ,
        output_names=['labels'],
        opset_version=12
    )


def t1():
    # 1.原始默认的配置文件
    # cfg_path = '../models/yolov5s_copy_00.yaml'

    # cfg_path = '../models/yolov5s_copy_01.yaml'
    # 2.增加每一层的每个grid cell 对应的anchor box对应的边框数目 --> 不改变层数，不改变预测分支的数目 P3\P4\P5

    cfg_path = '../models/yolov5s_copy_02.yaml'
    # 3.减少每一层的每个grid cell 对应的anchor box对应的边框数目 --> 不改变层数，不改变预测分支的数目 P3\P4\P5
    # 可以改变yaml文件的结构，也可以用自适应锚点计算
    # 业务场景：本身不需要那么多的计算量

    # cfg_path = '../models/yolov5s_copy_03.yaml'
    # 4.减少预测分支 --> 假定整个业务场景中的物体都是比较大的。
    # 在实际业务中，当物体的大小大于原图的3/20的时候，是为大物体，小于1/20的时候是小物体，可以用这个来看数量来确定网络结构
    # 这里删除了预测分支，一定要记着将anchor box 这个对应的预测组删除
    # cfg_path = '../models/yolov5s_copy_04.yaml'
    # 第二种改法，13 --> 20

    # cfg_path = '../models/yolov5s_copy_05.yaml'
    # 5.减少预测分支 --> 假定整个业务场景中的物体都是比较小的

    # cfg_path = '../models/yolov5s_copy_06.yaml'
    # 6.增加预测分支，--> 假定p3对应的feature map还是太深了，对小物体检测不到
    # 业务场景：当我实际预测的物体在原图中的大小小于 8 * 8 的时候，我在P3的时候其实预测时我的物体就会被下采样没了，所以我需要在P2的时候就进行预测

    # cfg_path = '../models/yolov5s_copy_07.yaml'
    # 7.使用新结构：实现方式1 --> 基于现有的结构实现的。
    # BiC结构优化：在Neck部分的FPN-PAN结构中，将特征融合的方式发生变化（上采样P_i2 + 当前特征C_i1 --> 上采样P_i2 + 当前特征c_i1 + 下采样C_i0）

    cfg_path = '../models/yolov5s_copy_08.yaml'
    # 8.使用新结构：实现方式2 --> 新增模块
    """
    新增模块的步骤：
    1.在common.py文件中新建模块，并测试模块
    例如：MP：下采样可以同时使用下采样和池化。
    2.在yolo.py文件的模块解析函数parse_model中，增加当前新模块的解析逻辑
    3.在你当前的模型配置yaml文件中进行修改，使用新模块
    4.测试
    5.训练使用
    python train.py --data ./data/coco128-copy.yaml --epochs 3 --weights  ./study/yolov5s.pt --batch-size 8 --workers 0 --name coco128_new_net --device cpu --cfg models/yolov5s_copy_08.yaml --hyp ./data/hyps/hyp.scratch-low.yaml
    """



    model = Model(
        cfg=cfg_path,
        ch = 3,
        nc = 80,
        anchors=None
    )
    imgs = torch.rand(6,3,608,608)
    # 训练时候返回各分支/各层对应的预测值，[N,na,H,W,nc + 1 +4]
    r = model(imgs)
    print(type(r))
    print(len(r))
    for r_tensor in r:
        print(r_tensor.shape)
    print(type(r[0]))
    # print(r[:1])

if __name__ == "__main__":
    t1()