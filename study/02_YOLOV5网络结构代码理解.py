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

if __name__ == "__main__":
    t00()