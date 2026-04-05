import  torch

import models.common
from models.yolo import  Model
from pathlib import Path


def t0():
    m = models.common.MP(128)
    x = torch.rand(4,128,608,608)
    r = m(x)
    # print(r)
    print(r.shape)

if __name__ == "__main__":
    t0()