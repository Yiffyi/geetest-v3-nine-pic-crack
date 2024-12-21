# geetest-v3-nine-pic-crack
自动完成极验三代九宫格验证码（米游社同款）

模型和数据集可以在 Hugging Face 下载。

请遵守我和项目 ravizhan/geetest-v3-click-crack 作者的 AGPL 许可证

省流：在本项目及其修改版本基础上搭建的（公开） API 服务，也需要（向使用者）开放源代码

https://huggingface.co/Yiffyi/resnet-geetest-v3-nine-pic
https://huggingface.co/datasets/Yiffyi/dataset-geetest-v3-nine-pic

## 项目参考

- https://github.com/luguoyixiazi/test_nine
- https://github.com/taisuii/ClassificationCaptchaOcr
- https://github.com/ravizhan/geetest-v3-click-crack

## 安装依赖

参考 `Pipfile` 文件，如果只做本地推理，安装默认的包即可。需要训练模型的，根据自己的环境选装 `train-cuda` 或者 `train-cpu` 类别。如果你需要在服务器上开一个 API 接口（`inference_server.py`）用于其他程序（米游社自动签到等）调用，可以安装 `http` 类别。

温馨提示：训练模型不需要去 Nvidia 官网下载 cuDNN，直接安装本项目依赖理论上就可以直接使用，很方便，推荐你尝试一下。

```shell
pip install pipenv
# 仅推理
pipenv install

# 训练
pipenv install --categories train-cuda

# API 接口
pipenv install --categories http
# 
```

## 模型简介

详情可参考 `train.py`。前期收集数据发现极验后端数据可能会出现 90 种不同的物品。某次验证中所有的图片都在这 90 个类别当中选取，即使它并非答案，也可以人工归类用作训练。人工标注约 200+ 次验证后，每个类别大概都有 20 张左右的数量。

每次极验 challenge 的小图都是一样的，一共 90 张，用 `cv2` 标准差直接判断问的是哪一类。

模型用 `ResNet18` 作为主干，将九宫格的每一格提取出 90 维的特征向量，选择概率最高的一个类别。将九张图片的类别与 challenge 小图的类别比较，选择匹配的图片。

效果：模型 40MB，纯 CPU 推理 9 张图片总耗时 <100ms，准确率 >95%。

## 搭建推理 API 服务器

如果你只想要快速开启一个 API 接口，用于自动通过极验验证码，可以参考 `inference_server.py` 文件，直接运行即可。

主程序会自动下载我训练好的模型，并在 `http://127.0.0.1:3333/crack_it` 开启一个 API 接口。

调用方法为 `http://127.0.0.1:3333/crack_it?gt=xxx&challenge=xxx`，其中 `xxx` 为极验验证码的 `gt` 和 `challenge` 参数。

## 手工构建数据集

如果你想要手动从零收集数据，那么你可以参考 `data_collector.py` 文件。

程序开头有一些参数需要修改，这里简单解释一下作用。

```python
# 用于匹配 challenge 小图类别的阈值，如果标准差小于这个值，则判定小图的类别为已有的某个类别。
# 实际如果匹配上是小于 1 的，不匹配大概在 50 左右。
GUESS_CATEGORY_THRESHOLD = 10

# 指定一共有 90 个类别。如果出现新类别或者数据集种类不够，自动报错推出。
FREEZE_CATEGORY_NUM = 90

# 是否提交极验服务验证结果
SHOW_GEETEST_RESULT = True
```

## 训练

在 `train.py` 文件，底部选择 `train` `convert` 等不同的人物。训练模型选择 `train` 并指定 `epoch`。

## 验证模型

执行 `model_validator.py` 文件，会尝试用训练好的模型通过极验的验证。程序开头有一些参数需要修改，可以直接进入文件查看注释。

=== 以下为原README ===

# geetest-v3-click-crack
极验三代文字点选验证码破解

**本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

**本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

**本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

# 使用方法

安装相关依赖

```commandline
pip install -r requirements.txt
```
运行

```commandline
python main.py
```

验证全过程耗时4s左右 (极验限制，不能更短)

成功率80%左右

# DEMO

``` python
# 实例化两个类
crack = Crack(gt, challenge)
model = Model()
# 按顺序执行以下四个函数
crack.gettype()
crack.get_c_s()
crack.ajax()
for retry in range(6):
    pic_content = crack.get_pic(retry)
    # 检测文字位置
    small_img, big_img = model.detect(pic_content)
    # 判断点选顺序
    result_list = model.siamese(small_img, big_img)
    point_list = []
    for i in result_list:
        left = str(round((i[0] + 30) / 333 * 10000))
        top = str(round((i[1] + 30) / 333 * 10000))
        point_list.append(f"{left}_{top}")
    # 验证请求
    # 注意 请确保验证与获取图片间隔不小于2s
    # 否则会报 duration short
    result = crack.verify(point_list)
    print(result)
    if eval(result)["data"]["result"] == "success":
        break
```

# 协议
本项目遵循 AGPL-3.0 协议开源，请遵守相关协议。

# 鸣谢
[ultralytics](https://github.com/ultralytics/ultralytics/) 提供目标检测模型

[Siamese-pytorch](https://github.com/bubbliiiing/Siamese-pytorch) 提供孪生网络模型

[biliTicker_gt](https://github.com/Amorter/biliTicker_gt) 提供部分思路

[https://www.52pojie.cn/thread-1909489-1-1.html](https://www.52pojie.cn/thread-1909489-1-1.html) 提供部分思路

ChatGPT 提供逆向支持
