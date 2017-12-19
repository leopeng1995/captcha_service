### Captcha Service 验证码识别服务

```bash
$ pip install -r requirements.txt
$ python models/amazon.py
```

基本思路：用 CNN 提取验证码特征，将验证码识别问题不单纯看成图像分类问题，而是看成是一种特殊的图像序列问题。CTC 是一种损失函数。
