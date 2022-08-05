import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth') #torch.save() でモデルを保存 #基本 .pt または .pth ファイルの形

model = models.vgg16() #pretrained=True は普通はしない
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

"""
簡単にこの動きは出来るようにしたい

torch.save(model, 'model.pth')
model = torch.load('model.pth')
"""