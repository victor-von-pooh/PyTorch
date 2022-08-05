import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def NeuralNetwork_elab():
    input_image = torch.rand(3,28,28)
    print(input_image.size())

    #nn.Flatten() : 28 × 28 = 784 より 'torch.Size([3, 28, 28])' → 'torch.Size([3, 784])' とする
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())

    #nn.Linear() : 線形変換を行い、 入力次元 a → 出力次元 b とする
    layer1 = nn.Linear(in_features=28*28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())

    #nn.ReLU() : 活性化関数「ReLU」
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")

    #nn.Sequential() : ニューラルネットを降順に組み立てる
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image = torch.rand(3,28,28)
    logits = seq_modules(input_image)

    #nn.Softmax() : 出力層から出たものをSoftmax関数に入れる #それぞれが確率となって表される
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu" #cuda(GPU)が使えなかったらCPU #Macはcudaが使えない
    print(f"Using {device} device")

    NeuralNetwork_elab()

    model = NeuralNetwork().to(device) #.to(device) でデバイスに送る
    print(model)

    #model.forward() で直接呼び出すのはNG

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters(): #パラメータの中身を確認したかったら .named_parameters() からできる
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")