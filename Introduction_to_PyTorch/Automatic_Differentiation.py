import torch

x = torch.ones(5) #入力テンソル
y = torch.zeros(3) #予想される出力
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

#逆伝播の関数は .grad_fn で使える #詳細は torch.autograd.Function より
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

#上で出した 'loss' に .backward() をつけると偏導関数が求まる #.grad でそれぞれの値に適用可
loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

#順伝播のみを扱いたい時などは torch.no_grad() または .detach() で
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

inp = torch.eye(5, requires_grad=True) #5次正方行列
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True) #(1)
print(f"First call\n{inp.grad}")
#(1)をもう一度やると結果が異なる
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
#これは前に行われた .grad が累積されているため
inp.grad.zero_() #.grad.zero() で初期化
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")