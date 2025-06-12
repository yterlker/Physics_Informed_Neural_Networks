import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
# 设置字体
config = {
    "font.family": 'serif',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# 设定随机种子以确保可重复性
random.seed(23)
np.random.seed(23)
torch.manual_seed(26)
selected_random = 10
# 文件夹路径
folder_path = 'PINN'
# 获取文件夹中所有Excel文件的列表
file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# 随机选择文件作为训练集
sample_size = 20
selected_files = random.sample(file_list, sample_size)
# 打印选取的文件名称
print("选取的文件:")
for file in selected_files:
    print(file)
# 加载选定文件的数据
data_frames = []
time_values = []
for file in selected_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    # 随机选取
    random_rows = df.sample(n=512, random_state=selected_random)
    data_frames.append(random_rows)
    # 提取时间信息
    time_value = int(''.join(filter(str.isdigit, os.path.splitext(file)[0]))) * 0.1
    time_values.extend([time_value] * len(random_rows))
# 合并所有加载的数据
combined_data = pd.concat(data_frames, ignore_index=True)
# 提取输入特征 (X, Y, Z, t) 和输出标签 (u, v, w, p, C)
X_data = torch.tensor(combined_data[['X Coordinate', 'Y Coordinate', 'Z Coordinate']].values, dtype=torch.float32)
t_data = torch.tensor(time_values, dtype=torch.float32).unsqueeze(1)  # 将时间值作为输入的一部分
# 构建输入张量
input_data = torch.cat([X_data, t_data], dim=1)
# 输出特征
output_data = torch.tensor(combined_data[['x-velocity', 'y-velocity', 'z-velocity', 'pressure', 'ch4']].values,
                           dtype=torch.float32)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    input_data, output_data, test_size=0.2, random_state=26, shuffle=True
)
# 查看设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的的设备为：{device}")
# 将输入和输出数据加载到设备
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
# 创建数据集
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
# 瓦斯出口边界条件，初始条件为零
mask = 0
mask_1 = ((X_train[:, 0] == 1.55) &
          ((X_train[:, 1] - 0.17) ** 2 + (X_train[:, 2] - 0.5) ** 2 <= 0.1 ** 2))
mask_1 = mask_1.to(device)
mask_2 = ((X_train[:, 0] == 1.55) &
          ((X_train[:, 1] - 0.17) ** 2 + (X_train[:, 2] - 1.02) ** 2 <= 0.1 ** 2))
mask_2 = mask_2.to(device)
# 选取数据对比预测数据
dp_plt = pd.read_excel("PINN\\data20.xlsx")
dp_plt = dp_plt.sample(n=256, random_state=selected_random)
print(dp_plt.shape)
# PINN
class SubNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation, dropout_prob):
        super(SubNetwork, self).__init__()
        layers = []
        previous_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(previous_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(activation())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            previous_size = size
        layers.append(nn.Linear(previous_size, output_size))
        self.network = nn.Sequential(*layers)

        # 对所有 Linear 层进行权重初始化
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


class PINN(nn.Module):
    def __init__(self, input_size=4, hidden_sizes=None, output_size=512):
        super(PINN, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 128, 128, 256, 256]
        self.shared = SubNetwork(input_size, hidden_sizes, output_size, activation=nn.LeakyReLU, dropout_prob=0)

        # 使用共享层的输出连接各个子网络
        self.subnet_u = SubNetwork(output_size, [512] * 2, 1, activation=nn.LeakyReLU, dropout_prob=0)
        self.subnet_v = SubNetwork(output_size, [512] * 2, 1, activation=nn.LeakyReLU, dropout_prob=0)
        self.subnet_w = SubNetwork(output_size, [512] * 2, 1, activation=nn.LeakyReLU, dropout_prob=0)
        self.subnet_p = SubNetwork(output_size, [512] * 2, 1, activation=nn.LeakyReLU, dropout_prob=0)
        self.subnet_c = SubNetwork(output_size, [512] * 2, 1, activation=nn.LeakyReLU, dropout_prob=0)

    def forward(self, x):
        # 将输入传递给共享部分
        shared_output = self.shared(x)
        # 传递给各个子网络
        u = self.subnet_u(shared_output)
        v = self.subnet_v(shared_output)
        w = self.subnet_w(shared_output)
        p = self.subnet_p(shared_output)

        c = self.subnet_c(shared_output)

        return u, v, w, p, c

# 定义物理损失函数，NS方程和边界瓦斯扩散方程
def physics_loss(model_, train_data_x, Ro, Mu, D_, mask_):
    x = train_data_x[:, 0:1]
    y = train_data_x[:, 1:2]
    z = train_data_x[:, 2:3]
    t = train_data_x[:, 3:4]

    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = z.requires_grad_(True)
    t = t.requires_grad_(True)

    # 将x, y, z, t合并成一个输入张量
    inputs = torch.cat([x, y, z, t], dim=1)

    # 确保输入张量具备梯度跟踪能力
    inputs = inputs.requires_grad_(True)

    # 调用模型，传入合并后的张量
    u_, v_, w_, p, C_ = model_(inputs)

    # 对速度u的导数计算
    u_t = torch.autograd.grad(u_, t, grad_outputs=torch.ones_like(u_), create_graph=True)[0]
    u_x = torch.autograd.grad(u_, x, grad_outputs=torch.ones_like(u_), create_graph=True)[0]
    u_y = torch.autograd.grad(u_, y, grad_outputs=torch.ones_like(u_), create_graph=True)[0]
    u_z = torch.autograd.grad(u_, z, grad_outputs=torch.ones_like(u_), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]

    # 对速度v的导数计算
    v_t = torch.autograd.grad(v_, t, grad_outputs=torch.ones_like(v_), create_graph=True)[0]
    v_x = torch.autograd.grad(v_, x, grad_outputs=torch.ones_like(v_), create_graph=True)[0]
    v_y = torch.autograd.grad(v_, y, grad_outputs=torch.ones_like(v_), create_graph=True)[0]
    v_z = torch.autograd.grad(v_, z, grad_outputs=torch.ones_like(v_), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]

    # 对速度w的导数计算
    w_t = torch.autograd.grad(w_, t, grad_outputs=torch.ones_like(w_), create_graph=True)[0]
    w_x = torch.autograd.grad(w_, x, grad_outputs=torch.ones_like(w_), create_graph=True)[0]
    w_y = torch.autograd.grad(w_, y, grad_outputs=torch.ones_like(w_), create_graph=True)[0]
    w_z = torch.autograd.grad(w_, z, grad_outputs=torch.ones_like(w_), create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), create_graph=True)[0]

    # 对压力p的导数计算
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # 对CH4浓度C的导数计算
    C_t = torch.autograd.grad(C_, t, grad_outputs=torch.ones_like(C_), create_graph=True)[0]
    C_x = torch.autograd.grad(C_, x, grad_outputs=torch.ones_like(C_), create_graph=True)[0]
    C_y = torch.autograd.grad(C_, y, grad_outputs=torch.ones_like(C_), create_graph=True)[0]
    C_z = torch.autograd.grad(C_, z, grad_outputs=torch.ones_like(C_), create_graph=True)[0]
    C_xx = torch.autograd.grad(C_x, x, grad_outputs=torch.ones_like(C_x), create_graph=True)[0]
    C_yy = torch.autograd.grad(C_y, y, grad_outputs=torch.ones_like(C_y), create_graph=True)[0]
    C_zz = torch.autograd.grad(C_z, z, grad_outputs=torch.ones_like(C_z), create_graph=True)[0]

    # Navier-Stokes方程残差计算
    f_u = (u_t + u_ * u_x + v_ * u_y + w_ * u_z) + p_x/Ro - Mu * (u_xx + u_yy + u_zz)
    f_v = (v_t + u_ * v_x + v_ * v_y + w_ * v_z) + p_y/Ro - Mu * (v_xx + v_yy + v_zz) - 9.81
    f_w = (w_t + u_ * w_x + v_ * w_y + w_ * w_z) + p_z/Ro - Mu * (w_xx + w_yy + w_zz)

    Q = 3.14 * 0.1**2 * 3.2
    m = 0.671 * Q

    # 连续性方程残差计算
    continuity = u_x + v_y + w_z
    # 根据时间选取瓦斯扩散边界
    T_p = t[0]
    if (T_p >= 1) and (T_p <= 3):
        mask_ = mask_1
    if (T_p >= 5) and (T_p <= 8):
        mask_ = mask_2
    # CH4边界扩散方程残差计算
    f_C = C_t + u_ * C_x + v_ * C_y + w_ * C_z - D_ * (C_xx + C_yy + C_zz) - mask_ * m

    return (torch.mean(f_u ** 2) + torch.mean(f_v ** 2) + torch.mean(f_w ** 2) +
            torch.mean(continuity ** 2) + torch.mean(f_C ** 2))
# 间隔点数
plot_idx = 2
# 对训练集过程可视化方法瓦斯C
def plot_and_save_c(epoch, c_plt, c_pred, folder='training_plots'):
    c_pred_plt = c_pred.detach().cpu().numpy().flatten()
    c_plt = c_plt.cpu().numpy().flatten()
    # 子采样，每plot_idx个点取一个
    num_points = len(dp_plt)
    indices = np.arange(0, num_points, plot_idx)
    C_pred_subsampled = c_pred_plt[indices]
    y_train_subsampled = c_plt[indices]
    indices_plt = np.arange(0,len(indices))
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(7, figsize=(16, 6))  # 指定第二个图形窗口
    plt.clf()  # 清除之前的图形
    plt.plot(indices_plt, y_train_subsampled, color='red', label='真实值', linewidth=2)
    plt.plot(indices_plt, C_pred_subsampled, color='blue', linestyle='-', marker='.', label='预测值')
    plt.title(f'第 {epoch} 轮训练集瓦斯浓度对比图')
    plt.xlabel('节点')
    plt.ylabel('瓦斯浓度')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/epoch_{epoch}.png')  # 保存图像
    plt.pause(0.01)  # 短暂暂停，以便图形更新
# 对训练集过程可视化风速V
def plot_and_save_v(epoch, v_plt, v_pred, folder='training_plots'):
    v_pred_plt = v_pred.detach().cpu().numpy().flatten()
    v_plt = v_plt.cpu().numpy().flatten()
    # 子采样，每plot_idx个点取一个
    num_points = len(dp_plt)
    indices = np.arange(0, num_points, plot_idx)
    C_pred_subsampled = v_pred_plt[indices]
    y_train_subsampled = v_plt[indices]
    indices_plt = np.arange(0,len(indices))
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(8, figsize=(16, 6))  # 指定第二个图形窗口
    plt.clf()  # 清除之前的图形
    plt.plot(indices_plt, y_train_subsampled, color='red', label='真实值', linewidth=2)
    plt.plot(indices_plt, C_pred_subsampled, color='blue', linestyle='-', marker='.', label='预测值')
    plt.title(f'第 {epoch} 轮训练集风速对比图')
    plt.xlabel('节点')
    plt.ylabel('风速')
    plt.ylim(0, 12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/epoch_{epoch}.png')  # 保存图像
    plt.pause(0.01)  # 短暂暂停，以便图形更新
# 对测试集测试效果可视化瓦斯C
def plot_and_save_c_test(epoch, c_plt_t, c_pred_t, folder='training_plots'):
    c_pred_plt = c_pred_t.detach().cpu().numpy().flatten()
    c_plt = c_plt_t.cpu().numpy().flatten()
    # 子采样，每plot_idx个点取一个
    num_points = len(dp_plt)
    indices = np.arange(0, num_points, plot_idx)
    C_pred_subsampled = c_pred_plt[indices]
    y_train_subsampled = c_plt[indices]
    indices_plt = np.arange(0,len(indices))
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(9, figsize=(16, 6))  # 指定第二个图形窗口
    plt.clf()  # 清除之前的图形
    plt.plot(indices_plt, y_train_subsampled, color='red', label='真实值', linewidth=2)
    plt.plot(indices_plt, C_pred_subsampled, color='blue', linestyle='-', marker='.', label='预测值')
    plt.title(f'第 {epoch} 轮测试集瓦斯浓度对比图')
    plt.xlabel('节点')
    plt.ylabel('瓦斯浓度')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/epoch_{epoch}.png')  # 保存图像
    plt.pause(0.01)  # 短暂暂停，以便图形更新
# 对测试集测试效果可视化风速V
def plot_and_save_v_test(epoch, v_plt_t, v_pred_t, folder='training_plots'):
    v_pred_plt = v_pred_t.detach().cpu().numpy().flatten()
    v_plt = v_plt_t.cpu().numpy().flatten()
    # 子采样，每个plot_idx点取一个
    num_points = len(dp_plt)
    indices = np.arange(0, num_points, plot_idx)
    C_pred_subsampled = v_pred_plt[indices]
    y_train_subsampled = v_plt[indices]
    indices_plt = np.arange(0,len(indices))
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(10, figsize=(16, 6))  # 指定第二个图形窗口
    plt.clf()  # 清除之前的图形
    plt.plot(indices_plt, y_train_subsampled, color='red', label='真实值', linewidth=2)
    plt.plot(indices_plt, C_pred_subsampled, color='blue', linestyle='-', marker='.', label='预测值')
    plt.title(f'第 {epoch} 轮测试集风速对比图')
    plt.xlabel('节点')
    plt.ylabel('风速')
    plt.ylim(0, 12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/epoch_{epoch}.png')  # 保存图像
    plt.pause(0.01)  # 短暂暂停，以便图形更新

# 初始化图表绘制的存储文件夹
plot_folder_c = 'GIFC'
plot_folder_v = 'GIFV'
plot_folder_c_t = 'GIFC_T'
plot_folder_v_t = 'GIFV_T'

# PINN模型训练
# 初始化模型和优化器
model = PINN().to(device)
optimizer = optim.Adam([
    {'params': model.shared.parameters(), 'lr': 0.0001},
    {'params': model.subnet_u.parameters(), 'lr': 0.0001},
    {'params': model.subnet_v.parameters(), 'lr': 0.0001},
    {'params': model.subnet_w.parameters(), 'lr': 0.0001},
    {'params': model.subnet_p.parameters(), 'lr': 0.0001},
    {'params': model.subnet_c.parameters(), 'lr': 0.0001}
], betas=(0.85, 0.999))  # 默认学习率

# 超参数
epochs = 6000
# 流体参数　ro为空气密度　ro_u为风速　muv为运动粘度（计算物理损失时方程进行了代换，两边除于密度，将动力粘度转化为运动粘度）
ro = 1.225
ro_u = 5
muv = 1.461e-5
# CH4扩散系数
D = 0.0016
lambda1, lambda2 = 1, 10  # 调节物理损失和数据损失的权重，同步放大物理损失

# 存储损失值
loss_history = []
test_loss_history = []

# 最近的测试损失
last_test_loss = None
# 训练循环
start_time = time.time()
current_time = datetime.now()
print(f"开始时间：{current_time}")
for epoch in range(epochs):
    model.train()
    # 遍历训练批次
    for X_batch, y_batch in train_loader:
        # 对每个优化器执行零梯度操作
        if isinstance(optimizer, dict):
            for opt in optimizer.values():
                opt.zero_grad()
        else:
            optimizer.zero_grad()

        # 前向传播，计算预测值
        u_pred, v_pred, w_pred, p_pred, C_pred = model(X_batch)
        v_p = torch.sqrt(u_pred ** 2 + v_pred ** 2 + w_pred ** 2)

        # 计算数据损失
        criterion = nn.MSELoss(reduction='mean')
        data_loss = (criterion(u_pred * 10, y_batch[:, 0:1] * 10) +
                criterion(v_pred * 10, y_batch[:, 1:2] * 10) +
                criterion(w_pred * 10, y_batch[:, 2:3] * 10) +
                criterion(p_pred, y_batch[:, 3:4]) +
                criterion(C_pred * 20, y_batch[:, 4:5] * 20))

        v_real = torch.sqrt(y_batch[:, 0:1] ** 2 + y_batch[:, 1:2] ** 2 + y_batch[:, 2:3] ** 2)
        # 物理和边界损失
        loss_phys = physics_loss(model, X_batch, ro, muv, D, mask)
        # 总损失
        total_loss = lambda1 * data_loss + lambda2 * loss_phys
        total_loss.backward()
        if epoch % 10 == 0:
            # 训练过程可视化图
            plot_and_save_c(epoch, y_batch[:, 4:5], C_pred, plot_folder_c)
            plot_and_save_v(epoch, v_real, v_p, plot_folder_v)

        # 对每个优化器执行更新步骤
        if isinstance(optimizer, dict):
            for opt in optimizer.values():
                opt.step(lambda: total_loss.item())
        else:
            optimizer.step()
        # 记录训练损失
    loss_history.append(total_loss.item())

    # 验证测试集
    model.eval()
    u_pred, v_pred, w_pred, p_pred, C_pred = model(X_test)
    v_test = torch.sqrt(u_pred ** 2 + v_pred ** 2 + w_pred ** 2)

    criterion = nn.MSELoss(reduction='mean')
    test_data_loss = (criterion(u_pred * 10, y_test[:, 0:1] * 10) +
                 criterion(v_pred * 10, y_test[:, 1:2] * 10) +
                 criterion(w_pred * 10, y_test[:, 2:3] * 10) +
                 criterion(p_pred, y_test[:, 3:4]) +
                 criterion(C_pred * 20, y_test[:, 4:5] * 20))
    v_real_t = torch.sqrt(y_test[:, 0:1] ** 2 + y_test[:, 1:2] ** 2 + y_test[:, 2:3] ** 2)

    # 测试损失
    test_loss = test_data_loss
    test_loss_history.append(test_loss.item())
    if (epoch) % 10 == 0:
        # 验证可视化图
        plot_and_save_c_test((epoch), y_test[:, 4:5], C_pred, plot_folder_c_t)
        plot_and_save_v_test((epoch), v_real_t, v_test, plot_folder_v_t)
    if total_loss.item() < 10 and test_loss < 10:
        print(f'轮数 {epoch}: 早停触发。')
        break
    if (epoch) % 2 == 0:
        # 初始化图表
        plt.figure(11, figsize=(16, 6))
        plt.ion()  # 打开交互模式
        # 更新图表
        plt.clf()  # 清除当前图形
        plt.plot(loss_history, label='训练损失', color='blue')
        plt.plot(test_loss_history, label='测试损失', color='red')
        plt.xlabel('步数')
        plt.ylabel('损失')
        plt.ylim(0, 1000)
        plt.title('PINN训练过程')
        plt.legend()
        plt.grid(True)
        plt.pause(0.01)
    # 打印损失
    if epoch % 10 == 0:
        print(f"训练物理边界损失：{loss_phys * lambda2:.1f}")
        print(f'第 {epoch}, 训练损失: {total_loss.item():.1f}, 测试损失: {test_loss.item():.1f}')
# 记录训练结束时间
end_time = time.time()
# 计算并打印总训练时长
total_time = end_time - start_time
print(f"总时长: {total_time:.1f} s")

# 检查CUDA设备是否可用，然后将模型移到CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# 保存模型的状态字典
torch.save(model.state_dict(), 'PINN_C.pth')

# 创建四个独立的输入张量，每个都是1x1的
dummy_x = torch.randn(1, 1)
dummy_y = torch.randn(1, 1)
dummy_z = torch.randn(1, 1)
dummy_t = torch.randn(1, 1)

# 确保所有输入也移到相同的设备
dummy_x = dummy_x.to(device)
dummy_y = dummy_y.to(device)
dummy_z = dummy_z.to(device)
dummy_t = dummy_t.to(device)

inputs_onnx = torch.cat([dummy_x, dummy_y, dummy_z, dummy_t], dim=1)

# 导出PINN模型为ONNX格式
torch.onnx.export(model, inputs_onnx, "PINN_C.onnx", input_names=['input'], output_names=['output'], opset_version=11)

plt.ioff()  # 关闭交互模式
plt.show()  # 显示最终图表

file_path = 'PINN\\data20.xlsx'
df = pd.read_excel(file_path)
t_fixed = 2
# 提取X, Y, Z
X = torch.tensor(df['X Coordinate'].values, dtype=torch.float32).to(device)
Y = torch.tensor(df['Y Coordinate'].values, dtype=torch.float32).to(device)
Z = torch.tensor(df['Z Coordinate'].values, dtype=torch.float32).to(device)
# 将Y坐标扩展到与X和Z相同的形状，并确保Y与X类型相同
T = torch.full_like(X, t_fixed, dtype=X.dtype).to(device)
X_sub = X.unsqueeze(1)
Y_sub = Y.unsqueeze(1)
Z_sub = Z.unsqueeze(1)
T_sub = T.unsqueeze(1)

input_model = torch.cat([X_sub, Y_sub, Z_sub, T_sub], dim=1)
u, v, w, _, C = model(input_model)

# 计算速度大小 (magnitude)`
velocity_magnitude = torch.sqrt(u**2 + v**2 + w**2)

# 对比
data = pd.read_excel(file_path)

data['V'] = np.sqrt(data['x-velocity']**2 + data['y-velocity']**2 + data['z-velocity']**2)
data['C'] = data['ch4']

# 间隔点数plot_idx个
data_subsampled = data.iloc[::10]

df = pd.read_excel(file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.tensor(df['X Coordinate'].values, dtype=torch.float32).to(device)
Y = torch.tensor(df['Y Coordinate'].values, dtype=torch.float32).to(device)
Z = torch.tensor(df['Z Coordinate'].values, dtype=torch.float32).to(device)
T = torch.full_like(X, t_fixed, dtype=X.dtype).to(device)
input_model = torch.stack([X, Y, Z, T], dim=1)
# 预测数据
u, v, w, _, C = model(input_model)
velocity_magnitude = torch.sqrt(u**2 + v**2 + w**2)
C_plot = C.cpu().detach().numpy()
velocity_magnitude_plot = velocity_magnitude.cpu().detach().numpy()

num_points = len(df)
# 间隔点数plot_idx个
indices = np.arange(0, num_points, 10)
C_plot_subsampled = C_plot[indices]
velocity_magnitude_plot_subsampled = velocity_magnitude_plot[indices]

subsampled_indices = np.arange(0, len(data_subsampled))
# 绘图
plt.figure(figsize=(16, 6))
plt.plot(subsampled_indices, data_subsampled['C'], label='真实值', color='red', linestyle='-', linewidth=2)
plt.plot(subsampled_indices, C_plot_subsampled, label='预测值', color='blue', linestyle='-', marker='.', linewidth=2)
plt.title('浓度对比')
plt.xlabel('节点')
plt.ylabel('瓦斯浓度')
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(subsampled_indices, data_subsampled['V'], label='真实值', color='red', linestyle='-', linewidth=2)
plt.plot(subsampled_indices, velocity_magnitude_plot_subsampled, label='预测值', color='blue', linestyle='-', marker='.', linewidth=2)
plt.title('风速对比')
plt.xlabel('节点')
plt.ylabel('风速')
plt.ylim(0, 12)
plt.grid(True)
plt.legend()
plt.show()
