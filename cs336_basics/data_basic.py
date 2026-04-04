import numpy.typing as npt
import torch
import numpy as np
import os
import yaml



def load_memmap_dataset(file_path, dtype=np.uint16):
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    # 计算元素个数：总字节数 / 每个元素的字节数
    num_elements = file_size // np.dtype(dtype).itemsize
    
    # 创建内存映射（只读模式 'r'）
    # 这不会占用 100GB 的内存，它只是建立了虚拟地址到磁盘文件的映射
    data = np.memmap(file_path, dtype=dtype, mode='r', shape=(num_elements,))
    return data

def get_batch(dataset,batch_size: int, context_length: int, device: str):
    # device: e.g. 'cpu' or 'cuda:0'
    # api: x_tensor = torch.from_numpy(np.array(x_batch)).to(torch.long).to(device)
    # 随机采样
    data_num=len(dataset)
    starts=np.random.randint(0,data_num-context_length,size=(batch_size,))
    x_batch=[dataset[s:(s+context_length)] for s in starts]
    y_batch=[dataset[(s+1):(s+context_length+1)] for s in starts]
    x_tensor = torch.from_numpy(np.array(x_batch)).to(torch.long).to(device)
    y_tensor = torch.from_numpy(np.array(y_batch)).to(torch.long).to(device)
    return (x_tensor,y_tensor)

def save_checkpoint(model,optimizer,iteration,out):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint,out) # 将对象 obj（通常是字典）序列化并写入 out

def load_checkpoint(src, model, optimizer):
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

