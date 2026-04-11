import math
import torch
import triton
import triton.language as tl
from torch import autograd
from typing import Type

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # 程序索引，相当于之前pytorch代码中的外层i
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 用相应的 batch 索引乘以每个张量的 batch 步长来偏移每个指针
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd), # 跳一个seq的位置/一个d_k上的位置要多少字节
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 从哪里开始取
        block_shape=(Q_TILE_SIZE, D), # q_block取多大
        order=(1, 0), # 内存顺序
    )
    q_block = tl.load(Q_block_ptr)
    # 初始化 m_i, l_i, o_i
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)



    for j in range(0,N_KEYS,K_TILE_SIZE):
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd), # 跳一个seq的位置/一个d_k上的位置要多少字节
            offsets=(j, 0), # 从哪里开始取
            block_shape=(K_TILE_SIZE, D), # q_block取多大
            order=(1, 0), # 内存顺序
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd), # 跳一个seq的位置/一个d_k上的位置要多少字节
            offsets=(j, 0), # 从哪里开始取
            block_shape=(K_TILE_SIZE, D), # q_block取多大
            order=(1, 0), # 内存顺序
        )
        # 1. 取 K/V Block
        k_block = tl.load(K_block_ptr)
        v_block = tl.load(V_block_ptr)
        # 2. QK^T
        s_ij = tl.dot(q_block, k_block.T).to(tl.float32) * scale # 强制将结果转为 float32 以防溢出
        # 关于mask
        if IS_CAUSAL:
            # 构造 Q 和 K 的位置
            q_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_idx = tl.arange(0, K_TILE_SIZE) + j
            # 掩码：Q 位置 >= K 位置才允许
            mask = q_idx[:, None] >= k_idx[None, :]
            s_ij = tl.where(mask, s_ij, -float("inf"))
        # 3. 计算 softmax, 这段是对着pytorch版本逐行翻译
        m_ij = tl.max(s_ij,1) # 沿着最后一维取最大值
        m_new = tl.maximum(m_i, m_ij) # 逐元素取最大
        exp_old = tl.exp(m_i - m_new)
        exp_new = tl.exp(s_ij - m_new[:,None])
        l_i = exp_old * l_i + tl.sum(exp_new,axis=1) # axis=1: 沿K维度求和
        o_i = exp_old[:,None] * o_i + tl.dot(exp_new,v_block)
        m_i = m_new
        # # 推进指针，取下一个K,V (advance会报错，好像只能advance列指针)
        # 这样子好像只有行指针可以advance
        # K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        # V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    o_i = o_i/l_i[:,None]
    l_i= m_i + tl.log(l_i) 

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od), # 跳一个seq的位置/一个d_k上的位置要多少字节
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 从哪里开始取
        block_shape=(Q_TILE_SIZE, D), # q_block取多大
        order=(1, 0), # 内存顺序
    )
    o_i_cast = o_i.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, o_i_cast)

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,), # 从哪里开始取
        block_shape=(Q_TILE_SIZE,), # q_block取多大
        order=(0,), # 内存顺序 wei
    )
    tl.store(L_block_ptr, l_i)