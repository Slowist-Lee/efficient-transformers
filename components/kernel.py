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
        o_i = exp_old[:,None] * o_i + tl.dot(exp_new.to(v_block.dtype), v_block)
        m_i = m_new
        # # 推进指针，取下一个K,V (advance会结果出错，所以好像只能advance列指针)
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
        order=(0,), # 内存顺序
    )
    tl.store(L_block_ptr, l_i)

@triton.jit
def flash_bwd_kernel_dq(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr, dO_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
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

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od), # 跳一个seq的位置/一个d_k上的位置要多少字节
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 从哪里开始取
        block_shape=(Q_TILE_SIZE, D), # q_block取多大
        order=(1, 0), # 内存顺序
    )
    O_block = tl.load(O_block_ptr).to(tl.float32)

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod), # 跳一个seq的位置/一个d_k上的位置要多少字节
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 从哪里开始取
        block_shape=(Q_TILE_SIZE, D), # q_block取多大
        order=(1, 0), # 内存顺序
    )
    dO_block = tl.load(dO_block_ptr)
    mul = O_block * dO_block.to(tl.float32)
    D_block = tl.sum(mul, axis=1) # 计算 D = O * dO
    
    dQ = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd), # 跳一个seq的位置/一个d_k上的位置要多少字节
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 从哪里开始取
        block_shape=(Q_TILE_SIZE, D), # q_block取多大
        order=(1, 0), # 内存顺序
    )

    q_block = tl.load(Q_block_ptr)

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,), # 从哪里开始取
        block_shape=(Q_TILE_SIZE,), # q_block取多大
        order=(0,), # 内存顺序
    )
    L_block = tl.load(L_block_ptr).to(tl.float32)

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
        # 2. 计算S
        s_ij = tl.dot(q_block, k_block.T).to(tl.float32) * scale # 强制将结果转为 float32 以防溢出
    
        # 处理mask
        if IS_CAUSAL:
            q_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_idx = tl.arange(0, K_TILE_SIZE) + j
            mask = q_idx[:, None] >= k_idx[None, :]
            s_ij = tl.where(mask, s_ij, -float("inf"))  # 被mask的地方设为 -inf

        # 3. 计算
        p_ij = tl.exp(s_ij - L_block[:,None])

        dP = tl.dot(dO_block, v_block.T).to(tl.float32)
        dS = p_ij * (dP - D_block[:, None]) * scale
        dQ += tl.dot(dS.to(k_block.dtype), k_block)

    # 最后写回全局dQ
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )
    tl.store(dQ_block_ptr, dQ.to(dQ_block_ptr.type.element_ty))

@triton.jit
def flash_bwd_kernel_dk_dv(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    kv_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    offs_k = kv_tile_index * K_TILE_SIZE

    # 1. 循环外：加载 K, V，并在 SRAM 初始化 dK, dV
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb, shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(offs_k, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(offs_k, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )
    k_block = tl.load(K_block_ptr)
    v_block = tl.load(V_block_ptr)

    dK_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # Causal 优化：只需要从对角线所在的 Q 块开始遍历到最后
    start_q = offs_k if IS_CAUSAL else 0

    # 2. 循环内：遍历 Q 块
    for i in range(start_q, N_QUERIES, Q_TILE_SIZE):
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
            offsets=(i, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob, shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
            offsets=(i, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob, shape=(N_QUERIES, D), strides=(stride_doq, stride_dod),
            offsets=(i, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb, shape=(N_QUERIES,), strides=(stride_lq,),
            offsets=(i,), block_shape=(Q_TILE_SIZE,), order=(0,)
        )

        q_block = tl.load(Q_block_ptr)
        O_block = tl.load(O_block_ptr).to(tl.float32)
        dO_block = tl.load(dO_block_ptr)
        L_block = tl.load(L_block_ptr).to(tl.float32)

        D_i = tl.sum(O_block * dO_block.to(tl.float32), axis=1)
        s_ij = tl.dot(q_block, k_block.T).to(tl.float32) * scale

        if IS_CAUSAL:
            q_idx = tl.arange(0, Q_TILE_SIZE) + i
            k_idx = tl.arange(0, K_TILE_SIZE) + offs_k
            mask = q_idx[:, None] >= k_idx[None, :]
            s_ij = tl.where(mask, s_ij, -float("inf"))

        p_ij = tl.exp(s_ij - L_block[:, None])
        
        # SRAM 累加 dV
        dV_block += tl.dot(p_ij.T.to(dO_block.dtype), dO_block)
        
        dP = tl.dot(dO_block, v_block.T).to(tl.float32)
        dS = p_ij * (dP - D_i[:, None]) * scale

        # SRAM 累加 dK
        dK_block += tl.dot(dS.T, q_block.to(tl.float32))

    # 3. 循环结束：一次性使用块指针将 dK, dV 写回全局内存
    dK_out_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb, shape=(N_KEYS, D), strides=(stride_dkk, stride_dkd),
        offsets=(offs_k, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )
    tl.store(dK_out_block_ptr, dK_block.to(dK_out_block_ptr.type.element_ty))

    dV_out_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb, shape=(N_KEYS, D), strides=(stride_dvk, stride_dvd),
        offsets=(offs_k, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )
    tl.store(dV_out_block_ptr, dV_block.to(dV_out_block_ptr.type.element_ty))