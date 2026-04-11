import torch
import torch.nn as nn
import math
from einops import einsum
from torch import autograd
import triton
import triton.language as tl
from .kernel import flash_fwd_kernel

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Linear(nn.Module):

    def __init__(self,in_features,out_features,device=None,dtype=None):

        super().__init__()

        self.in_features=in_features
        self.out_features=out_features

        self.weight=nn.Parameter(torch.empty((out_features,in_features),device=device,dtype=dtype))

        sigma=math.sqrt(2/(in_features+out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma, 
            a=-3*sigma, 
            b=3*sigma
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return y

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        d_model: int, 模型的隐藏维度
        eps: float = 1e-5, 用于数值稳定性的 Epsilon 值
        '''
        super().__init__()
        # 根据 3.4.1 节要求，RMSNorm 的缩放参数初始化为 1
        # 使用 nn.Parameter 使其成为可学习参数
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 记录原始数据类型 (例如 torch.float16 或 torch.bfloat16)
        in_dtype = x.dtype
        # 2. 向上转型到 float32 以防平方运算溢出
        x = x.to(torch.float32)
        # 3. 计算均方值 (Mean Square)
        # 对最后一个维度 d_model 求平方后的平均值
        # keepdim=True 保证形状为 (batch, seq, 1)，便于后续广播计算
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        # 4. 计算归一化结果：x * (1 / sqrt(ms + eps))
        # torch.rsqrt 是 1/sqrt 的快捷且高效的写法
        # 同时乘以缩放参数 self.g (也转为 float32 参与计算)
        result = x * torch.rsqrt(ms + self.eps) * self.g.to(torch.float32)
        # 5. 将结果转回原始数据类型返回
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.w1_weight=nn.Parameter(torch.empty((d_ff,d_model), device=device, dtype=dtype))
        self.w2_weight=nn.Parameter(torch.empty((d_model,d_ff), device=device, dtype=dtype))
        self.w3_weight=nn.Parameter(torch.empty((d_ff,d_model), device=device, dtype=dtype))     
        sigma=math.sqrt(2/(d_ff+d_model))
        nn.init.trunc_normal_(
            self.w1_weight,
            mean=0.0,
            std=sigma, 
            a=-3*sigma, 
            b=3*sigma
        )
        nn.init.trunc_normal_(
            self.w2_weight,
            mean=0.0,
            std=sigma, 
            a=-3*sigma, 
            b=3*sigma
        )
        nn.init.trunc_normal_(
            self.w3_weight,
            mean=0.0,
            std=sigma, 
            a=-3*sigma, 
            b=3*sigma
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = einsum(x, self.w1_weight, "... d_in, d_out d_in -> ... d_out")
        op1=x1*torch.sigmoid(x1)
        op2=einsum(x, self.w3_weight, "... d_in, d_out d_in -> ... d_out")
        op3 = einsum(op1,op2,"..., ... -> ...")
        output = einsum(op3, self.w2_weight, "... d_in, d_out d_in -> ... d_out")
        return output

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        exponent=torch.arange(0,d_k,2).float()
        # 旋转角度
        freq=1/(theta**(exponent/d_k))
        # 生成位置
        position=torch.arange(max_seq_len) # 公式里的i
        # 总角度
        freqs=torch.outer(position,freq) # 外积
        # 在最后一个维度上，把每个元素重复 2 次
        freqs = torch.repeat_interleave(freqs, 2, dim=-1) 
        # 结果形状直接就是 (max_seq_len, d_k)，内容是 [θ0, θ0, θ1, θ1...]
        # 3. 计算 cos 和 sin 并调整形状
        cos_cached = torch.cos(freqs).reshape(max_seq_len, d_k)
        sin_cached = torch.sin(freqs).reshape(max_seq_len, d_k)
        # 4. Buffer
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        # 1. 查表：根据位置拿到这一 batch 对应的[位置]的 cos 和 sin
        # 变成: (1, 1, seq_len, head_dim)
        cos = self.cos_cached[token_positions].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[token_positions].unsqueeze(0).unsqueeze(0)
        # 偶数下标
        x_even = x[..., 0::2]
        # 奇数下标
        x_odd = x[..., 1::2]
        cos_even = cos[..., 0::2]
        sin_even = sin[..., 0::2]
        # 新下标
        x_rot_even = x_even * cos_even - x_odd * sin_even
        x_rot_odd  = x_even * sin_even + x_odd * cos_even
        out = torch.empty_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        return out

def softmax(x: torch.Tensor,dim=-1):
    m=x.amax(dim=dim, keepdim=True)
    x=x-m
    y=torch.exp(x)
    y_sum=y.sum(dim=dim,keepdim=True)
    out=y/y_sum
    return out

def logsumexp(x: torch.Tensor,dim=-1):
    m=x.amax(dim=dim, keepdim=True)
    x=x-m
    y=torch.exp(x)
    y_sum=y.sum(dim=dim,keepdim=True)
    out=torch.log(y_sum)+m
    return out

def Attention(q: torch.Tensor, k:torch.Tensor,v:torch.Tensor,mask=None):
    d_k = q.size(-1) 
    q_dot_k=einsum(q,k, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    scaled_score=q_dot_k/math.sqrt(d_k)
    if mask is not None:
        # 把 scaled_scores 中 mask 为 False 的地方替换为负无穷大 (AI写的)
        scaled_score=scaled_score.masked_fill(mask == False, float('-inf'))
    out=einsum(softmax(scaled_score,dim=-1),v,"... seq_q seq_k,... seq_k d_v -> ... seq_q d_v")
    return out

# 使用模块封装attention，使得能够编译

class AttnBlock(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q: (..., seq_q, d_k)
        # k: (..., seq_k, d_k)
        # v: (..., seq_k, d_v)
        d_k = q.size(-1)
        # (..., seq_q, seq_k)
        q_dot_k = torch.einsum("...qd,...kd->...qk", q, k)
        scaled_score = q_dot_k / math.sqrt(d_k)
        if mask is not None:
            scaled_score = scaled_score.masked_fill(~mask, float("-inf"))
        attn = softmax(scaled_score, dim=-1)
        # (..., seq_q, d_v)
        out = torch.einsum("...qk,...kd->...qd", attn, v)
        return out

class FastAttnTriton(autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v, is_causal=False):
        q_block_size = 32
        kv_block_size = 32
        # q: (..., seq_q, d_k)
        # k: (..., seq_k, d_k)
        # v: (..., seq_k, d_v)
        d_k = q.size(-1)
        seq_q = q.size(-2)
        seq_k = k.size(-2)
        # 缩放因子：scale
        scale = 1.0/math.sqrt(d_k)
        # 改：向上取整
        num_q_blocks = triton.cdiv(seq_q,q_block_size) # Q 被分成了几个块
        num_kv_blocks = triton.cdiv(seq_k,kv_block_size) # KV 被分成了几个块

        # 将 Batch 和 Head 维度展平
        q_flat = q.contiguous().view(-1,seq_q,d_k)
        k_flat = k.contiguous().view(-1,seq_k,d_k)
        v_flat = v.contiguous().view(-1,seq_k,d_k)

        group_num = q_flat.shape[0]

        # 初始化输出 O 和 logsumexp L
        # O保存最终结果，会作为累加起点先要被从零初始化
        o_flat = torch.zeros_like(q_flat)
        grid=(num_q_blocks,group_num)
        
        # L代表每组每个位置的logsumexp的和
        l_flat = torch.zeros((group_num, seq_q), device=q.device, dtype=torch.float32)

        # 调用 Triton Kernel
        flash_fwd_kernel[grid](
            q_flat, k_flat, v_flat,
            o_flat, l_flat,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
            l_flat.stride(0), l_flat.stride(1),
            seq_q, seq_k,
            scale,
            D=d_k,
            Q_TILE_SIZE=q_block_size,
            K_TILE_SIZE=kv_block_size,
            IS_CAUSAL=is_causal,
        )

        out = o_flat.view_as(q)
        l_flat = l_flat.view(*q.shape[:-1])
        ctx.is_causal = is_causal
        ctx.save_for_backward(q,k,v,out,l_flat)
        return out
    @staticmethod
    @torch.compile
    def backward(ctx,grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dO = grad_output
        d_k = Q.shape[-1]
        seq_q = Q.shape[-2]
        seq_k = K.shape[-2]
        D = (O * dO).sum(dim=-1)  # [B, nh, T]
        scale = 1.0 / math.sqrt(d_k)
        S = Q @ K.transpose(-2, -1) * scale


        if is_causal:
            # 生成一个右上角为 -inf 的掩码矩阵
            # 注意 diagonal=1 表示主对角线保留，从主对角线往上一格开始 mask
            mask = torch.triu(
                torch.full((seq_q, seq_k), float('-inf'), device=Q.device, dtype=S.dtype), 
                diagonal=1
            )
            S = S + mask
        

        P = torch.exp(S - L.unsqueeze(-1))
        dV = P.transpose(-2, -1) @ dO
        dP = dO @ V.transpose(-2, -1)
        dS = P * (dP - D.unsqueeze(-1))
        dQ = dS @ K * scale
        dK = dS.transpose(-2, -1) @ Q * scale
        return dQ, dK, dV, None


class FastAttn(autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v, is_causal=False):
        q_block_size = 32
        kv_block_size = 32
        # q: (..., seq_q, d_k)
        # k: (..., seq_k, d_k)
        # v: (..., seq_k, d_v)
        d_k = q.size(-1)
        seq_q = q.size(-2)
        seq_k = k.size(-2)
        # 缩放因子：scale
        scale = 1.0/math.sqrt(d_k)
        # 改：向上取整
        num_q_blocks = (seq_q + q_block_size - 1) // q_block_size # Q 被分成了几个块
        num_kv_blocks = (seq_k + kv_block_size - 1) // kv_block_size # KV 被分成了几个块

        # 将 Batch 和 Head 维度展平
        q_flat = q.contiguous().view(-1,seq_q,d_k)
        k_flat = k.contiguous().view(-1,seq_k,d_k)
        v_flat = v.contiguous().view(-1,seq_k,d_k)

        group_num = q_flat.shape[0]

        # 初始化输出 O 和 logsumexp L
        # O保存最终结果，会作为累加起点先要被从零初始化
        o_flat = torch.zeros_like(q_flat)
        # L代表每组每个位置的logsumexp的和
        l = torch.zeros((group_num, seq_q), device=q.device, dtype=q.dtype)
        m = torch.full((group_num, seq_q), -float('inf'), device=q.device, dtype=q.dtype)

        # 遍历所有Q
        for i in range(num_q_blocks):
            # 计算Q的起始位置
            q_start = i * q_block_size
            q_end = min(q_start + q_block_size, seq_q)
            # 当前q_block：从q_flat里面取
            # 由于 q_flat = q.contiguous().view(-1,seq_q,d_k)
            q_block = q_flat[:,q_start:q_end,:]
            # 当前Q对应的历史统计量
            # 由于o_flat = torch.zeros_like(q_flat)
            o_i = o_flat[:,q_start:q_end,:]
            # 由于l/m = torch.zeros((group_num, seq_q)
            l_i = l[:,q_start:q_end]
            m_i = m[:,q_start:q_end]
            # 内层循环，遍历所有KV分块
            for j in range(num_kv_blocks):
                # 取KV块
                kv_start = j * kv_block_size
                kv_end = min(kv_start + kv_block_size, seq_k)
                k_block = k_flat[:,kv_start:kv_end,:]
                v_block = v_flat[:,kv_start:kv_end,:]


                # QK^T
                s_ij = torch.einsum("...qd,...kd->...qk", q_block, k_block) * scale

                # mask
                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=s_ij.device).view(1, -1, 1)
                    kv_idx = torch.arange(kv_start, kv_end, device=s_ij.device).view(1, 1, -1)
                    causal_mask = q_idx >= kv_idx
                    s_ij = s_ij.masked_fill(~causal_mask, -float('inf')) 

                # 计算softmax
                # 1. 找目前块的行最大值: 对每个 Q token，求它对一整个 KV 块的最大值
                m_ij = torch.max(s_ij,dim=-1,keepdim=True)[0] # (group_num, q_block_size, 1)
                # 2. 找全局最大值 (squeeze的意思：把最后一维消除)
                m_new = torch.max(m_i, m_ij.squeeze(-1))
                # 3. 计算当前块指数和
                exp_new = torch.exp(s_ij - m_new.unsqueeze(-1))
                sum_exp_new = torch.sum(exp_new, dim=-1)
                # 4. 更新之前的内容：
                exp_old = torch.exp(m_i - m_new).unsqueeze(-1)
                # 更新全局指数和
                l_new = exp_old.squeeze(-1) * l_i + sum_exp_new
                # 更新输出O
                # o_new = exp_old*o_i + exp_new * V
                o_i = exp_old * o_i + torch.einsum("...qk,...kd->...qd", exp_new, v_block)
                # 5. 更新迭代量
                m_i=m_new
                l_i=l_new
            
            o_i=o_i/l_i.unsqueeze(-1)
            # 写回到最终结果o_flat里
            o_flat[:, q_start:q_end, :] = o_i
            # 为了反向传播
            m[:, q_start:q_end] = m_i
            l[:, q_start:q_end] = m_i + torch.log(l_i)
        out = o_flat.view_as(q)
        ctx.save_for_backward(l,q,k,v,out)
        return out



class MHA(nn.Module):
    def __init__(self,d_model:int,num_heads:int,use_rope=False,theta=None,max_seq_len=None,device=None,dtype=None):
        super().__init__()
        
        d_k=d_model//num_heads
        d_v=d_k

        self.num_heads=num_heads
        self.d_k=d_k
        self.d_model=d_model
        # x: (seq_len,d_model). qkv: (d_model, 3 * d_model)

        self.qkv_proj=Linear(d_model,3*d_model,device=device,dtype=dtype)
        self.o_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.use_rope=use_rope
        if use_rope:
            self.rope=RoPE(theta=theta,d_k=d_k,max_seq_len=max_seq_len)
    def forward(self,x:torch.Tensor,token_positions=None):
        batch_size, seq_len, _ = x.shape
        # 1. 映射得到 qkv
        # 形状: (batch_size, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)

        # 2. 拆分并重排维度
        # 目标是将 3 提到最前面，方便 chunk 拆分出 q, k, v
        # (B, S, 3, H, d_k) -> (3, B, H, S, d_k)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 应用RoPE
        if self.use_rope:
            q=self.rope(q,token_positions)
            k=self.rope(k,token_positions)
        # 此时 q, k, v 形状完美变为: (batch_size, num_heads, seq_len, d_k)
        # 定义因果掩码
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        causal_mask = torch.tril(causal_mask).unsqueeze(0).unsqueeze(0)

        # attn_out 形状: (batch_size, num_heads, seq_len, d_k)
        attn_out = Attention(q, k, v, mask=causal_mask)
        # 6. 把维度换回来: (batch_size, seq_len, num_heads, d_k)
        # 注意: PyTorch 中 transpose 后内存不再连续，调用 contiguous() 是为了后续能够使用 view()
        attn_out = attn_out.transpose(1, 2).contiguous()
        
        # 7. 拼接所有的头 (Flatten/Reshape)
        # 形状变为: (batch_size, seq_len, d_model)
        attn_out = attn_out.view(batch_size, seq_len, self.d_model)
        out = self.o_proj(attn_out)
        
        return out

def Cross_Entropy(inputs,targets):
    batch_size,_=inputs.shape
    rows=torch.arange(batch_size) 
    losses=logsumexp(inputs,dim=-1)-inputs[rows,targets]
    return losses.mean()

class Transformer_block(nn.Module):
    def __init__(self,d_model:int, num_heads:int,d_ff:int,use_rope=False,theta=None,max_seq_len=None,use_norm=True,device=None,dtype=None):
        super().__init__()
        

        self.Multihead_Attention=MHA(d_model=d_model,num_heads=num_heads,use_rope=use_rope,theta=theta,max_seq_len=max_seq_len,device=device,dtype=dtype)
        if use_norm:
            self.rmsnorm1=RMSNorm(d_model=d_model)
            self.rmsnorm2=RMSNorm(d_model=d_model)
        else:
            self.rmsnorm1=Identity()
            self.rmsnorm2=Identity()
        self.ffn=SwiGLU(d_model=d_model,d_ff=d_ff)

    def forward(self,x: torch.Tensor,token_positions=None) -> torch.Tensor:
        out1=self.Multihead_Attention(self.rmsnorm1(x),token_positions)
        out1=out1+x
        out2=self.ffn(self.rmsnorm2(out1))
        out=out1+out2
        return out
    
class TransformerLM(nn.Module):
    def __init__(self,vocab_size:int, context_length:int, num_layers:int, d_model:int, num_heads:int,d_ff:int,use_rope=False,use_norm=True,theta=None,device=None,dtype=None):
        super().__init__()
        self.input_embed=Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.use_rope=use_rope
        if not use_rope:
            self.pos_embed = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([
            Transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                use_rope=use_rope
            ) 
            for _ in range(num_layers)
        ])
        if use_norm:
            self.rmsnorm=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        else:
            self.rmsnorm=Identity()
        self.linear=Linear(in_features=d_model,out_features=vocab_size)
    
    def forward(self,x: torch.Tensor,token_positions=None) -> torch.Tensor:
        seq_len = x.size(1)

        # 1. 顶层统一处理位置信息 (兜底逻辑)
        if token_positions is None:
            # 生成 [0, 1, ..., seq_len-1] 的一维张量，并增加 batch 维度
            # 形状变为 (1, seq_len)，利用 PyTorch 的广播机制足够给下游使用
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x=self.input_embed(x)
        if not self.use_rope:
            x=self.pos_embed(x)
        for layer in self.layers:
            x=layer(x,token_positions)
        x=self.rmsnorm(x)
        x=self.linear(x)
        return x