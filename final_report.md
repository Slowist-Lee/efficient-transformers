# Attention Benchmark Report

`*.pickle` 文件可以拖入： https://docs.pytorch.org/memory_viz


```bash
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --stop-on-range-exit=true -o base_profile python attn_benchmark.py --mode base
```


| mode    |   d_model |   seq_len |   fwd_time(ms) |   bwd_time(ms) |   fwd_TFLOPS |   bwd_TFLOPS |   peak_mem(MB) | status   |
|:--------|----------:|----------:|---------------:|---------------:|-------------:|-------------:|---------------:|:---------|
| compile |        16 |       256 |           0.23 |           0.55 |         0.15 |         0.15 |          23.39 | Success  |
| compile |        16 |      1024 |           0.31 |           0.31 |         1.75 |         4.32 |         116.81 | Success  |
| base    |        16 |      4096 |           4.39 |          12.91 |         1.96 |         1.66 |        3614.5  | Success  |
| compile |        16 |      4096 |           1.73 |           3.25 |         4.96 |         6.61 |        1570.5  | Success  |
| flash   |        16 |      4096 |           0.36 |           3.95 |        24.12 |         5.43 |        1558.38 | Success  |
| compile |        16 |      8192 |           6.75 |          12.74 |         5.09 |         6.74 |        6196.75 | Success  |
| compile |        16 |     16384 |         nan    |         nan    |       nan    |       nan    |         nan    | OOM      |
| compile |        32 |       256 |           0.37 |           0.46 |         0.18 |         0.36 |          69.52 | Success  |
| compile |        32 |      1024 |           0.27 |           0.35 |         3.98 |         7.68 |         121.31 | Success  |
| base    |        32 |      4096 |           4.41 |          12.93 |         3.89 |         3.32 |        3628.5  | Success  |
| compile |        32 |      4096 |           2.35 |           3.84 |         7.3  |        11.18 |        1588.5  | Success  |
| flash   |        32 |      4096 |           0.5  |           3.86 |        34.08 |        11.14 |        1572.38 | Success  |
| compile |        32 |      8192 |           9.07 |          16.63 |         7.58 |        10.33 |        6232.75 | Success  |
| compile |        32 |     16384 |         nan    |         nan    |       nan    |       nan    |         nan    | OOM      |
| compile |        64 |       256 |           0.27 |           0.39 |         0.5  |         0.86 |         118.77 | Success  |
| compile |        64 |      1024 |           0.28 |           0.38 |         7.69 |        14.04 |         130.31 | Success  |
| base    |        64 |      4096 |           4.58 |          13.31 |         7.5  |         6.45 |        3656.5  | Success  |
| compile |        64 |      4096 |           1.91 |           3.63 |        18.03 |        23.67 |        1624.5  | Success  |
| flash   |        64 |      4096 |           0.69 |           4.17 |        49.57 |        20.59 |        1600.38 | Success  |
| compile |        64 |      8192 |           7.02 |          13.01 |        19.59 |        26.41 |        6304.75 | Success  |
| compile |        64 |     16384 |         nan    |         nan    |       nan    |       nan    |         nan    | OOM      |
| compile |       128 |       256 |           0.32 |           0.4  |         0.85 |         1.7  |         217.27 | Success  |
| compile |       128 |      1024 |           0.37 |           0.54 |        11.47 |        19.97 |         148.31 | Success  |
| base    |       128 |      4096 |           5.51 |          15.04 |        12.47 |        11.42 |        3712.5  | Success  |
| compile |       128 |      4096 |           2.92 |           5.44 |        23.53 |        31.59 |        1696.5  | Success  |
| flash   |       128 |      4096 |           1.17 |           6.59 |        58.91 |        26.07 |        1656.38 | Success  |
| compile |       128 |      8192 |          11.37 |          20.75 |        24.17 |        33.12 |        6448.75 | Success  |
| compile |       128 |     16384 |         nan    |         nan    |       nan    |       nan    |         nan    | OOM      |

细节分析：

Flash:

```bash
(assignment1-basics) ➜  assignment1-basics git:(main)
Generating SQLite file flash_profile.sqlite from flash_profile.nsys-rep
Processing [flash_profile.sqlite] with [/usr/local/cuda-12.8/nsight-systems-2024.6.2/host-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  --------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     29.4    2,102,607,084        904  2,325,892.8  1,396,334.5     2,144  22,648,082  4,048,607.1  triton_poi_fused_exp_mul_sub_unsqueeze_2                                                            
     14.7    1,050,165,479      1,362    771,046.6    375,727.5     2,784   8,976,476  1,355,644.4  void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (i…
     12.5      893,896,395        454  1,968,934.8  1,096,591.5     7,232  18,979,196  3,449,580.6  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)                       
     11.8      841,492,766      1,356    620,569.9    423,056.0     7,360   5,879,518  1,000,112.1  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nt_align1>(T1::Params)                        
     11.2      802,573,646        908    883,891.7    235,647.5     4,064  16,207,965  1,815,672.1  flash_fwd_kernel                                                                                    
      8.4      601,524,182        452  1,330,805.7    795,456.0     8,512  13,748,732  2,260,544.0  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nt_align1>(T1::Params)                       
      6.6      471,536,877        678    695,482.1    459,295.0     8,896   6,798,621  1,142,753.0  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_align1>(T1::Params)                        
      4.6      327,680,997        226  1,449,915.9    864,768.0     8,864  14,324,956  2,489,416.6  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(T1::Params)                       
      0.3       21,379,381      2,352      9,089.9      4,000.0       832      65,376     12,694.9  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::arr…
      0.1       10,061,881      5,088      1,977.6        768.0       640      39,552      2,781.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<c…
      0.1        9,709,946      1,356      7,160.7      4,064.0     1,120      59,520      7,730.7  triton_poi_fused_3                                                                                  
      0.1        6,154,717        681      9,037.8      8,608.0     1,216      43,360      8,488.6  triton_red_fused_mul_sum_1                                                                          
      0.1        5,596,959      1,702      3,288.5      2,208.0       736      30,304      3,490.7  triton_poi_fused_bmm_expand_transpose_0                                                             
      0.1        4,700,443        908      5,176.7      3,840.0     3,424      44,544      3,646.7  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapp…
      0.0          823,359        452      1,821.6      1,568.0       768       6,976      1,318.9  triton_poi_fused_mul_3                                                                              
      0.0          520,128        227      2,291.3      2,528.0       832       7,488      1,626.0  triton_per_fused_mul_sum_1                                                                          
      0.0          403,744         60      6,729.1      4,000.0     1,152      47,104      8,280.5  void at::native::<unnamed>::distribution_elementwise_grid_stride_kernel<float, (int)4, void at::nat…
      0.0           79,840        110        725.8        736.0       704         928         25.2  triton_poi_fused_bmm_transpose_0                            
```

Base:

```bash
(assignment1-basics) ➜  assignment1-basics git:(main) ✗ nsys stats --report cuda_gpu_kern_sum base_profile.nsys-rep
Generating SQLite file base_profile.sqlite from base_profile.nsys-rep
Processing [base_profile.sqlite] with [/usr/local/cuda-12.8/nsight-systems-2024.6.2/host-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  --------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     15.6    3,092,298,097      3,520    878,493.8    363,087.5     1,984   3,368,602  1,133,960.2  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
     11.4    2,258,918,481      1,760  1,283,476.4    519,039.5     1,504   4,097,276  1,670,190.9  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, floa…
      8.1    1,599,898,050      1,776    900,843.5    691,102.5     1,312   3,298,875  1,162,251.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<float, float, floa…
      7.9    1,569,106,024      2,560    612,932.0     16,928.0     1,216   3,343,004  1,021,916.0  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      7.8    1,554,772,387      1,760    883,393.4    360,959.5     1,312   2,824,026  1,149,405.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::neg_kernel_cuda(at::TensorIterat…
      5.7    1,139,777,089      2,448    465,595.2      7,168.0       896   4,083,867  1,170,677.9  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::arr…
      5.5    1,101,267,282      3,520    312,860.0      8,288.0     1,695   1,475,679    494,670.8  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapp…
      4.9      971,182,061      1,100    882,892.8    866,638.0     2,560   1,780,731    781,776.6  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator…
      3.9      781,423,047      1,323    590,644.8    391,551.0     2,976   8,651,031    817,734.6  void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (i…
      3.9      774,315,729        880    879,904.2    351,295.5     1,376   3,280,313  1,152,485.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIterat…
      3.8      758,477,956      1,100    689,525.4    196,799.5     8,416   3,236,285    954,673.9  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nt_align1>(T1::Params)                       
      3.6      715,826,454      1,320    542,292.8    239,071.5     8,864   2,402,171    670,833.6  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_align1>(T1::Params)                        
      3.6      713,332,261      1,100    648,483.9    659,278.5     2,112   1,324,510    571,813.0  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<long, at::native::func_wrappe…
      3.2      642,403,853        441  1,456,698.1  1,117,470.0     7,296  17,952,596  1,968,709.1  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)                       
      2.7      533,917,709        880    606,724.7    247,407.5     2,944   1,933,308    786,372.1  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::…
      2.4      474,005,930        440  1,077,286.2    464,703.5     8,832   3,437,277  1,355,481.6  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(T1::Params)                       
      2.4      468,901,558        880    532,842.7    222,351.5     1,856   1,687,484    684,411.0  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      1.9      385,461,300        896    430,202.3    326,784.0     1,824   1,517,983    542,295.0  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapp…
      1.6      319,589,001        660    484,225.8    219,247.5     7,488   2,021,980    599,732.3  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nt_align1>(T1::Params)                        
      0.0        8,096,952      1,856      4,362.6      2,800.0       832      75,648      6,075.8  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      0.0        6,367,893      3,232      1,970.3        832.0       672      15,328      2,270.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<c…
      0.0        2,854,905        880      3,244.2      3,264.0     2,623       4,032        424.1  void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::nat…
      0.0          410,367         60      6,839.4      3,872.0     1,120      51,840      9,009.9  void at::native::<unnamed>::distribution_elementwise_grid_stride_kernel<float, (int)4, void at::nat…
```