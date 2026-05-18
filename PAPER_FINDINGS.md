# Prime ChainSentinel — Paper Finding Notes

## Corrected Merkle finding

The initial Merkle prediction model used standalone hash throughput to estimate tree construction time as:

```text
t_naive = (N - 1) / T_h
```

where `N - 1` represents only the number of internal Merkle nodes. In the smoke benchmark, this naive model can significantly underestimate measured Merkle construction time. This is a relevant experimental finding because raw standalone hash throughput is not sufficient to predict end-to-end Merkle construction cost.

Part of the error comes from model incompleteness: the measured implementation hashes both the original transaction leaves and the internal Merkle nodes, giving approximately:

```text
total_hash_ops = N + (N - 1) = 2N - 1
```

The corrected operation-count model is therefore:

```text
t_corrected = (2N - 1) / T_h
```

After correcting for full pipeline hash count, the remaining residual error can be interpreted more defensibly as implementation overhead from Python dispatch, memory allocation, list traversal, cache behavior, and data movement.

## Suggested paper wording

The initial Merkle prediction model used standalone hash throughput to estimate tree construction time as `t = (N - 1) / T_h`, where `N - 1` represents the number of internal nodes. In the CPU smoke benchmark, this naive model underestimated measured Merkle construction time. This gap is a useful experimental finding because it shows that raw hash throughput alone does not capture end-to-end Merkle construction cost. Part of the error comes from model incompleteness: the measured implementation hashes both the original transaction leaves and the internal Merkle nodes, giving approximately `2N - 1` hash operations rather than only `N - 1`. After correcting for full pipeline hash count, the remaining residual error can be interpreted as implementation overhead from Python dispatch, memory allocation, list traversal, cache behavior, and data movement. This motivates comparing CPU and MI300X baselines not only as raw throughput measurements but as workload-level simulation profiles.

## GPU boundary wording

The current GPU benchmark does not claim native MI300X SHA-256, SHA3-256, or BLAKE3 throughput. It measures a PyTorch byte-parallel synthetic workload used as a simulation acceleration baseline. Native GPU cryptographic hash throughput would require dedicated HIP/CUDA/OpenCL kernels.
