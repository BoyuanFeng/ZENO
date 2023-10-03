# ZENO

ZENO is a type-based optimization framework for accelerating zero-knowledge neural network (zkNN) inference, published at ASPLOS'24. ZENO is a research prototype that is unaudited and should not be relied on in production.



## Software Dependencies

- Please follow this [instruction](https://www.rust-lang.org/tools/install) to install Rust.

- We test the code using `rustc 1.43.0`. Use `rustup override set 1.43.0` to specify the rust version for compilation.

- Since many Rust dependency packages may not be backward-compatible, we strongly recommend building the code with the provided `Cargo.lock`. Please refer to `run.sh` file for details.

## Files & Directory

- `Arkworks/`: A fork of the Arkworks implementation for ZK Proof, especially Groth16.
- `ZENO-engine/`: Implementation of ZENO optimizations.
- `zkNN-circuit/`: Implementation of zkNN circuits.

## Hardware Requirement

A laptop should be sufficient to generate proof for small neural networks such as LeNet.
Generating proof for large neural networks such as ResNet-50 may need a machine with sufficient (e.g., 256GB) RAM.

## Experiments

### Private Image & Public Weight Baseline

```
cd zkNN-circuit/baseline-one-private
sh ./run.sh
```

### Private Image & Private Weight Baseline

```
cd zkNN-circuit/baseline-both-private
sh ./run.sh
```

### Optimized zkNN with ZENO

```
cd zkNN-circuit/all-optimizations
sh ./run.sh
```
