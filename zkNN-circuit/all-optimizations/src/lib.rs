#[macro_use]
extern crate algebra_core;
extern crate groth16;
extern crate r1cs_core;
extern crate r1cs_std;
extern crate rand;
extern crate rand_chacha;
extern crate rand_xorshift;


pub mod demo_circuit;
pub mod argmax_circuit;
pub mod avg_pool_circuit;
pub mod commit_circuit;
pub mod conv_circuit;
pub mod cosine_circuit;
pub mod full_circuit;
pub mod lenet_circuit;
pub mod mul_circuit;
pub mod pedersen_commit;
pub mod read_inputs;
pub mod relu_circuit;
pub mod vgg_circuit;
pub mod resnet18_circuit;
pub mod resnet18_both_private_circuit;
pub mod resnet50_circuit;
pub mod vgg_both_private_circuit;
pub mod both_private_circuit;
pub mod residual_add_circuit;
pub mod vanilla;


//=======================
// dimensions
//=======================
pub(crate) const M: usize = 128;
pub(crate) const N: usize = 10;

//should be consistent
//pub(crate) const SIMD_5VEC_EXTRA_BITS: u32 = 3; //not used in our implementation
pub(crate) const SIMD_4VEC_EXTRA_BITS: u32 = 12; //in case the long vector dot product overflow. 12 can hold at least vector of length 2^12
pub(crate) const SIMD_3VEC_EXTRA_BITS: u32 = 20;
//pub(crate) const SIMD_2VEC_EXTRA_BITS: u32 = 68;
pub(crate) const M_EXP: u32 = 22;

pub(crate) const SIMD_BOTTLENECK: usize = 210;
//=======================
// data
//=======================

pub const FACE_HEIGHT: usize = 46;
pub const FACE_HEIGHT_FC1: usize = 5;
pub const FACE_WIDTH: usize = 56;
pub const FACE_WIDTH_FC1: usize = 8;

//=======================
// Commitments
//=======================
pub type Commit = [u8; 32];
pub type Open = [u8; 32];
