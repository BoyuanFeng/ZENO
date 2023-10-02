use algebra::ed_on_bls12_381::*;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::time::Instant;
use zk_ml_knit_encoding::pedersen_commit::*;
use zk_ml_knit_encoding::read_inputs::*;
use zk_ml_knit_encoding::vanilla::*;
use zk_ml_knit_encoding::resnet50_circuit::*;

fn rand_4d_vec_generator(a: usize, b: usize, c: usize, d: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    let mut res = vec![vec![vec![vec![0; d]; c]; b]; a];

    res
}

fn rand_2d_vec_generator(a: usize, b: usize) -> Vec<Vec<u8>> {
    let mut res = vec![vec![2; b]; a];

    res
}

fn main() {
    let mut rng = rand::thread_rng();

    println!("RESNET18 testing. use null parameters and input just to benchmark performance");

    let x = rand_4d_vec_generator(1, 3, 32, 32);


    let conv21_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 3, 1, 1);
    let conv22_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 16, 3, 3);
    let conv23_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 16, 1, 1);
    let conv24_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 64, 1, 1);
    let conv25_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 16, 3, 3);
    let conv26_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 16, 1, 1);
    let conv27_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 64, 1, 1);
    let conv28_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 16, 3, 3);
    let conv29_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 16, 1, 1);
    let multiplier_conv21: Vec<f32> = vec![1.5; 16];
    let multiplier_conv22: Vec<f32> = vec![1.5; 16];
    let multiplier_conv23: Vec<f32> = vec![1.5; 64];
    let multiplier_conv24: Vec<f32> = vec![1.5; 16];
    let multiplier_conv25: Vec<f32> = vec![1.5; 16];
    let multiplier_conv26: Vec<f32> = vec![1.5; 64];
    let multiplier_conv27: Vec<f32> = vec![1.5; 16];
    let multiplier_conv28: Vec<f32> = vec![1.5; 16];
    let multiplier_conv29: Vec<f32> = vec![1.5; 64];

    let conv31_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 64, 1, 1);
    let conv32_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 32, 3, 3);
    let conv33_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 32, 1, 1);
    let conv34_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 128, 1, 1);
    let conv35_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 32, 3, 3);
    let conv36_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 32, 1, 1);
    let conv37_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 128, 1, 1);
    let conv38_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 32, 3, 3);
    let conv39_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 32, 1, 1);
    let conv3_10_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 128, 1, 1);
    let conv3_11_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 32, 3, 3);
    let conv3_12_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 32, 1, 1);
    let multiplier_conv31: Vec<f32> = vec![1.5; 32];
    let multiplier_conv32: Vec<f32> = vec![1.5; 32];
    let multiplier_conv33: Vec<f32> = vec![1.5; 128];
    let multiplier_conv34: Vec<f32> = vec![1.5; 32];
    let multiplier_conv35: Vec<f32> = vec![1.5; 32];
    let multiplier_conv36: Vec<f32> = vec![1.5; 128];
    let multiplier_conv37: Vec<f32> = vec![1.5; 32];
    let multiplier_conv38: Vec<f32> = vec![1.5; 32];
    let multiplier_conv39: Vec<f32> = vec![1.5; 128];
    let multiplier_conv3_10: Vec<f32> = vec![1.5; 32];
    let multiplier_conv3_11: Vec<f32> = vec![1.5; 32];
    let multiplier_conv3_12: Vec<f32> = vec![1.5; 128];

    let conv41_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 128, 1, 1);
    let conv42_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 64, 3, 3);
    let conv43_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 64, 1, 1);
    let conv44_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 256, 1, 1);
    let conv45_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 64, 3, 3);
    let conv46_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 64, 1, 1);
    let conv47_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 256, 1, 1);
    let conv48_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 64, 3, 3);
    let conv49_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 64, 1, 1);
    let conv4_10_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 256, 1, 1);
    let conv4_11_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 64, 3, 3);
    let conv4_12_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 64, 1, 1);
    let conv4_13_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 256, 1, 1);
    let conv4_14_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 64, 3, 3);
    let conv4_15_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 64, 1, 1);
    let conv4_16_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 256, 1, 1);
    let conv4_17_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 64, 3, 3);
    let conv4_18_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 64, 1, 1);
    let multiplier_conv41: Vec<f32> = vec![1.5; 64];
    let multiplier_conv42: Vec<f32> = vec![1.5; 64];
    let multiplier_conv43: Vec<f32> = vec![1.5; 256];
    let multiplier_conv44: Vec<f32> = vec![1.5; 64];
    let multiplier_conv45: Vec<f32> = vec![1.5; 64];
    let multiplier_conv46: Vec<f32> = vec![1.5; 256];
    let multiplier_conv47: Vec<f32> = vec![1.5; 64];
    let multiplier_conv48: Vec<f32> = vec![1.5; 64];
    let multiplier_conv49: Vec<f32> = vec![1.5; 256];
    let multiplier_conv4_10: Vec<f32> = vec![1.5; 64];
    let multiplier_conv4_11: Vec<f32> = vec![1.5; 64];
    let multiplier_conv4_12: Vec<f32> = vec![1.5; 256];
    let multiplier_conv4_13: Vec<f32> = vec![1.5; 64];
    let multiplier_conv4_14: Vec<f32> = vec![1.5; 64];
    let multiplier_conv4_15: Vec<f32> = vec![1.5; 256];
    let multiplier_conv4_16: Vec<f32> = vec![1.5; 64];
    let multiplier_conv4_17: Vec<f32> = vec![1.5; 64];
    let multiplier_conv4_18: Vec<f32> = vec![1.5; 256];

    let conv51_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 256, 1, 1);
    let conv52_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 128, 3, 3);
    let conv53_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(512, 128, 1, 1);
    let conv54_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 512, 3, 3);
    let conv55_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 128, 1, 1);
    let conv56_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(512, 128, 3, 3);
    let conv57_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 512, 1, 1);
    let conv58_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 128, 3, 3);
    let conv59_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(512, 128, 1, 1);
    let multiplier_conv51: Vec<f32> = vec![1.5; 128];
    let multiplier_conv52: Vec<f32> = vec![1.5; 128];
    let multiplier_conv53: Vec<f32> = vec![1.5; 512];
    let multiplier_conv54: Vec<f32> = vec![1.5; 128];
    let multiplier_conv55: Vec<f32> = vec![1.5; 128];
    let multiplier_conv56: Vec<f32> = vec![1.5; 512];
    let multiplier_conv57: Vec<f32> = vec![1.5; 128];
    let multiplier_conv58: Vec<f32> = vec![1.5; 128];
    let multiplier_conv59: Vec<f32> = vec![1.5; 512];

    // let fc1_w: Vec<Vec<u8>> = rand_2d_vec_generator(1024, 1024);
    let fc1_w: Vec<Vec<u8>> = rand_2d_vec_generator(10, 512);
    // let multiplier_fc1: Vec<f32> = vec![1.5; 1024];
    let multiplier_fc1: Vec<f32> = vec![1.5; 10];


    let x_0: Vec<u8> = vec![128];

    let conv21_output_0: Vec<u8> = vec![128];
    let conv22_output_0: Vec<u8> = vec![128];
    let conv23_output_0: Vec<u8> = vec![128];
    let conv24_output_0: Vec<u8> = vec![128];
    let conv25_output_0: Vec<u8> = vec![128];
    let conv26_output_0: Vec<u8> = vec![128];
    let conv27_output_0: Vec<u8> = vec![128];
    let conv28_output_0: Vec<u8> = vec![128];
    let conv29_output_0: Vec<u8> = vec![128];

    let conv31_output_0: Vec<u8> = vec![128];
    let conv32_output_0: Vec<u8> = vec![128];
    let conv33_output_0: Vec<u8> = vec![128];
    let conv34_output_0: Vec<u8> = vec![128];
    let conv35_output_0: Vec<u8> = vec![128];
    let conv36_output_0: Vec<u8> = vec![128];
    let conv37_output_0: Vec<u8> = vec![128];
    let conv38_output_0: Vec<u8> = vec![128];
    let conv39_output_0: Vec<u8> = vec![128];
    let conv3_10_output_0: Vec<u8> = vec![128];
    let conv3_11_output_0: Vec<u8> = vec![128];
    let conv3_12_output_0: Vec<u8> = vec![128];

    let conv41_output_0: Vec<u8> = vec![128];
    let conv42_output_0: Vec<u8> = vec![128];
    let conv43_output_0: Vec<u8> = vec![128];
    let conv44_output_0: Vec<u8> = vec![128];
    let conv45_output_0: Vec<u8> = vec![128];
    let conv46_output_0: Vec<u8> = vec![128];
    let conv47_output_0: Vec<u8> = vec![128];
    let conv48_output_0: Vec<u8> = vec![128];
    let conv49_output_0: Vec<u8> = vec![128];
    let conv4_10_output_0: Vec<u8> = vec![128];
    let conv4_11_output_0: Vec<u8> = vec![128];
    let conv4_12_output_0: Vec<u8> = vec![128];
    let conv4_13_output_0: Vec<u8> = vec![128];
    let conv4_14_output_0: Vec<u8> = vec![128];
    let conv4_15_output_0: Vec<u8> = vec![128];
    let conv4_16_output_0: Vec<u8> = vec![128];
    let conv4_17_output_0: Vec<u8> = vec![128];
    let conv4_18_output_0: Vec<u8> = vec![128];

    let conv51_output_0: Vec<u8> = vec![128];
    let conv52_output_0: Vec<u8> = vec![128];
    let conv53_output_0: Vec<u8> = vec![128];
    let conv54_output_0: Vec<u8> = vec![128];
    let conv55_output_0: Vec<u8> = vec![128];
    let conv56_output_0: Vec<u8> = vec![128];
    let conv57_output_0: Vec<u8> = vec![128];
    let conv58_output_0: Vec<u8> = vec![128];
    let conv59_output_0: Vec<u8> = vec![128];

    let fc1_output_0: Vec<u8> = vec![128];

    let conv21_weights_0: Vec<u8> = vec![128];
    let conv22_weights_0: Vec<u8> = vec![128];
    let conv23_weights_0: Vec<u8> = vec![128];
    let conv24_weights_0: Vec<u8> = vec![128];
    let conv25_weights_0: Vec<u8> = vec![128];
    let conv26_weights_0: Vec<u8> = vec![128];
    let conv27_weights_0: Vec<u8> = vec![128];
    let conv28_weights_0: Vec<u8> = vec![128];
    let conv29_weights_0: Vec<u8> = vec![128];

    let conv31_weights_0: Vec<u8> = vec![128];
    let conv32_weights_0: Vec<u8> = vec![128];
    let conv33_weights_0: Vec<u8> = vec![128];
    let conv34_weights_0: Vec<u8> = vec![128];
    let conv35_weights_0: Vec<u8> = vec![128];
    let conv36_weights_0: Vec<u8> = vec![128];
    let conv37_weights_0: Vec<u8> = vec![128];
    let conv38_weights_0: Vec<u8> = vec![128];
    let conv39_weights_0: Vec<u8> = vec![128];
    let conv3_10_weights_0: Vec<u8> = vec![128];
    let conv3_11_weights_0: Vec<u8> = vec![128];
    let conv3_12_weights_0: Vec<u8> = vec![128];

    let conv41_weights_0: Vec<u8> = vec![128];
    let conv42_weights_0: Vec<u8> = vec![128];
    let conv43_weights_0: Vec<u8> = vec![128];
    let conv44_weights_0: Vec<u8> = vec![128];
    let conv45_weights_0: Vec<u8> = vec![128];
    let conv46_weights_0: Vec<u8> = vec![128];
    let conv47_weights_0: Vec<u8> = vec![128];
    let conv48_weights_0: Vec<u8> = vec![128];
    let conv49_weights_0: Vec<u8> = vec![128];
    let conv4_10_weights_0: Vec<u8> = vec![128];
    let conv4_11_weights_0: Vec<u8> = vec![128];
    let conv4_12_weights_0: Vec<u8> = vec![128];
    let conv4_13_weights_0: Vec<u8> = vec![128];
    let conv4_14_weights_0: Vec<u8> = vec![128];
    let conv4_15_weights_0: Vec<u8> = vec![128];
    let conv4_16_weights_0: Vec<u8> = vec![128];
    let conv4_17_weights_0: Vec<u8> = vec![128];
    let conv4_18_weights_0: Vec<u8> = vec![128];

    let conv51_weights_0: Vec<u8> = vec![128];
    let conv52_weights_0: Vec<u8> = vec![128];
    let conv53_weights_0: Vec<u8> = vec![128];
    let conv54_weights_0: Vec<u8> = vec![128];
    let conv55_weights_0: Vec<u8> = vec![128];
    let conv56_weights_0: Vec<u8> = vec![128];
    let conv57_weights_0: Vec<u8> = vec![128];
    let conv58_weights_0: Vec<u8> = vec![128];
    let conv59_weights_0: Vec<u8> = vec![128];

    let fc1_weights_0: Vec<u8> = vec![128];


    println!("finish reading parameters");

    let conv_residual1_weight: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 3, 1, 1);
    let conv_residual1_output_0:Vec<u8> = vec![128];
    let conv_residual1_weights_0: Vec<u8> = vec![128];
    let conv_residual1_multiplier: Vec<f32> = vec![1.5; 64];

    let conv_residual4_weight: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 64, 1, 1);
    let conv_residual4_output_0:Vec<u8> = vec![128];
    let conv_residual4_weights_0: Vec<u8> = vec![128];
    let conv_residual4_multiplier: Vec<f32> = vec![1.5; 128];

    let conv_residual8_weight: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 128, 1, 1);
    let conv_residual8_output_0:Vec<u8> = vec![128];
    let conv_residual8_weights_0: Vec<u8> = vec![128];
    let conv_residual8_multiplier: Vec<f32> = vec![1.5; 256];

    let conv_residual11_weight: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(256, 256, 1, 1);
    let conv_residual11_output_0:Vec<u8> = vec![128];
    let conv_residual11_weights_0: Vec<u8> = vec![128];
    let conv_residual11_multiplier: Vec<f32> = vec![1.5; 256];

    let conv_residual14_weight: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(512, 256, 1, 1);
    let conv_residual14_output_0:Vec<u8> = vec![128];
    let conv_residual14_weights_0: Vec<u8> = vec![128];
    let conv_residual14_multiplier: Vec<f32> = vec![1.5; 512];

    let add_residual1_output_0:Vec<u8> = vec![128];
    let add_residual2_output_0:Vec<u8> = vec![128];
    let add_residual3_output_0:Vec<u8> = vec![128];
    let add_residual4_output_0:Vec<u8> = vec![128];
    let add_residual5_output_0:Vec<u8> = vec![128];
    let add_residual6_output_0:Vec<u8> = vec![128];
    let add_residual7_output_0:Vec<u8> = vec![128];
    let add_residual8_output_0:Vec<u8> = vec![128];
    let add_residual9_output_0:Vec<u8> = vec![128];
    let add_residual10_output_0:Vec<u8> = vec![128];
    let add_residual11_output_0:Vec<u8> = vec![128];
    let add_residual12_output_0:Vec<u8> = vec![128];
    let add_residual13_output_0:Vec<u8> = vec![128];
    let add_residual14_output_0:Vec<u8> = vec![128];
    let add_residual15_output_0:Vec<u8> = vec![128];
    let add_residual16_output_0:Vec<u8> = vec![128];

    let add_residual1_first_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual2_first_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual3_first_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual4_first_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual5_first_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual6_first_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual7_first_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual8_first_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual9_first_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual10_first_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual11_first_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual12_first_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual13_first_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual14_first_multiplier: Vec<f32> = vec![1.5; 512];
    let add_residual15_first_multiplier: Vec<f32> = vec![1.5; 512];
    let add_residual16_first_multiplier: Vec<f32> = vec![1.5; 512];

    let add_residual1_second_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual2_second_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual3_second_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual4_second_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual5_second_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual6_second_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual7_second_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual8_second_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual9_second_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual10_second_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual11_second_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual12_second_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual13_second_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual14_second_multiplier: Vec<f32> = vec![1.5; 512];
    let add_residual15_second_multiplier: Vec<f32> = vec![1.5; 512];
    let add_residual16_second_multiplier: Vec<f32> = vec![1.5; 512];
    
    println!("Begin forwarding.");

    let z: Vec<Vec<u8>> = resnet50_circuit_forward_u8(
        1, //padding
        x.clone(),

        conv21_w.clone(),
        conv22_w.clone(),
        conv23_w.clone(),
        conv24_w.clone(),
        conv25_w.clone(),
        conv26_w.clone(),
        conv27_w.clone(),
        conv28_w.clone(),
        conv29_w.clone(),
        // ----------------------------- --
        conv31_w.clone(),
        conv32_w.clone(),
        conv33_w.clone(),
        conv34_w.clone(),
        conv35_w.clone(),
        conv36_w.clone(),
        conv37_w.clone(),
        conv38_w.clone(),
        conv39_w.clone(),
        conv3_10_w.clone(),
        conv3_11_w.clone(),
        conv3_12_w.clone(),

        conv41_w.clone(),
        conv42_w.clone(),
        conv43_w.clone(),
        conv44_w.clone(),
        conv45_w.clone(),
        conv46_w.clone(),
        conv47_w.clone(),
        conv48_w.clone(),
        conv49_w.clone(),
        conv4_10_w.clone(),
        conv4_11_w.clone(),
        conv4_12_w.clone(),
        conv4_13_w.clone(),
        conv4_14_w.clone(),
        conv4_15_w.clone(),
        conv4_16_w.clone(),
        conv4_17_w.clone(),
        conv4_18_w.clone(),
    
        //------------------------------ --
        conv51_w.clone(),
        conv52_w.clone(),
        conv53_w.clone(),
        conv54_w.clone(),
        conv55_w.clone(),
        conv56_w.clone(),
        conv57_w.clone(),
        conv58_w.clone(),
        conv59_w.clone(),

        conv_residual1_weight.clone(),  //residual kernel 1
        conv_residual4_weight.clone(),  //residual kernel 1
        conv_residual8_weight.clone(),  //residual kernel 1
        conv_residual11_weight.clone(),  //residual kernel 1
        conv_residual14_weight.clone(),  //residual kernel 1

        fc1_w.clone(),

        x_0[0],

        conv21_output_0[0],
        conv22_output_0[0],
        conv23_output_0[0],
        conv24_output_0[0],
        conv25_output_0[0],
        conv26_output_0[0],
        conv27_output_0[0],
        conv28_output_0[0],
        conv29_output_0[0],
    
        conv31_output_0[0],
        conv32_output_0[0],
        conv33_output_0[0],
        conv34_output_0[0],
        conv35_output_0[0],
        conv36_output_0[0],
        conv37_output_0[0],
        conv38_output_0[0],
        conv39_output_0[0],
        conv3_10_output_0[0],
        conv3_11_output_0[0],
        conv3_12_output_0[0],


        conv41_output_0[0],
        conv42_output_0[0],
        conv43_output_0[0],
        conv44_output_0[0],
        conv45_output_0[0],
        conv46_output_0[0],
        conv47_output_0[0],
        conv48_output_0[0],
        conv49_output_0[0],
        conv4_10_output_0[0],
        conv4_11_output_0[0],
        conv4_12_output_0[0],
        conv4_13_output_0[0],
        conv4_14_output_0[0],
        conv4_15_output_0[0],
        conv4_16_output_0[0],
        conv4_17_output_0[0],
        conv4_18_output_0[0],
    
        conv51_output_0[0],
        conv52_output_0[0],
        conv53_output_0[0],
        conv54_output_0[0],
        conv55_output_0[0],
        conv56_output_0[0],
        conv57_output_0[0],
        conv58_output_0[0],
        conv59_output_0[0],
    
        conv_residual1_output_0[0], //residual output_0 1
        conv_residual4_output_0[0], //residual output_0 1
        conv_residual8_output_0[0], //residual output_0 1
        conv_residual11_output_0[0], //residual output_0 1
        conv_residual14_output_0[0], //residual output_0 1

        fc1_output_0[0],

        conv21_weights_0[0],
        conv22_weights_0[0],
        conv23_weights_0[0],
        conv24_weights_0[0],
        conv25_weights_0[0],
        conv26_weights_0[0],
        conv27_weights_0[0],
        conv28_weights_0[0],
        conv29_weights_0[0],
    
        conv31_weights_0[0],
        conv32_weights_0[0],
        conv33_weights_0[0],
        conv34_weights_0[0],
        conv35_weights_0[0],
        conv36_weights_0[0],
        conv37_weights_0[0],
        conv38_weights_0[0],
        conv39_weights_0[0],
        conv3_10_weights_0[0],
        conv3_11_weights_0[0],
        conv3_12_weights_0[0],


        conv41_weights_0[0],
        conv42_weights_0[0],
        conv43_weights_0[0],
        conv44_weights_0[0],
        conv45_weights_0[0],
        conv46_weights_0[0],
        conv47_weights_0[0],
        conv48_weights_0[0],
        conv49_weights_0[0],
        conv4_10_weights_0[0],
        conv4_11_weights_0[0],
        conv4_12_weights_0[0],
        conv4_13_weights_0[0],
        conv4_14_weights_0[0],
        conv4_15_weights_0[0],
        conv4_16_weights_0[0],
        conv4_17_weights_0[0],
        conv4_18_weights_0[0],
    
        conv51_weights_0[0],
        conv52_weights_0[0],
        conv53_weights_0[0],
        conv54_weights_0[0],
        conv55_weights_0[0],
        conv56_weights_0[0],
        conv57_weights_0[0],
        conv58_weights_0[0],
        conv59_weights_0[0],
    
        conv_residual1_weights_0[0], //residual weights_0 1
        conv_residual4_weights_0[0], //residual weights_0 1
        conv_residual8_weights_0[0], //residual weights_0 1
        conv_residual11_weights_0[0], //residual weights_0 1
        conv_residual14_weights_0[0], //residual weights_0 1

        fc1_weights_0[0],

        multiplier_conv21.clone(),
        multiplier_conv22.clone(),
        multiplier_conv23.clone(),
        multiplier_conv24.clone(),
        multiplier_conv25.clone(),
        multiplier_conv26.clone(),
        multiplier_conv27.clone(),
        multiplier_conv28.clone(),
        multiplier_conv29.clone(),

        multiplier_conv31.clone(),
        multiplier_conv31.clone(),
        multiplier_conv33.clone(),
        multiplier_conv34.clone(),
        multiplier_conv35.clone(),
        multiplier_conv36.clone(),
        multiplier_conv37.clone(),
        multiplier_conv38.clone(),
        multiplier_conv39.clone(),
        multiplier_conv3_10.clone(),
        multiplier_conv3_11.clone(),
        multiplier_conv3_12.clone(),

        multiplier_conv41.clone(),
        multiplier_conv41.clone(),
        multiplier_conv43.clone(),
        multiplier_conv44.clone(),
        multiplier_conv45.clone(),
        multiplier_conv46.clone(),
        multiplier_conv47.clone(),
        multiplier_conv48.clone(),
        multiplier_conv49.clone(),
        multiplier_conv4_10.clone(),
        multiplier_conv4_11.clone(),
        multiplier_conv4_12.clone(),
        multiplier_conv4_13.clone(),
        multiplier_conv4_14.clone(),
        multiplier_conv4_15.clone(),
        multiplier_conv4_16.clone(),
        multiplier_conv4_17.clone(),
        multiplier_conv4_18.clone(),

        multiplier_conv51.clone(),
        multiplier_conv52.clone(),
        multiplier_conv53.clone(),
        multiplier_conv54.clone(),
        multiplier_conv55.clone(),
        multiplier_conv56.clone(),
        multiplier_conv57.clone(),
        multiplier_conv58.clone(),
        multiplier_conv59.clone(),


        conv_residual1_multiplier.clone(),  //residual multiplier_0 1
        conv_residual4_multiplier.clone(),  //residual multiplier_0 1
        conv_residual8_multiplier.clone(),  //residual multiplier_0 1
        conv_residual11_multiplier.clone(),  //residual multiplier_0 1
        conv_residual14_multiplier.clone(),  //residual multiplier_0 1

        add_residual1_output_0[0],
        add_residual2_output_0[0],
        add_residual3_output_0[0],
        add_residual4_output_0[0],
        add_residual5_output_0[0],
        add_residual6_output_0[0],
        add_residual7_output_0[0],
        add_residual8_output_0[0],
        add_residual9_output_0[0],
        add_residual10_output_0[0],
        add_residual11_output_0[0],
        add_residual12_output_0[0],
        add_residual13_output_0[0],
        add_residual14_output_0[0],
        add_residual15_output_0[0],
        add_residual16_output_0[0],
    
        add_residual1_first_multiplier.clone(),
        add_residual2_first_multiplier.clone(),
        add_residual3_first_multiplier.clone(),
        add_residual4_first_multiplier.clone(),
        add_residual5_first_multiplier.clone(),
        add_residual6_first_multiplier.clone(),
        add_residual7_first_multiplier.clone(),
        add_residual8_first_multiplier.clone(),
        add_residual9_first_multiplier.clone(),
        add_residual10_first_multiplier.clone(),
        add_residual11_first_multiplier.clone(),
        add_residual12_first_multiplier.clone(),
        add_residual13_first_multiplier.clone(),
        add_residual14_first_multiplier.clone(),
        add_residual15_first_multiplier.clone(),
        add_residual16_first_multiplier.clone(),
    
        add_residual1_second_multiplier.clone(),
        add_residual2_second_multiplier.clone(),
        add_residual3_second_multiplier.clone(),
        add_residual4_second_multiplier.clone(),
        add_residual5_second_multiplier.clone(),
        add_residual6_second_multiplier.clone(),
        add_residual7_second_multiplier.clone(),
        add_residual8_second_multiplier.clone(),
        add_residual9_second_multiplier.clone(),
        add_residual10_second_multiplier.clone(),
        add_residual11_second_multiplier.clone(),
        add_residual12_second_multiplier.clone(),
        add_residual13_second_multiplier.clone(),
        add_residual14_second_multiplier.clone(),
        add_residual15_second_multiplier.clone(),
        add_residual16_second_multiplier.clone(),

        multiplier_fc1.clone(),
    );
    
    println!("finish forwarding");

    //batch size is only one for faster calculation of total constraints
    let flattened_x3d: Vec<Vec<Vec<u8>>> = x.clone().into_iter().flatten().collect();
    let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
    let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();

    let flattened_z1d: Vec<u8> = z.clone().into_iter().flatten().collect();

    //println!("x outside {:?}", x.clone());
    // println!("z outside {:?}", flattened_z1d.clone());
    let begin = Instant::now();
    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&flattened_x1d, &param, &x_open);

    println!("finish pedersen");

    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&flattened_z1d, &param, &z_open);
    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));
    //we only do one image in zk proof.

    let full_circuit = Resnet50CircuitU8OptimizedLv2PedersenPublicNNWeights {
        padding:1,
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,


        conv21_weights: conv21_w.clone(),
        conv22_weights: conv22_w.clone(),
        conv23_weights: conv23_w.clone(),
        conv24_weights: conv24_w.clone(),
        conv25_weights: conv25_w.clone(),
        conv26_weights: conv26_w.clone(),
        conv27_weights: conv27_w.clone(),
        conv28_weights: conv28_w.clone(),
        conv29_weights: conv29_w.clone(),

        conv31_weights: conv31_w.clone(),
        conv32_weights: conv32_w.clone(),
        conv33_weights: conv33_w.clone(),
        conv34_weights: conv34_w.clone(),
        conv35_weights: conv35_w.clone(),
        conv36_weights: conv36_w.clone(),
        conv37_weights: conv37_w.clone(),
        conv38_weights: conv38_w.clone(),
        conv39_weights: conv39_w.clone(),
        conv3_10_weights: conv3_10_w.clone(),
        conv3_11_weights: conv3_11_w.clone(),
        conv3_12_weights: conv3_12_w.clone(),

        conv41_weights: conv41_w.clone(),
        conv42_weights: conv42_w.clone(),
        conv43_weights: conv43_w.clone(),
        conv44_weights: conv44_w.clone(),
        conv45_weights: conv45_w.clone(),
        conv46_weights: conv46_w.clone(),
        conv47_weights: conv47_w.clone(),
        conv48_weights: conv48_w.clone(),
        conv49_weights: conv49_w.clone(),
        conv4_10_weights: conv4_10_w.clone(),
        conv4_11_weights: conv4_11_w.clone(),
        conv4_12_weights: conv4_12_w.clone(),
        conv4_13_weights: conv4_13_w.clone(),
        conv4_14_weights: conv4_14_w.clone(),
        conv4_15_weights: conv4_15_w.clone(),
        conv4_16_weights: conv4_16_w.clone(),
        conv4_17_weights: conv4_17_w.clone(),
        conv4_18_weights: conv4_18_w.clone(),

        conv51_weights: conv51_w.clone(),
        conv52_weights: conv52_w.clone(),
        conv53_weights: conv53_w.clone(),
        conv54_weights: conv54_w.clone(),
        conv55_weights: conv55_w.clone(),
        conv56_weights: conv56_w.clone(),
        conv57_weights: conv57_w.clone(),
        conv58_weights: conv58_w.clone(),
        conv59_weights: conv59_w.clone(),

        conv_residual1_weight: conv_residual1_weight.clone(),  //residual kernel 1
        conv_residual4_weight: conv_residual4_weight.clone(),  //residual kernel 1
        conv_residual8_weight: conv_residual8_weight.clone(),  //residual kernel 1
        conv_residual11_weight: conv_residual11_weight.clone(),  //residual kernel 1
        conv_residual14_weight: conv_residual14_weight.clone(),  //residual kernel 1

        fc1_weights: fc1_w.clone(),


        //zero points for quantization.
        x_0: x_0[0],

        conv21_output_0: conv21_output_0[0],
        conv22_output_0: conv22_output_0[0],
        conv23_output_0: conv23_output_0[0],
        conv24_output_0: conv24_output_0[0],
        conv25_output_0: conv25_output_0[0],
        conv26_output_0: conv26_output_0[0],
        conv27_output_0: conv27_output_0[0],
        conv28_output_0: conv28_output_0[0],
        conv29_output_0: conv29_output_0[0],

        conv31_output_0: conv31_output_0[0],
        conv32_output_0: conv32_output_0[0],
        conv33_output_0: conv33_output_0[0],
        conv34_output_0: conv34_output_0[0],
        conv35_output_0: conv35_output_0[0],
        conv36_output_0: conv36_output_0[0],
        conv37_output_0: conv37_output_0[0],
        conv38_output_0: conv38_output_0[0],
        conv39_output_0: conv39_output_0[0],
        conv3_10_output_0: conv3_10_output_0[0],
        conv3_11_output_0: conv3_11_output_0[0],
        conv3_12_output_0: conv3_12_output_0[0],

        conv41_output_0: conv41_output_0[0],
        conv42_output_0: conv42_output_0[0],
        conv43_output_0: conv43_output_0[0],
        conv44_output_0: conv44_output_0[0],
        conv45_output_0: conv45_output_0[0],
        conv46_output_0: conv46_output_0[0],
        conv47_output_0: conv47_output_0[0],
        conv48_output_0: conv48_output_0[0],
        conv49_output_0: conv49_output_0[0],
        conv4_10_output_0: conv4_10_output_0[0],
        conv4_11_output_0: conv4_11_output_0[0],
        conv4_12_output_0: conv4_12_output_0[0],
        conv4_13_output_0: conv4_13_output_0[0],
        conv4_14_output_0: conv4_14_output_0[0],
        conv4_15_output_0: conv4_15_output_0[0],
        conv4_16_output_0: conv4_16_output_0[0],
        conv4_17_output_0: conv4_17_output_0[0],
        conv4_18_output_0: conv4_18_output_0[0],

        conv51_output_0: conv51_output_0[0],
        conv52_output_0: conv52_output_0[0],
        conv53_output_0: conv53_output_0[0],
        conv54_output_0: conv54_output_0[0],
        conv55_output_0: conv55_output_0[0],
        conv56_output_0: conv56_output_0[0],
        conv57_output_0: conv57_output_0[0],
        conv58_output_0: conv58_output_0[0],
        conv59_output_0: conv59_output_0[0],

        conv_residual1_output_0: conv_residual1_output_0[0], //residual output_0 1
        conv_residual4_output_0: conv_residual4_output_0[0], //residual output_0 1
        conv_residual8_output_0: conv_residual8_output_0[0], //residual output_0 1
        conv_residual11_output_0: conv_residual11_output_0[0], //residual output_0 1
        conv_residual14_output_0: conv_residual14_output_0[0], //residual output_0 1

        fc1_output_0: fc1_output_0[0],

        conv21_weights_0: conv21_weights_0[0],
        conv22_weights_0: conv22_weights_0[0],
        conv23_weights_0: conv23_weights_0[0],
        conv24_weights_0: conv24_weights_0[0],
        conv25_weights_0: conv25_weights_0[0],
        conv26_weights_0: conv26_weights_0[0],
        conv27_weights_0: conv27_weights_0[0],
        conv28_weights_0: conv28_weights_0[0],
        conv29_weights_0: conv29_weights_0[0],

        conv31_weights_0: conv31_weights_0[0],
        conv32_weights_0: conv32_weights_0[0],
        conv33_weights_0: conv33_weights_0[0],
        conv34_weights_0: conv34_weights_0[0],
        conv35_weights_0: conv35_weights_0[0],
        conv36_weights_0: conv36_weights_0[0],
        conv37_weights_0: conv37_weights_0[0],
        conv38_weights_0: conv38_weights_0[0],
        conv39_weights_0: conv39_weights_0[0],
        conv3_10_weights_0: conv3_10_weights_0[0],
        conv3_11_weights_0: conv3_11_weights_0[0],
        conv3_12_weights_0: conv3_12_weights_0[0],

        conv41_weights_0: conv41_weights_0[0],
        conv42_weights_0: conv42_weights_0[0],
        conv43_weights_0: conv43_weights_0[0],
        conv44_weights_0: conv44_weights_0[0],
        conv45_weights_0: conv45_weights_0[0],
        conv46_weights_0: conv46_weights_0[0],
        conv47_weights_0: conv47_weights_0[0],
        conv48_weights_0: conv48_weights_0[0],
        conv49_weights_0: conv49_weights_0[0],
        conv4_10_weights_0: conv4_10_weights_0[0],
        conv4_11_weights_0: conv4_11_weights_0[0],
        conv4_12_weights_0: conv4_12_weights_0[0],
        conv4_13_weights_0: conv4_13_weights_0[0],
        conv4_14_weights_0: conv4_14_weights_0[0],
        conv4_15_weights_0: conv4_15_weights_0[0],
        conv4_16_weights_0: conv4_16_weights_0[0],
        conv4_17_weights_0: conv4_17_weights_0[0],
        conv4_18_weights_0: conv4_18_weights_0[0],

        conv51_weights_0: conv51_weights_0[0],
        conv52_weights_0: conv52_weights_0[0],
        conv53_weights_0: conv53_weights_0[0],
        conv54_weights_0: conv54_weights_0[0],
        conv55_weights_0: conv55_weights_0[0],
        conv56_weights_0: conv56_weights_0[0],
        conv57_weights_0: conv57_weights_0[0],
        conv58_weights_0: conv58_weights_0[0],
        conv59_weights_0: conv59_weights_0[0],

        conv_residual1_weights_0: conv_residual1_weights_0[0], //residual weights_0 1
        conv_residual4_weights_0: conv_residual4_weights_0[0], //residual weights_0 1
        conv_residual8_weights_0: conv_residual8_weights_0[0], //residual weights_0 1
        conv_residual11_weights_0: conv_residual11_weights_0[0], //residual weights_0 1
        conv_residual14_weights_0: conv_residual14_weights_0[0], //residual weights_0 1

        fc1_weights_0: fc1_weights_0[0],


        //multiplier for quantization

        conv21_multiplier: multiplier_conv21.clone(),
        conv22_multiplier: multiplier_conv22.clone(),
        conv23_multiplier: multiplier_conv23.clone(),
        conv24_multiplier: multiplier_conv24.clone(),
        conv25_multiplier: multiplier_conv25.clone(),
        conv26_multiplier: multiplier_conv26.clone(),
        conv27_multiplier: multiplier_conv27.clone(),
        conv28_multiplier: multiplier_conv28.clone(),
        conv29_multiplier: multiplier_conv29.clone(),

        conv31_multiplier: multiplier_conv31.clone(),
        conv32_multiplier: multiplier_conv32.clone(),
        conv33_multiplier: multiplier_conv33.clone(),
        conv34_multiplier: multiplier_conv34.clone(),
        conv35_multiplier: multiplier_conv35.clone(),
        conv36_multiplier: multiplier_conv36.clone(),
        conv37_multiplier: multiplier_conv37.clone(),
        conv38_multiplier: multiplier_conv38.clone(),
        conv39_multiplier: multiplier_conv39.clone(),
        conv3_10_multiplier: multiplier_conv3_10.clone(),
        conv3_11_multiplier: multiplier_conv3_11.clone(),
        conv3_12_multiplier: multiplier_conv3_12.clone(),

        conv41_multiplier: multiplier_conv41.clone(),
        conv42_multiplier: multiplier_conv42.clone(),
        conv43_multiplier: multiplier_conv43.clone(),
        conv44_multiplier: multiplier_conv44.clone(),
        conv45_multiplier: multiplier_conv45.clone(),
        conv46_multiplier: multiplier_conv46.clone(),
        conv47_multiplier: multiplier_conv47.clone(),
        conv48_multiplier: multiplier_conv48.clone(),
        conv49_multiplier: multiplier_conv49.clone(),
        conv4_10_multiplier: multiplier_conv4_10.clone(),
        conv4_11_multiplier: multiplier_conv4_11.clone(),
        conv4_12_multiplier: multiplier_conv4_12.clone(),
        conv4_13_multiplier: multiplier_conv4_13.clone(),
        conv4_14_multiplier: multiplier_conv4_14.clone(),
        conv4_15_multiplier: multiplier_conv4_15.clone(),
        conv4_16_multiplier: multiplier_conv4_16.clone(),
        conv4_17_multiplier: multiplier_conv4_17.clone(),
        conv4_18_multiplier: multiplier_conv4_18.clone(),

        conv51_multiplier: multiplier_conv51.clone(),
        conv52_multiplier: multiplier_conv52.clone(),
        conv53_multiplier: multiplier_conv53.clone(),
        conv54_multiplier: multiplier_conv54.clone(),
        conv55_multiplier: multiplier_conv55.clone(),
        conv56_multiplier: multiplier_conv56.clone(),
        conv57_multiplier: multiplier_conv57.clone(),
        conv58_multiplier: multiplier_conv58.clone(),
        conv59_multiplier: multiplier_conv59.clone(),

        conv_residual1_multiplier: conv_residual1_multiplier.clone(),  //residual multiplier_0 1
        conv_residual4_multiplier: conv_residual4_multiplier.clone(),  //residual multiplier_0 1
        conv_residual8_multiplier: conv_residual8_multiplier.clone(),  //residual multiplier_0 1
        conv_residual11_multiplier: conv_residual11_multiplier.clone(),  //residual multiplier_0 1
        conv_residual14_multiplier: conv_residual14_multiplier.clone(),  //residual multiplier_0 1

        add_residual1_output_0: add_residual1_output_0[0],
        add_residual2_output_0: add_residual2_output_0[0],
        add_residual3_output_0: add_residual3_output_0[0],
        add_residual4_output_0: add_residual4_output_0[0],
        add_residual5_output_0: add_residual5_output_0[0],
        add_residual6_output_0: add_residual6_output_0[0],
        add_residual7_output_0: add_residual7_output_0[0],
        add_residual8_output_0: add_residual8_output_0[0],
        add_residual9_output_0: add_residual9_output_0[0],
        add_residual10_output_0: add_residual10_output_0[0],
        add_residual11_output_0: add_residual11_output_0[0],
        add_residual12_output_0: add_residual12_output_0[0],
        add_residual13_output_0: add_residual13_output_0[0],
        add_residual14_output_0: add_residual14_output_0[0],
        add_residual15_output_0: add_residual15_output_0[0],
        add_residual16_output_0: add_residual16_output_0[0],

        add_residual1_first_multiplier: add_residual1_first_multiplier.clone(),
        add_residual2_first_multiplier: add_residual2_first_multiplier.clone(),
        add_residual3_first_multiplier: add_residual3_first_multiplier.clone(),
        add_residual4_first_multiplier: add_residual4_first_multiplier.clone(),
        add_residual5_first_multiplier: add_residual5_first_multiplier.clone(),
        add_residual6_first_multiplier: add_residual6_first_multiplier.clone(),
        add_residual7_first_multiplier: add_residual7_first_multiplier.clone(),
        add_residual8_first_multiplier: add_residual8_first_multiplier.clone(),
        add_residual9_first_multiplier: add_residual9_first_multiplier.clone(),
        add_residual10_first_multiplier: add_residual10_first_multiplier.clone(),
        add_residual11_first_multiplier: add_residual11_first_multiplier.clone(),
        add_residual12_first_multiplier: add_residual12_first_multiplier.clone(),
        add_residual13_first_multiplier: add_residual13_first_multiplier.clone(),
        add_residual14_first_multiplier: add_residual14_first_multiplier.clone(),
        add_residual15_first_multiplier: add_residual15_first_multiplier.clone(),
        add_residual16_first_multiplier: add_residual16_first_multiplier.clone(),

        add_residual1_second_multiplier: add_residual1_second_multiplier.clone(),
        add_residual2_second_multiplier: add_residual2_second_multiplier.clone(),
        add_residual3_second_multiplier: add_residual3_second_multiplier.clone(),
        add_residual4_second_multiplier: add_residual4_second_multiplier.clone(),
        add_residual5_second_multiplier: add_residual5_second_multiplier.clone(),
        add_residual6_second_multiplier: add_residual6_second_multiplier.clone(),
        add_residual7_second_multiplier: add_residual7_second_multiplier.clone(),
        add_residual8_second_multiplier: add_residual8_second_multiplier.clone(),
        add_residual9_second_multiplier: add_residual9_second_multiplier.clone(),
        add_residual10_second_multiplier: add_residual10_second_multiplier.clone(),
        add_residual11_second_multiplier: add_residual11_second_multiplier.clone(),
        add_residual12_second_multiplier: add_residual12_second_multiplier.clone(),
        add_residual13_second_multiplier: add_residual13_second_multiplier.clone(),
        add_residual14_second_multiplier: add_residual14_second_multiplier.clone(),
        add_residual15_second_multiplier: add_residual15_second_multiplier.clone(),
        add_residual16_second_multiplier: add_residual16_second_multiplier.clone(),

        multiplier_fc1: multiplier_fc1.clone(),


        z: z.clone(),
        z_open: z_open,
        z_com: z_com,
        knit_encoding: true,
    };

    // // sanity checks
    // {
    //     let sanity_cs = ConstraintSystem::<Fq>::new_ref();
    //     full_circuit
    //         .clone()
    //         .generate_constraints(sanity_cs.clone())
    //         .unwrap();

    //     let res = sanity_cs.is_satisfied().unwrap();
    //     println!("are the constraints satisfied?: {}\n", res);

    //     if !res {
    //         println!(
    //             "{:?} {} {:#?}",
    //             sanity_cs.constraint_names(),
    //             sanity_cs.num_constraints(),
    //             sanity_cs.which_is_unsatisfied().unwrap()
    //         );
    //     }
    // }
    println!("start generating random parameters");
    let begin = Instant::now();

    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let end = Instant::now();

    println!("setup time {:?}", end.duration_since(begin));

    let mut buf = vec![];
    param.serialize(&mut buf).unwrap();
    println!("crs size: {}", buf.len());

    let pvk = prepare_verifying_key(&param.vk);
    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));

    let commitment = [x_com.x, x_com.y, z_com.x, z_com.y].to_vec();

    let inputs: Vec<Fq> = [
        commitment[..].as_ref(),
    ]
    .concat();

    let begin = Instant::now();
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());
    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}
