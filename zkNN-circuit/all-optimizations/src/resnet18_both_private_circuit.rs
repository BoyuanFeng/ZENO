use crate::argmax_circuit::*;
use crate::avg_pool_circuit::*;
use crate::conv_circuit::*;
use crate::cosine_circuit::*;
use crate::mul_circuit::*;
use crate::relu_circuit::*;
use crate::both_private_circuit::*;
use crate::residual_add_circuit::*;
use crate::vanilla::*;
use crate::*;

use algebra::ed_on_bls12_381::*;
use algebra_core::Zero;
use pedersen_commit::*;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::fields::fp::FpVar;
use std::cmp::min;

#[derive(Clone)]
pub struct Resnet18CircuitBothPrivate {
    pub padding:usize,
    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,

    pub conv21_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv21_open: PedersenRandomness,
    pub conv21_com_vec: Vec<PedersenCommitment>,

    pub conv22_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv22_open: PedersenRandomness,
    pub conv22_com_vec: Vec<PedersenCommitment>,

    pub conv23_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv23_open: PedersenRandomness,
    pub conv23_com_vec: Vec<PedersenCommitment>,

    pub conv24_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv24_open: PedersenRandomness,
    pub conv24_com_vec: Vec<PedersenCommitment>,
    //------------------------------ --
    pub conv31_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv31_open: PedersenRandomness,
    pub conv31_com_vec: Vec<PedersenCommitment>,

    pub conv32_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv32_open: PedersenRandomness,
    pub conv32_com_vec: Vec<PedersenCommitment>,

    pub conv33_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv33_open: PedersenRandomness,
    pub conv33_com_vec: Vec<PedersenCommitment>,

    pub conv34_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv34_open: PedersenRandomness,
    pub conv34_com_vec: Vec<PedersenCommitment>,

    //------------------------------ --
    pub conv41_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv41_open: PedersenRandomness,
    pub conv41_com_vec: Vec<PedersenCommitment>,

    pub conv42_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv42_open: PedersenRandomness,
    pub conv42_com_vec: Vec<PedersenCommitment>,

    pub conv43_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv43_open: PedersenRandomness,
    pub conv43_com_vec: Vec<PedersenCommitment>,

    pub conv44_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv44_open: PedersenRandomness,
    pub conv44_com_vec: Vec<PedersenCommitment>,

    //------------------------------ --
    pub conv51_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv51_open: PedersenRandomness,
    pub conv51_com_vec: Vec<PedersenCommitment>,

    pub conv52_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv52_open: PedersenRandomness,
    pub conv52_com_vec: Vec<PedersenCommitment>,

    pub conv53_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv53_open: PedersenRandomness,
    pub conv53_com_vec: Vec<PedersenCommitment>,

    pub conv54_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv54_open: PedersenRandomness,
    pub conv54_com_vec: Vec<PedersenCommitment>,

    pub conv_residuel1_kernel: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    pub conv_residuel1_open: PedersenRandomness,
    pub conv_residuel1_com_vec: Vec<PedersenCommitment>,

    pub conv_residuel2_kernel: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    pub conv_residuel2_open: PedersenRandomness,
    pub conv_residuel2_com_vec: Vec<PedersenCommitment>,

    pub conv_residuel3_kernel: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    pub conv_residuel3_open: PedersenRandomness,
    pub conv_residuel3_com_vec: Vec<PedersenCommitment>,

    pub conv_residuel4_kernel: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    pub conv_residuel4_open: PedersenRandomness,
    pub conv_residuel4_com_vec: Vec<PedersenCommitment>,

    pub fc1_weights: Vec<Vec<u8>>,
    pub fc1_weights_open: PedersenRandomness,
    pub fc1_weights_com_vec: Vec<PedersenCommitment>,


    //zero points for quantization.
    pub x_0: u8,

    pub conv21_output_0: u8,
    pub conv22_output_0: u8,
    pub conv23_output_0: u8,
    pub conv24_output_0: u8,

    pub conv31_output_0: u8,
    pub conv32_output_0: u8,
    pub conv33_output_0: u8,
    pub conv34_output_0: u8,

    pub conv41_output_0: u8,
    pub conv42_output_0: u8,
    pub conv43_output_0: u8,
    pub conv44_output_0: u8,

    pub conv51_output_0: u8,
    pub conv52_output_0: u8,
    pub conv53_output_0: u8,
    pub conv54_output_0: u8,

    pub conv_residuel1_output_0:u8,  // residual output_0 1
    pub conv_residuel2_output_0:u8,  // residual output_0 1
    pub conv_residuel3_output_0:u8,  // residual output_0 1
    pub conv_residuel4_output_0:u8,  // residual output_0 1

    pub fc1_output_0: u8,


    pub conv21_weights_0: u8,
    pub conv22_weights_0: u8,
    pub conv23_weights_0: u8,
    pub conv24_weights_0: u8,

    pub conv31_weights_0: u8,
    pub conv32_weights_0: u8,
    pub conv33_weights_0: u8,
    pub conv34_weights_0: u8,

    pub conv41_weights_0: u8,
    pub conv42_weights_0: u8,
    pub conv43_weights_0: u8,
    pub conv44_weights_0: u8,

    pub conv51_weights_0: u8,
    pub conv52_weights_0: u8,
    pub conv53_weights_0: u8,
    pub conv54_weights_0: u8,

    pub conv_residuel1_weights_0:u8,  // residual weights_0 1
    pub conv_residuel2_weights_0:u8,  // residual weights_0 1
    pub conv_residuel3_weights_0:u8,  // residual weights_0 1
    pub conv_residuel4_weights_0:u8,  // residual weights_0 1

    pub fc1_weights_0: u8,


    //multiplier for quantization


    pub multiplier_conv21: Vec<f32>,
    pub multiplier_conv22: Vec<f32>,
    pub multiplier_conv23: Vec<f32>,
    pub multiplier_conv24: Vec<f32>,

    pub multiplier_conv31: Vec<f32>,
    pub multiplier_conv32: Vec<f32>,
    pub multiplier_conv33: Vec<f32>,
    pub multiplier_conv34: Vec<f32>,

    pub multiplier_conv41: Vec<f32>,
    pub multiplier_conv42: Vec<f32>,
    pub multiplier_conv43: Vec<f32>,
    pub multiplier_conv44: Vec<f32>,

    pub multiplier_conv51: Vec<f32>,
    pub multiplier_conv52: Vec<f32>,
    pub multiplier_conv53: Vec<f32>,
    pub multiplier_conv54: Vec<f32>,

    pub multiplier_residuel1:Vec<f32>,  // residual multiplier 1
    pub multiplier_residuel2:Vec<f32>,  // residual multiplier 1
    pub multiplier_residuel3:Vec<f32>,  // residual multiplier 1
    pub multiplier_residuel4:Vec<f32>,  // residual multiplier 1

    pub residual1_input_0:u8,
    pub residual2_input_0:u8,
    pub residual3_input_0:u8,
    pub residual4_input_0:u8,
    pub residual5_input_0:u8,
    pub residual6_input_0:u8,
    pub residual7_input_0:u8,
    pub residual8_input_0:u8,

    pub residual1_output_0:u8,
    pub residual2_output_0:u8,
    pub residual3_output_0:u8,
    pub residual4_output_0:u8,
    pub residual5_output_0:u8,
    pub residual6_output_0:u8,
    pub residual7_output_0:u8,
    pub residual8_output_0:u8,

    pub residual1_multiplier:Vec<f32>,
    pub residual2_multiplier:Vec<f32>,
    pub residual3_multiplier:Vec<f32>,
    pub residual4_multiplier:Vec<f32>,
    pub residual5_multiplier:Vec<f32>,
    pub residual6_multiplier:Vec<f32>,
    pub residual7_multiplier:Vec<f32>,
    pub residual8_multiplier:Vec<f32>,

    pub multiplier_fc1: Vec<f32>,


    //we do not need multiplier in relu and AvgPool layer
    pub z: Vec<Vec<u8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
    pub knit_encoding: bool,
}

impl ConstraintSynthesizer<Fq> for Resnet18CircuitBothPrivate {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("Resnet18 is setup mode: {}", cs.is_in_setup_mode());
        //println!("{:?}", self.x);
        //because we assume that weights are public constants, prover does not need to commit the weights.

        // x commitment
        let flattened_x3d: Vec<Vec<Vec<u8>>> = self.x.clone().into_iter().flatten().collect();
        let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
        let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();
        let x_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_x1d.clone(),
            open: self.x_open,
            commit: self.x_com,
        };
        x_com_circuit.generate_constraints(cs.clone())?;
        let mut _cir_number = cs.num_constraints();
        // #[cfg(debug_assertion)]
        println!("Number of constraints for x commitment {}", _cir_number);

        // Weight commitment

        let len_per_commit = PERDERSON_WINDOW_NUM * PERDERSON_WINDOW_SIZE / 8; //for vec<u8> commitment

        // TODO: Repeat for all weights: conv21 => fc1
        let conv21_mat_1d = convert_4d_vector_into_1d(self.conv21_weights.clone());
        let num_of_commit_conv21 = conv21_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv21 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv21_mat_1d.len()) {
                tmp.push(conv21_mat_1d[j]);
            }
            let conv21_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv21_open.clone(),
                commit: self.conv21_com_vec[i],
            };
            conv21_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let conv22_mat_1d = convert_4d_vector_into_1d(self.conv22_weights.clone());
        let num_of_commit_conv22 = conv22_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv22 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv22_mat_1d.len()) {
                tmp.push(conv22_mat_1d[j]);
            }
            let conv22_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv22_open.clone(),
                commit: self.conv22_com_vec[i],
            };
            conv22_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let conv23_mat_1d = convert_4d_vector_into_1d(self.conv23_weights.clone());
        let num_of_commit_conv23 = conv23_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv23 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv23_mat_1d.len()) {
                tmp.push(conv23_mat_1d[j]);
            }
            let conv23_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv23_open.clone(),
                commit: self.conv23_com_vec[i],
            };
            conv23_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let conv24_mat_1d = convert_4d_vector_into_1d(self.conv24_weights.clone());
        let num_of_commit_conv24 = conv24_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv24 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv24_mat_1d.len()) {
                tmp.push(conv24_mat_1d[j]);
            }
            let conv24_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv24_open.clone(),
                commit: self.conv24_com_vec[i],
            };
            conv24_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        let conv31_mat_1d = convert_4d_vector_into_1d(self.conv31_weights.clone());
        let num_of_commit_conv31 = conv31_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv31 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv31_mat_1d.len()) {
                tmp.push(conv31_mat_1d[j]);
            }
            let conv31_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv31_open.clone(),
                commit: self.conv31_com_vec[i],
            };
            conv31_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv32 commit =========================================================================
        let conv32_mat_1d = convert_4d_vector_into_1d(self.conv32_weights.clone());
        let num_of_commit_conv32 = conv32_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv32 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv32_mat_1d.len()) {
                tmp.push(conv32_mat_1d[j]);
            }
            let conv32_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv32_open.clone(),
                commit: self.conv32_com_vec[i],
            };
            conv32_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv33 commit =========================================================================
        let conv33_mat_1d = convert_4d_vector_into_1d(self.conv33_weights.clone());
        let num_of_commit_conv33 = conv33_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv33 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv33_mat_1d.len()) {
                tmp.push(conv33_mat_1d[j]);
            }
            let conv33_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv33_open.clone(),
                commit: self.conv33_com_vec[i],
            };
            conv33_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv34 commit =========================================================================
        let conv34_mat_1d = convert_4d_vector_into_1d(self.conv34_weights.clone());
        let num_of_commit_conv34 = conv34_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv34 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv34_mat_1d.len()) {
                tmp.push(conv34_mat_1d[j]);
            }
            let conv34_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv34_open.clone(),
                commit: self.conv34_com_vec[i],
            };
            conv34_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv41 commit =========================================================================
        let conv41_mat_1d = convert_4d_vector_into_1d(self.conv41_weights.clone());
        let num_of_commit_conv41 = conv41_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv41 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv41_mat_1d.len()) {
                tmp.push(conv41_mat_1d[j]);
            }
            let conv41_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv41_open.clone(),
                commit: self.conv41_com_vec[i],
            };
            conv41_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv42 commit =========================================================================
        let conv42_mat_1d = convert_4d_vector_into_1d(self.conv42_weights.clone());
        let num_of_commit_conv42 = conv42_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv42 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv42_mat_1d.len()) {
                tmp.push(conv42_mat_1d[j]);
            }
            let conv42_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv42_open.clone(),
                commit: self.conv42_com_vec[i],
            };
            conv42_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv43 commit =========================================================================
        let conv43_mat_1d = convert_4d_vector_into_1d(self.conv43_weights.clone());
        let num_of_commit_conv43 = conv43_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv43 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv43_mat_1d.len()) {
                tmp.push(conv43_mat_1d[j]);
            }
            let conv43_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv43_open.clone(),
                commit: self.conv43_com_vec[i],
            };
            conv43_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv44 commit =========================================================================
        let conv44_mat_1d = convert_4d_vector_into_1d(self.conv44_weights.clone());
        let num_of_commit_conv44 = conv44_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv44 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv44_mat_1d.len()) {
                tmp.push(conv44_mat_1d[j]);
            }
            let conv44_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv44_open.clone(),
                commit: self.conv44_com_vec[i],
            };
            conv44_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv51 commit =========================================================================
        let conv51_mat_1d = convert_4d_vector_into_1d(self.conv51_weights.clone());
        let num_of_commit_conv51 = conv51_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv51 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv51_mat_1d.len()) {
                tmp.push(conv51_mat_1d[j]);
            }
            let conv51_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv51_open.clone(),
                commit: self.conv51_com_vec[i],
            };
            conv51_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv52 commit =========================================================================
        let conv52_mat_1d = convert_4d_vector_into_1d(self.conv52_weights.clone());
        let num_of_commit_conv52 = conv52_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv52 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv52_mat_1d.len()) {
                tmp.push(conv52_mat_1d[j]);
            }
            let conv52_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv52_open.clone(),
                commit: self.conv52_com_vec[i],
            };
            conv52_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv53 commit =========================================================================
        let conv53_mat_1d = convert_4d_vector_into_1d(self.conv53_weights.clone());
        let num_of_commit_conv53 = conv53_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv53 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv53_mat_1d.len()) {
                tmp.push(conv53_mat_1d[j]);
            }
            let conv53_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv53_open.clone(),
                commit: self.conv53_com_vec[i],
            };
            conv53_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv54 commit =========================================================================
        let conv54_mat_1d = convert_4d_vector_into_1d(self.conv54_weights.clone());
        let num_of_commit_conv54 = conv54_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv54 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv54_mat_1d.len()) {
                tmp.push(conv54_mat_1d[j]);
            }
            let conv54_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv54_open.clone(),
                commit: self.conv54_com_vec[i],
            };
            conv54_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv residual 1 commit =========================================================================
        let conv_residuel1_mat_1d = convert_4d_vector_into_1d(self.conv_residuel1_kernel.clone());
        let num_of_commit_conv_residuel1 = conv_residuel1_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv_residuel1 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv_residuel1_mat_1d.len()) {
                tmp.push(conv_residuel1_mat_1d[j]);
            }
            let conv_residuel1_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv_residuel1_open.clone(),
                commit: self.conv_residuel1_com_vec[i],
            };
            conv_residuel1_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv_residuel2 commit =========================================================================
        let conv_residuel2_mat_1d = convert_4d_vector_into_1d(self.conv_residuel2_kernel.clone());
        let num_of_commit_conv_residuel2 = conv_residuel2_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv_residuel2 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv_residuel2_mat_1d.len()) {
                tmp.push(conv_residuel2_mat_1d[j]);
            }
            let conv_residuel2_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv_residuel2_open.clone(),
                commit: self.conv_residuel2_com_vec[i],
            };
            conv_residuel2_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv_residuel3 commit =========================================================================
        let conv_residuel3_mat_1d = convert_4d_vector_into_1d(self.conv_residuel3_kernel.clone());
        let num_of_commit_conv_residuel3 = conv_residuel3_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv_residuel3 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv_residuel3_mat_1d.len()) {
                tmp.push(conv_residuel3_mat_1d[j]);
            }
            let conv_residuel3_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv_residuel3_open.clone(),
                commit: self.conv_residuel3_com_vec[i],
            };
            conv_residuel3_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // conv_residuel4 commit =========================================================================
        let conv_residuel4_mat_1d = convert_4d_vector_into_1d(self.conv_residuel4_kernel.clone());
        let num_of_commit_conv_residuel4 = conv_residuel4_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv_residuel4 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv_residuel4_mat_1d.len()) {
                tmp.push(conv_residuel4_mat_1d[j]);
            }
            let conv_residuel4_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv_residuel4_open.clone(),
                commit: self.conv_residuel4_com_vec[i],
            };
            conv_residuel4_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

        // fc1 commit =========================================================================
        let fc1_mat_1d = convert_2d_vector_into_1d(self.fc1_weights.clone());
        let num_of_commit_fc1 = fc1_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_fc1 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, fc1_mat_1d.len()) {
                tmp.push(fc1_mat_1d[j]);
            }
            let fc1_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.fc1_weights_open.clone(),
                commit: self.fc1_weights_com_vec[i],
            };
            fc1_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );


        // residual 1*1 conv 1 =================================================================================
        let mut residual1 = vec![vec![vec![vec![0u8; self.x[0][0][0].len()];  // w - kernel_size + 1
                self.x[0][0].len()]; // h - kernel_size + 1
                self.conv22_weights.len()]; //number of conv kernels
                self.x.len()]; //input (image) batch size

        let (remainder_residual1, div_residual1) = vec_conv_with_remainder_u8(
            &self.x,
            &self.conv_residuel1_kernel,
            &mut residual1,
            self.x_0,
            self.conv_residuel1_weights_0,
            self.conv_residuel1_output_0,
            &self.multiplier_residuel1,
        );
        
        let residual_conv1_circuit = ConvCircuitOp3Wrapper{
            x: self.x.clone(),
            conv_kernel: self.conv_residuel1_kernel.clone(),
            y: residual1.clone(),

            remainder: remainder_residual1.clone(),
            div: div_residual1.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv_residuel1_weights_0,
            y_0: self.conv_residuel1_output_0,

            multiplier: self.multiplier_residuel1,
            padding: 0,
        };

        residual_conv1_circuit.generate_constraints(cs.clone())?;

        //----------------------------------------------------------------------
        let padded_x = padding_helper(self.x.clone(), self.padding);

        let mut conv21_output = vec![vec![vec![vec![0u8; padded_x[0][0][0].len() - self.conv21_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_x[0][0].len() - self.conv21_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv21_weights.len()]; //number of conv kernels
        padded_x.len()]; //input (image) batch size

        let (remainder_conv21, div_conv21) = vec_conv_with_remainder_u8(
            &padded_x,
            &self.conv21_weights,
            &mut conv21_output,
            self.x_0,
            self.conv21_weights_0,
            self.conv21_output_0,
            &self.multiplier_conv21,
        );

        let conv21_circuit = ConvCircuitOp3Wrapper{
            x: self.x.clone(),
            conv_kernel: self.conv21_weights.clone(),
            y: conv21_output.clone(),

            remainder: remainder_conv21.clone(),
            div: div_conv21.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv21_weights_0,
            y_0: self.conv21_output_0,

            multiplier: self.multiplier_conv21,
            padding: self.padding,
        };
        conv21_circuit.generate_constraints(cs.clone())?;

        let cmp_res1 = relu4d_u8(&mut conv21_output.clone(), self.x_0);
        let relu1_circuit = ReLUWrapper{
            x: conv21_output.clone(),
            cmp_res: cmp_res1,
            y_0: self.conv21_output_0,
        };
        relu1_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv21 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //----------------------------------------------------------------------
        let padded_conv21_output = padding_helper(conv21_output.clone(), self.padding);

        let mut conv22_output = vec![vec![vec![vec![0u8; padded_conv21_output[0][0][0].len() - self.conv22_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv21_output[0][0].len() - self.conv22_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv22_weights.len()]; //number of conv kernels
        padded_conv21_output.len()]; //input (image) batch size

        let (remainder_conv22, div_conv22) = vec_conv_with_remainder_u8(
            &padded_conv21_output,
            &self.conv22_weights,
            &mut conv22_output,
            self.conv21_output_0,
            self.conv22_weights_0,
            self.conv22_output_0,
            &self.multiplier_conv22,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv22_circuit = ConvCircuitOp3Wrapper {
            x: conv21_output.clone(),
            conv_kernel: self.conv22_weights.clone(),
            y: conv22_output.clone(),

            remainder: remainder_conv22.clone(),
            div: div_conv22.clone(),

            x_0: self.conv21_output_0,
            conv_kernel_0: self.conv22_weights_0,
            y_0: self.conv22_output_0,

            multiplier: self.multiplier_conv22.clone(),
            padding: self.padding,
        };
        conv22_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv22 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // residual add 1 =================================================================================================================
        let mut residual_0_output = vec![vec![vec![vec![0u8; residual1[0][0][0].len()];  // w - kernel_size + 1
                                                residual1[0][0].len()]; // h - kernel_size + 1
                                                residual1[0].len()]; //number of conv kernels
                                                residual1.len()]; //input (image) batch size
    
        // residual_add(&residual1,&conv22_output,&mut residual_0_output);  // Residual layer 1
        let (residual1_remainder, residual1_div) = residual_add_plaintext(&residual1, &conv22_output,&mut residual_0_output, self.residual1_input_0, 
                                                self.conv22_output_0, self.residual1_output_0, &self.residual1_multiplier, &self.multiplier_conv22);

        let residual1_circuit = ResidualAddCircuit{
            input1: residual1.clone(),
            input2: conv22_output.clone(),
            output: residual_0_output.clone(),

            remainder: residual1_remainder.clone(),
            div: residual1_div.clone(),

            input1_0: self.residual1_input_0,
            input2_0: self.conv22_output_0,
            output_0: self.residual1_output_0,

            multiplier_1: self.residual1_multiplier,
            multiplier_2: self.multiplier_conv22,
            knit_encoding: self.knit_encoding,
        };
        residual1_circuit.generate_constraints(cs.clone())?;

        let cmp_res2 = relu4d_u8(&mut conv22_output.clone(), self.x_0);
        let relu2_circuit = ReLUWrapper{
            x: conv22_output.clone(),
            cmp_res: cmp_res2,
            y_0: self.conv22_output_0,
        };
        relu2_circuit.generate_constraints(cs.clone())?;

        println!(
            "Number of constraints for residual1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let padded_conv22_output = padding_helper(residual_0_output.clone(), self.padding);
        
        // =================================================================================================================================

        // let padded_conv22_output = padding_helper(conv22_output, self.padding);

        let mut conv23_output = vec![vec![vec![vec![0u8; padded_conv22_output[0][0][0].len() - self.conv23_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv22_output[0][0].len() - self.conv23_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv23_weights.len()]; //number of conv kernels
        padded_conv22_output.len()]; //input (image) batch size

        let (remainder_conv23, div_conv23) = vec_conv_with_remainder_u8(
            &padded_conv22_output,
            &self.conv23_weights,
            &mut conv23_output,
            self.conv22_output_0,
            self.conv23_weights_0,
            self.conv23_output_0,
            &self.multiplier_conv23,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv23_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv22_output.clone(),
            conv_kernel: self.conv23_weights.clone(),
            y: conv23_output.clone(),

            remainder: remainder_conv23.clone(),
            div: div_conv23.clone(),

            x_0: self.conv22_output_0,
            conv_kernel_0: self.conv23_weights_0,
            y_0: self.conv23_output_0,

            multiplier: self.multiplier_conv23,
            padding: self.padding,
        };
        conv23_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv23 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let cmp_res3 = relu4d_u8(&mut conv23_output.clone(), self.x_0);
        let relu3_circuit = ReLUWrapper{
            x: conv23_output.clone(),
            cmp_res: cmp_res3,
            y_0: self.conv23_output_0,
        };
        relu3_circuit.generate_constraints(cs.clone())?;


        let padded_conv23_output = padding_helper(conv23_output.clone(), self.padding);

        let mut conv24_output = vec![vec![vec![vec![0u8; padded_conv23_output[0][0][0].len() - self.conv24_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv23_output[0][0].len() - self.conv23_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv24_weights.len()]; //number of conv kernels
        padded_conv23_output.len()]; //input (image) batch size

        let (remainder_conv24, div_conv24) = vec_conv_with_remainder_u8(
            &padded_conv23_output,
            &self.conv24_weights,
            &mut conv24_output,
            self.conv23_output_0,
            self.conv24_weights_0,
            self.conv24_output_0,
            &self.multiplier_conv24,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv24_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv23_output.clone(),
            conv_kernel: self.conv24_weights.clone(),
            y: conv24_output.clone(),

            remainder: remainder_conv24.clone(),
            div: div_conv24.clone(),

            x_0: self.conv23_output_0,
            conv_kernel_0: self.conv24_weights_0,
            y_0: self.conv24_output_0,

            multiplier: self.multiplier_conv24.clone(),
            padding: self.padding,
        };
        conv24_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv24 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        // residual add 2 =================================================================================================================
        let mut residual_1_output = vec![vec![vec![vec![0u8; conv24_output[0][0][0].len()];  // w - kernel_size + 1
                                        conv24_output[0][0].len()]; // h - kernel_size + 1
                                        conv24_output[0].len()]; //number of conv kernels
                                        conv24_output.len()]; //input (image) batch size
    
        // residual_add(&residual_0_output,&conv24_output,&mut residual_1_output);  // Residual layer 1

        let (residual2_remainder, residual2_div) = residual_add_plaintext(&residual_0_output, &conv24_output,&mut residual_1_output, self.residual2_input_0, 
                    self.conv24_output_0, self.residual2_output_0, &self.residual2_multiplier, &self.multiplier_conv24);

        let residual2_circuit = ResidualAddCircuit{
        input1: residual_0_output.clone(),
        input2: conv24_output.clone(),
        output: residual_1_output.clone(),

        remainder: residual2_remainder.clone(),
        div: residual2_div.clone(),

        input1_0: self.residual2_input_0,
        input2_0: self.conv24_output_0,
        output_0: self.residual2_output_0,

        multiplier_1: self.residual2_multiplier,
        multiplier_2: self.multiplier_conv24,
        knit_encoding: self.knit_encoding,
        };
        residual2_circuit.generate_constraints(cs.clone())?;

        println!(
        "Number of constraints for residual2 {} accumulated constraints {}",
        cs.num_constraints() - _cir_number,
        cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let cmp_res3 = relu4d_u8(&mut conv23_output.clone(), self.x_0);
        let relu3_circuit = ReLUWrapper{
            x: conv23_output.clone(),
            cmp_res: cmp_res3,
            y_0: self.conv23_output_0,
        };
        relu3_circuit.generate_constraints(cs.clone())?;

        let cmp_res4 = relu4d_u8(&mut residual_1_output.clone(), self.x_0);
        let relu4_circuit = ReLUWrapper{
            x: residual_1_output.clone(),
            cmp_res: cmp_res4,
            y_0: self.residual2_output_0,
        };
        relu4_circuit.generate_constraints(cs.clone())?;

        let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&residual_1_output.clone(), 2);
        
        // =================================================================================================================================
         //avg pool2 layer
        // let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&conv24_output, 2);

        let avg_pool2_circuit = AvgPoolWrapper {
            x: residual_1_output.clone(),
            y: avg_pool2_output.clone(),
            kernel_size: 2,
            remainder: avg2_remainder.clone(),
        };
        avg_pool2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        // residual 1*1 conv 2 =================================================================================
        let mut residual2 = vec![vec![vec![vec![0u8; avg_pool2_output[0][0][0].len()];  // w - kernel_size + 1
                                avg_pool2_output[0][0].len()]; // h - kernel_size + 1
                                self.conv32_weights.len()]; //number of conv kernels
                                avg_pool2_output.len()]; //input (image) batch size

        let (remainder_residual2, div_residual2) = vec_conv_with_remainder_u8(
            &avg_pool2_output,
            &self.conv_residuel2_kernel,
            &mut residual2,
            self.conv24_output_0,
            self.conv_residuel2_weights_0,
            self.conv_residuel2_output_0,
            &self.multiplier_residuel2,
        );
        
        let residual_conv2_circuit = ConvCircuitOp3Wrapper {
            x: avg_pool2_output.clone(),
            conv_kernel: self.conv_residuel2_kernel.clone(),
            y: residual2.clone(),

            remainder: remainder_residual2.clone(),
            div: div_residual2.clone(),

            x_0: self.conv24_output_0,
            conv_kernel_0: self.conv_residuel2_weights_0,
            y_0: self.conv_residuel2_output_0,

            multiplier: self.multiplier_residuel2,
            padding: self.padding,
        };
        residual_conv2_circuit.generate_constraints(cs.clone())?;
        

        //----------------------------------------------------------------------
        let padded_conv24_output = padding_helper(avg_pool2_output.clone(), self.padding);

        let mut conv31_output = vec![vec![vec![vec![0u8; padded_conv24_output[0][0][0].len() - self.conv31_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv24_output[0][0].len() - self.conv31_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv31_weights.len()]; //number of conv kernels
        padded_conv24_output.len()]; //input (image) batch size

        let (remainder_conv31, div_conv31) = vec_conv_with_remainder_u8(
            &padded_conv24_output,
            &self.conv31_weights,
            &mut conv31_output,
            self.conv22_output_0,
            self.conv31_weights_0,
            self.conv31_output_0,
            &self.multiplier_conv31,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv31_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv24_output.clone(),
            conv_kernel: self.conv31_weights.clone(),
            y: conv31_output.clone(),
            

            remainder: remainder_conv31.clone(),
            div: div_conv31.clone(),

            x_0: self.conv22_output_0,
            conv_kernel_0: self.conv31_weights_0,
            y_0: self.conv31_output_0,

            multiplier: self.multiplier_conv31,
            padding: self.padding,
        };
        conv31_circuit.generate_constraints(cs.clone())?;

        let cmp_res5 = relu4d_u8(&mut conv31_output.clone(), self.x_0);
        let relu5_circuit = ReLUWrapper{
            x: conv31_output.clone(),
            cmp_res: cmp_res5,
            y_0: self.conv31_output_0,
        };
        relu5_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv31 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        //----------------------------------------------------------------------
        let padded_conv31_output = padding_helper(conv31_output.clone(), self.padding);

        let mut conv32_output = vec![vec![vec![vec![0u8; padded_conv31_output[0][0][0].len() - self.conv32_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv31_output[0][0].len() - self.conv32_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv32_weights.len()]; //number of conv kernels
        padded_conv31_output.len()]; //input (image) batch size

        let (remainder_conv32, div_conv32) = vec_conv_with_remainder_u8(
            &padded_conv31_output,
            &self.conv32_weights,
            &mut conv32_output,
            self.conv31_output_0,
            self.conv32_weights_0,
            self.conv32_output_0,
            &self.multiplier_conv32,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv32_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv31_output.clone(),
            conv_kernel: self.conv32_weights.clone(),
            y: conv32_output.clone(),
            

            remainder: remainder_conv32.clone(),
            div: div_conv32.clone(),

            x_0: self.conv31_output_0,
            conv_kernel_0: self.conv32_weights_0,
            y_0: self.conv32_output_0,

            multiplier: self.multiplier_conv32.clone(),
            padding: self.padding,
        };
        conv32_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv32 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        // residual add 3 =================================================================================================================
        let mut residual_2_output = vec![vec![vec![vec![0u8; conv32_output[0][0][0].len()];  // w - kernel_size + 1
                                            conv32_output[0][0].len()]; // h - kernel_size + 1
                                            conv32_output[0].len()]; //number of conv kernels
                                            conv32_output.len()]; //input (image) batch size

        let (residual3_remainder, residual3_div) = residual_add_plaintext(&residual2, &conv32_output,&mut residual_2_output, self.residual3_input_0, 
            self.conv32_output_0, self.residual3_output_0, &self.residual3_multiplier, &self.multiplier_conv32);

        let residual3_circuit = ResidualAddCircuit{
        input1: residual2.clone(),
        input2: conv32_output.clone(),
        output: residual_2_output.clone(),

        remainder: residual3_remainder.clone(),
        div: residual3_div.clone(),

        input1_0: self.residual3_input_0,
        input2_0: self.conv32_output_0,
        output_0: self.residual3_output_0,

        multiplier_1: self.residual3_multiplier,
        multiplier_2: self.multiplier_conv32,
        knit_encoding: self.knit_encoding,
        };
        residual3_circuit.generate_constraints(cs.clone())?;

        println!(
        "Number of constraints for residual3 {} accumulated constraints {}",
        cs.num_constraints() - _cir_number,
        cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let cmp_res6 = relu4d_u8(&mut residual_2_output.clone(), self.x_0);
        let relu6_circuit = ReLUWrapper{
            x: residual_2_output.clone(),
            cmp_res: cmp_res6,
            y_0: self.residual3_output_0,
        };
        relu6_circuit.generate_constraints(cs.clone())?;

        let padded_conv32_output = padding_helper(residual_2_output.clone(), self.padding);

        // =================================================================================================================================



        //----------------------------------------------------------------------
        // let padded_conv32_output = padding_helper(conv32_output, self.padding);

        let mut conv33_output = vec![vec![vec![vec![0u8; padded_conv32_output[0][0][0].len() - self.conv33_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv32_output[0][0].len() - self.conv33_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv33_weights.len()]; //number of conv kernels
        padded_conv32_output.len()]; //input (image) batch size

        let (remainder_conv33, div_conv33) = vec_conv_with_remainder_u8(
            &padded_conv32_output,
            &self.conv33_weights,
            &mut conv33_output,
            self.conv32_output_0,
            self.conv33_weights_0,
            self.conv33_output_0,
            &self.multiplier_conv33,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv33_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv32_output.clone(),
            conv_kernel: self.conv33_weights.clone(),
            y: conv33_output.clone(),
            

            remainder: remainder_conv33.clone(),
            div: div_conv33.clone(),

            x_0: self.conv32_output_0,
            conv_kernel_0: self.conv33_weights_0,
            y_0: self.conv33_output_0,

            multiplier: self.multiplier_conv33,
            padding: self.padding,
        };
        conv33_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv33 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let cmp_res7 = relu4d_u8(&mut conv33_output.clone(), self.x_0);
        let relu7_circuit = ReLUWrapper{
            x: conv33_output.clone(),
            cmp_res: cmp_res7,
            y_0: self.conv33_output_0,
        };
        relu7_circuit.generate_constraints(cs.clone())?;

        //----------------------------------------------------------------------
        let padded_conv33_output = padding_helper(conv33_output.clone(), self.padding);

        let mut conv34_output = vec![vec![vec![vec![0u8; padded_conv33_output[0][0][0].len() - self.conv34_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv33_output[0][0].len() - self.conv34_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv34_weights.len()]; //number of conv kernels
        padded_conv33_output.len()]; //input (image) batch size

        let (remainder_conv34, div_conv34) = vec_conv_with_remainder_u8(
            &padded_conv33_output,
            &self.conv34_weights,
            &mut conv34_output,
            self.conv33_output_0,
            self.conv34_weights_0,
            self.conv34_output_0,
            &self.multiplier_conv34,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv34_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv33_output.clone(),
            conv_kernel: self.conv34_weights.clone(),
            y: conv34_output.clone(),
            

            remainder: remainder_conv34.clone(),
            div: div_conv34.clone(),

            x_0: self.conv33_output_0,
            conv_kernel_0: self.conv34_weights_0,
            y_0: self.conv34_output_0,

            multiplier: self.multiplier_conv34.clone(),
            padding: self.padding,
        };
        conv34_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv34 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        // residual add 4 =================================================================================================================
        let mut residual_3_output = vec![vec![vec![vec![0u8; conv34_output[0][0][0].len()];  // w - kernel_size + 1
                                        conv34_output[0][0].len()]; // h - kernel_size + 1
                                        conv34_output[0].len()]; //number of conv kernels
                                        conv34_output.len()]; //input (image) batch size


        let (residual4_remainder, residual4_div) = residual_add_plaintext(&residual_2_output, &conv34_output,&mut residual_3_output, self.residual4_input_0, 
            self.conv34_output_0, self.residual4_output_0, &self.residual4_multiplier, &self.multiplier_conv34);

        let residual4_circuit = ResidualAddCircuit{
        input1: residual_2_output.clone(),
        input2: conv34_output.clone(),
        output: residual_3_output.clone(),

        remainder: residual4_remainder.clone(),
        div: residual4_div.clone(),

        input1_0: self.residual4_input_0,
        input2_0: self.conv34_output_0,
        output_0: self.residual4_output_0,

        multiplier_1: self.residual4_multiplier,
        multiplier_2: self.multiplier_conv34,
        knit_encoding: self.knit_encoding,
        };
        residual4_circuit.generate_constraints(cs.clone())?;

        println!(
        "Number of constraints for residual4 {} accumulated constraints {}",
        cs.num_constraints() - _cir_number,
        cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let cmp_res8 = relu4d_u8(&mut residual_3_output.clone(), self.x_0);
        let relu8_circuit = ReLUWrapper{
            x: residual_3_output.clone(),
            cmp_res: cmp_res8,
            y_0: self.residual4_output_0,
        };
        relu8_circuit.generate_constraints(cs.clone())?;

        let (avg_pool3_output, avg3_remainder) = avg_pool_with_remainder_scala_u8(&residual_3_output.clone(), 2);

        // =================================================================================================================================

        let avg_pool3_circuit = AvgPoolWrapper {
            // x: conv34_output.clone(),
            x: residual_3_output.clone(),
            y: avg_pool3_output.clone(),
            kernel_size: 2,
            remainder: avg3_remainder.clone(),
        };
        avg_pool3_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        // residual 1*1 conv 3 =================================================================================
        let mut residual3 = vec![vec![vec![vec![0u8; avg_pool3_output[0][0][0].len()];  // w - kernel_size + 1
                                avg_pool3_output[0][0].len()]; // h - kernel_size + 1
                                self.conv42_weights.len()]; //number of conv kernels
                                avg_pool3_output.len()]; //input (image) batch size

        let (remainder_residual3, div_residual3) = vec_conv_with_remainder_u8(
            &avg_pool3_output,
            &self.conv_residuel3_kernel,
            &mut residual3,
            self.conv34_output_0,
            self.conv_residuel3_weights_0,
            self.conv_residuel3_output_0,
            &self.multiplier_residuel3,
        );

        let residual_conv3_circuit = ConvCircuitOp3Wrapper {
            x: avg_pool3_output.clone(),
            conv_kernel: self.conv_residuel3_kernel.clone(),
            y: residual3.clone(),

            remainder: remainder_residual3.clone(),
            div: div_residual3.clone(),

            x_0: self.conv34_output_0,
            conv_kernel_0: self.conv_residuel3_weights_0,
            y_0: self.conv_residuel3_output_0,

            multiplier: self.multiplier_residuel3,
            padding: self.padding,
        };
        residual_conv3_circuit.generate_constraints(cs.clone())?;



        //----------------------------------------------------------------------
        let padded_conv34_output = padding_helper(avg_pool3_output.clone(), self.padding);

        let mut conv41_output = vec![vec![vec![vec![0u8; padded_conv34_output[0][0][0].len() - self.conv41_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv34_output[0][0].len() - self.conv41_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv41_weights.len()]; //number of conv kernels
    padded_conv34_output.len()]; //input (image) batch size

        let (remainder_conv41, div_conv41) = vec_conv_with_remainder_u8(
            &padded_conv34_output,
            &self.conv41_weights,
            &mut conv41_output,
            self.conv34_output_0,
            self.conv41_weights_0,
            self.conv41_output_0,
            &self.multiplier_conv41,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv41_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv34_output.clone(),
            conv_kernel: self.conv41_weights.clone(),
            y: conv41_output.clone(),
            

            remainder: remainder_conv41.clone(),
            div: div_conv41.clone(),

            x_0: self.conv34_output_0,
            conv_kernel_0: self.conv41_weights_0,
            y_0: self.conv41_output_0,

            multiplier: self.multiplier_conv41,
            padding: self.padding,
        };
        conv41_circuit.generate_constraints(cs.clone())?;

        let cmp_res10 = relu4d_u8(&mut conv41_output.clone(), self.x_0);
        let relu10_circuit = ReLUWrapper{
            x: conv41_output.clone(),
            cmp_res: cmp_res10,
            y_0: self.conv41_output_0,
        };
        relu10_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv41 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let cmp_res9 = relu4d_u8(&mut conv41_output.clone(), self.x_0);
        let relu9_circuit = ReLUWrapper{
            x: conv41_output.clone(),
            cmp_res: cmp_res9,
            y_0: self.conv41_output_0,
        };
        relu9_circuit.generate_constraints(cs.clone())?;

        //----------------------------------------------------------------------
        let padded_conv41_output = padding_helper(conv41_output.clone(), self.padding);
        let mut conv42_output = vec![vec![vec![vec![0u8; padded_conv41_output[0][0][0].len() - self.conv42_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv41_output[0][0].len() - self.conv42_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv42_weights.len()]; //number of conv kernels
    padded_conv41_output.len()]; //input (image) batch size

        let (remainder_conv42, div_conv42) = vec_conv_with_remainder_u8(
            &padded_conv41_output,
            &self.conv42_weights,
            &mut conv42_output,
            self.conv41_output_0,
            self.conv42_weights_0,
            self.conv42_output_0,
            &self.multiplier_conv42,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv42_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv41_output.clone(),
            conv_kernel: self.conv42_weights.clone(),
            y: conv42_output.clone(),
            

            remainder: remainder_conv42.clone(),
            div: div_conv42.clone(),

            x_0: self.conv41_output_0,
            conv_kernel_0: self.conv42_weights_0,
            y_0: self.conv42_output_0,

            multiplier: self.multiplier_conv42.clone(),
            padding: 0,
        };
        conv42_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv42 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();



        // residual add 5 =================================================================================================================
        let mut residual_4_output = vec![vec![vec![vec![0u8; conv42_output[0][0][0].len()];  // w - kernel_size + 1
                                conv42_output[0][0].len()]; // h - kernel_size + 1
                                conv42_output[0].len()]; //number of conv kernels
                                conv42_output.len()]; //input (image) batch size

        let (residual5_remainder, residual5_div) = residual_add_plaintext(&residual3, &conv42_output,&mut residual_4_output, self.residual5_input_0, 
            self.conv42_output_0, self.residual5_output_0, &self.residual5_multiplier, &self.multiplier_conv42);

        let residual5_circuit = ResidualAddCircuit{
        input1: residual3.clone(),
        input2: conv42_output.clone(),
        output: residual_4_output.clone(),

        remainder: residual5_remainder.clone(),
        div: residual5_div.clone(),

        input1_0: self.residual5_input_0,
        input2_0: self.conv42_output_0,
        output_0: self.residual5_output_0,

        multiplier_1: self.residual5_multiplier,
        multiplier_2: self.multiplier_conv42,
        knit_encoding: self.knit_encoding,
        };
        residual5_circuit.generate_constraints(cs.clone())?;

        println!(
        "Number of constraints for residual5 {} accumulated constraints {}",
        cs.num_constraints() - _cir_number,
        cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let cmp_res11 = relu4d_u8(&mut residual_4_output.clone(), self.x_0);
        let relu11_circuit = ReLUWrapper{
            x: residual_4_output.clone(),
            cmp_res: cmp_res11,
            y_0: self.residual5_output_0,
        };
        relu11_circuit.generate_constraints(cs.clone())?;

        let padded_conv42_output = padding_helper(residual_4_output.clone(), self.padding);
        // =================================================================================================================================



        //----------------------------------------------------------------------
        // let padded_conv42_output = padding_helper(conv42_output, self.padding);

        let mut conv43_output = vec![vec![vec![vec![0u8; padded_conv42_output[0][0][0].len() - self.conv43_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv42_output[0][0].len() - self.conv43_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv43_weights.len()]; //number of conv kernels
    padded_conv42_output.len()]; //input (image) batch size

        let (remainder_conv43, div_conv43) = vec_conv_with_remainder_u8(
            &padded_conv42_output,
            &self.conv43_weights,
            &mut conv43_output,
            self.conv42_output_0,
            self.conv43_weights_0,
            self.conv43_output_0,
            &self.multiplier_conv43,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv43_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv42_output.clone(),
            conv_kernel: self.conv43_weights.clone(),
            y: conv43_output.clone(),
            

            remainder: remainder_conv43.clone(),
            div: div_conv43.clone(),

            x_0: self.conv42_output_0,
            conv_kernel_0: self.conv43_weights_0,
            y_0: self.conv43_output_0,

            multiplier: self.multiplier_conv43,
            padding: self.padding,
        };
        conv43_circuit.generate_constraints(cs.clone())?;

        let cmp_res12 = relu4d_u8(&mut conv43_output.clone(), self.x_0);
        let relu12_circuit = ReLUWrapper{
            x: conv43_output.clone(),
            cmp_res: cmp_res12,
            y_0: self.conv43_output_0,
        };
        relu12_circuit.generate_constraints(cs.clone())?;
        
        //----------------------------------------------------------------------
        let padded_conv43_output = padding_helper(conv43_output.clone(), self.padding);
        let mut conv44_output = vec![vec![vec![vec![0u8; padded_conv43_output[0][0][0].len() - self.conv44_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv43_output[0][0].len() - self.conv43_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv44_weights.len()]; //number of conv kernels
    padded_conv43_output.len()]; //input (image) batch size

        let (remainder_conv44, div_conv44) = vec_conv_with_remainder_u8(
            &padded_conv43_output,
            &self.conv44_weights,
            &mut conv44_output,
            self.conv43_output_0,
            self.conv44_weights_0,
            self.conv44_output_0,
            &self.multiplier_conv44,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv44_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv43_output.clone(),
            conv_kernel: self.conv44_weights.clone(),
            y: conv44_output.clone(),
            

            remainder: remainder_conv44.clone(),
            div: div_conv44.clone(),

            x_0: self.conv43_output_0,
            conv_kernel_0: self.conv44_weights_0,
            y_0: self.conv44_output_0,

            multiplier: self.multiplier_conv44.clone(),
            padding: 0,
        };
        conv44_circuit.generate_constraints(cs.clone())?;

        let cmp_res13 = relu4d_u8(&mut conv44_output.clone(), self.x_0);
        let relu13_circuit = ReLUWrapper{
            x: conv44_output.clone(),
            cmp_res: cmp_res13,
            y_0: self.conv44_output_0,
        };
        relu13_circuit.generate_constraints(cs.clone())?;

        
        // residual add 6 =================================================================================================================
        let mut residual_5_output = vec![vec![vec![vec![0u8; conv44_output[0][0][0].len()];  // w - kernel_size + 1
                                    conv44_output[0][0].len()]; // h - kernel_size + 1
                                    conv44_output[0].len()]; //number of conv kernels
                                    conv44_output.len()]; //input (image) batch size

        let (residual6_remainder, residual6_div) = residual_add_plaintext(&residual_4_output, &conv44_output,&mut residual_5_output, self.residual6_input_0, 
            self.conv44_output_0, self.residual6_output_0, &self.residual6_multiplier, &self.multiplier_conv44);

        let residual6_circuit = ResidualAddCircuit{
        input1: residual_4_output.clone(),
        input2: conv44_output.clone(),
        output: residual_5_output.clone(),

        remainder: residual6_remainder.clone(),
        div: residual6_div.clone(),

        input1_0: self.residual6_input_0,
        input2_0: self.conv44_output_0,
        output_0: self.residual6_output_0,

        multiplier_1: self.residual6_multiplier,
        multiplier_2: self.multiplier_conv44,
        knit_encoding: self.knit_encoding,
        };
        residual6_circuit.generate_constraints(cs.clone())?;

        println!(
        "Number of constraints for residual6 {} accumulated constraints {}",
        cs.num_constraints() - _cir_number,
        cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let (avg_pool4_output, avg4_remainder) = avg_pool_with_remainder_scala_u8(&residual_5_output, 2);

        // =================================================================================================================================

        
        //avg pool3 layer
        // let (avg_pool4_output, avg4_remainder) =
        //     avg_pool_with_remainder_scala_u8(&conv44_output, 2);
        let avg_pool4_circuit = AvgPoolWrapper {
            // x: conv44_output.clone(),
            x: residual_5_output.clone(),
            y: avg_pool4_output.clone(),
            kernel_size: 2,
            remainder: avg4_remainder.clone(),
        };
        avg_pool4_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool4 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        // residual 1*1 conv 4 =================================================================================
        let mut residual4 = vec![vec![vec![vec![0u8; avg_pool4_output[0][0][0].len()];  // w - kernel_size + 1
                avg_pool4_output[0][0].len()]; // h - kernel_size + 1
                self.conv52_weights.len()]; //number of conv kernels
                avg_pool4_output.len()]; //input (image) batch size

        let (remainder_residual4, div_residual4) = vec_conv_with_remainder_u8(
            &avg_pool4_output,
            &self.conv_residuel4_kernel,
            &mut residual4,
            self.conv44_output_0,
            self.conv_residuel4_weights_0,
            self.conv_residuel4_output_0,
            &self.multiplier_residuel4,
        );

        let residual_conv4_circuit = ConvCircuitOp3Wrapper {
            x: avg_pool4_output.clone(),
            conv_kernel: self.conv_residuel4_kernel.clone(),
            y: residual4.clone(),

            remainder: remainder_residual4.clone(),
            div: div_residual4.clone(),

            x_0: self.conv44_output_0,
            conv_kernel_0: self.conv_residuel4_weights_0,
            y_0: self.conv_residuel4_output_0,

            multiplier: self.multiplier_residuel4,
            padding: self.padding,
        };
        residual_conv4_circuit.generate_constraints(cs.clone())?;
        
        //----------------------------------------------------------------------
       
        let padded_conv44_output = padding_helper(avg_pool4_output.clone(), self.padding);

        let mut conv51_output = vec![vec![vec![vec![0u8; padded_conv44_output[0][0][0].len() - self.conv51_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv44_output[0][0].len() - self.conv51_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv51_weights.len()]; //number of conv kernels
        padded_conv44_output.len()]; //input (image) batch size

        let (remainder_conv51, div_conv51) = vec_conv_with_remainder_u8(
            &padded_conv44_output,
            &self.conv51_weights,
            &mut conv51_output,
            self.conv44_output_0,
            self.conv51_weights_0,
            self.conv51_output_0,
            &self.multiplier_conv51,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv51_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv44_output.clone(),
            conv_kernel: self.conv51_weights.clone(),
            y: conv51_output.clone(),
            

            remainder: remainder_conv51.clone(),
            div: div_conv51.clone(),

            x_0: self.conv44_output_0,
            conv_kernel_0: self.conv51_weights_0,
            y_0: self.conv51_output_0,

            multiplier: self.multiplier_conv51,
            padding: self.padding,
        };
        conv51_circuit.generate_constraints(cs.clone())?;

        let cmp_res14 = relu4d_u8(&mut conv51_output.clone(), self.x_0);
        let relu14_circuit = ReLUWrapper{
            x: conv51_output.clone(),
            cmp_res: cmp_res14,
            y_0: self.conv51_output_0,
        };
        relu14_circuit.generate_constraints(cs.clone())?;


        //----------------------------------------------------------------------
        let padded_conv51_output = padding_helper(conv51_output.clone(), self.padding);

        let mut conv52_output = vec![vec![vec![vec![0u8; padded_conv51_output[0][0][0].len() - self.conv52_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv51_output[0][0].len() - self.conv52_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv52_weights.len()]; //number of conv kernels
    padded_conv51_output.len()]; //input (image) batch size

        let (remainder_conv52, div_conv52) = vec_conv_with_remainder_u8(
            &padded_conv51_output,
            &self.conv52_weights,
            &mut conv52_output,
            self.conv51_output_0,
            self.conv52_weights_0,
            self.conv52_output_0,
            &self.multiplier_conv52,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv52_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv51_output.clone(),
            conv_kernel: self.conv52_weights.clone(),
            y: conv52_output.clone(),
            

            remainder: remainder_conv52.clone(),
            div: div_conv52.clone(),

            x_0: self.conv51_output_0,
            conv_kernel_0: self.conv52_weights_0,
            y_0: self.conv52_output_0,

            multiplier: self.multiplier_conv52.clone(),
            padding: self.padding,
        };
        conv52_circuit.generate_constraints(cs.clone())?;

      
        let cmp_res15 = relu4d_u8(&mut conv52_output.clone(), self.x_0);
        let relu15_circuit = ReLUWrapper{
            x: conv52_output.clone(),
            cmp_res: cmp_res15,
            y_0: self.conv52_output_0,
        };
        relu15_circuit.generate_constraints(cs.clone())?;

        // residual add 7 =================================================================================================================
        let mut residual_6_output = vec![vec![vec![vec![0u8; conv52_output[0][0][0].len()];  // w - kernel_size + 1
                                        conv52_output[0][0].len()]; // h - kernel_size + 1
                                        conv52_output[0].len()]; //number of conv kernels
                                        conv52_output.len()]; //input (image) batch size


        let (residual7_remainder, residual7_div) = residual_add_plaintext(&residual4, &conv52_output,&mut residual_6_output, self.residual7_input_0, 
            self.conv52_output_0, self.residual7_output_0, &self.residual7_multiplier, &self.multiplier_conv52);
            
        let residual7_circuit = ResidualAddCircuit{
        input1: residual4.clone(),
        input2: conv52_output.clone(),
        output: residual_6_output.clone(),

        remainder: residual7_remainder.clone(),
        div: residual7_div.clone(),

        input1_0: self.residual7_input_0,
        input2_0: self.conv52_output_0,
        output_0: self.residual7_output_0,

        multiplier_1: self.residual7_multiplier,
        multiplier_2: self.multiplier_conv52,
        knit_encoding: self.knit_encoding,
        };
        residual7_circuit.generate_constraints(cs.clone())?;

        println!(
        "Number of constraints for residual7 {} accumulated constraints {}",
        cs.num_constraints() - _cir_number,
        cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let padded_conv52_output = padding_helper(residual_6_output.clone(), self.padding);
        // =================================================================================================================================
      
      
        //----------------------------------------------------------------------
        // let padded_conv52_output = padding_helper(conv52_output, self.padding);

        let mut conv53_output = vec![vec![vec![vec![0u8; padded_conv52_output[0][0][0].len() - self.conv53_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv52_output[0][0].len() - self.conv53_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv53_weights.len()]; //number of conv kernels
    padded_conv52_output.len()]; //input (image) batch size

        let (remainder_conv53, div_conv53) = vec_conv_with_remainder_u8(
            &padded_conv52_output,
            &self.conv53_weights,
            &mut conv53_output,
            self.conv52_output_0,
            self.conv53_weights_0,
            self.conv53_output_0,
            &self.multiplier_conv53,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv53_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv52_output.clone(),
            conv_kernel: self.conv53_weights.clone(),
            y: conv53_output.clone(),
            

            remainder: remainder_conv53.clone(),
            div: div_conv53.clone(),

            x_0: self.conv52_output_0,
            conv_kernel_0: self.conv53_weights_0,
            y_0: self.conv53_output_0,

            multiplier: self.multiplier_conv53,
            padding: self.padding,
        };
        conv53_circuit.generate_constraints(cs.clone())?;

        let cmp_res16 = relu4d_u8(&mut conv53_output.clone(), self.x_0);
        let relu16_circuit = ReLUWrapper{
            x: conv53_output.clone(),
            cmp_res: cmp_res16,
            y_0: self.conv53_output_0,
        };
        relu16_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv53 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        //----------------------------------------------------------------------
        let padded_conv53_output = padding_helper(conv53_output.clone(), self.padding);

        let mut conv54_output = vec![vec![vec![vec![0u8; padded_conv53_output[0][0][0].len() - self.conv54_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv53_output[0][0].len() - self.conv54_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv54_weights.len()]; //number of conv kernels
    padded_conv53_output.len()]; //input (image) batch size

        let (remainder_conv54, div_conv54) = vec_conv_with_remainder_u8(
            &padded_conv53_output,
            &self.conv54_weights,
            &mut conv54_output,
            self.conv53_output_0,
            self.conv54_weights_0,
            self.conv54_output_0,
            &self.multiplier_conv54,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv54_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv53_output.clone(),
            conv_kernel: self.conv54_weights.clone(),
            y: conv54_output.clone(),
            

            remainder: remainder_conv54.clone(),
            div: div_conv54.clone(),

            x_0: self.conv53_output_0,
            conv_kernel_0: self.conv54_weights_0,
            y_0: self.conv54_output_0,

            multiplier: self.multiplier_conv54.clone(),
            padding: self.padding,
        };
        conv54_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv54 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();



        // residual add 8 =================================================================================================================
        let mut residual_7_output = vec![vec![vec![vec![0u8; conv54_output[0][0][0].len()];  // w - kernel_size + 1
                                conv54_output[0][0].len()]; // h - kernel_size + 1
                                conv54_output[0].len()]; //number of conv kernels
                                conv54_output.len()]; //input (image) batch size

        let (residual8_remainder, residual8_div) = residual_add_plaintext(&residual_6_output, &conv54_output,&mut residual_7_output, self.residual8_input_0, 
            self.conv54_output_0, self.residual8_output_0, &self.residual8_multiplier, &self.multiplier_conv54);
            
        let residual8_circuit = ResidualAddCircuit{
        input1: residual_6_output.clone(),
        input2: conv54_output.clone(),
        output: residual_7_output.clone(),

        remainder: residual8_remainder.clone(),
        div: residual8_div.clone(),

        input1_0: self.residual8_input_0,
        input2_0: self.conv54_output_0,
        output_0: self.residual8_output_0,

        multiplier_1: self.residual8_multiplier,
        multiplier_2: self.multiplier_conv54,
        knit_encoding: self.knit_encoding,
        };
        residual8_circuit.generate_constraints(cs.clone())?;

        println!(
        "Number of constraints for residual8 {} accumulated constraints {}",
        cs.num_constraints() - _cir_number,
        cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let cmp_res17 = relu4d_u8(&mut residual_7_output.clone(), self.x_0);
        let relu17_circuit = ReLUWrapper{
            x: residual_7_output.clone(),
            cmp_res: cmp_res17,
            y_0: self.residual8_output_0,
        };
        relu17_circuit.generate_constraints(cs.clone())?;

        let (avg_pool5_output, avg5_remainder) = avg_pool_with_remainder_scala_u8(&residual_7_output, 4);

        // =================================================================================================================================

        let avg_pool5_circuit = AvgPoolWrapper {
            // x: conv44_output.clone(),
            x: residual_7_output.clone(),
            y: avg_pool5_output.clone(),
            kernel_size: 4,
            remainder: avg5_remainder.clone(),
        };
        avg_pool5_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool4 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //---------------------------------------------------------------------

        // let mut transformed_conv54_output = vec![
        //     vec![
        //         0u8;
        //         conv54_output[0].len()
        //             * conv54_output[0][0].len()
        //             * conv54_output[0][0][0].len()
        //     ];
        //     conv54_output.len()
        // ];
        // for i in 0..conv54_output.len() {
        //     let mut counter = 0;
        //     for j in 0..conv54_output[0].len() {
        //         for p in 0..conv54_output[0][0].len() {
        //             for q in 0..conv54_output[0][0][0].len() {
        //                 transformed_conv54_output[i][counter] = conv54_output[i][j][p][q];
        //                 counter += 1;
        //             }
        //         }
        //     }
        // }
        let mut transformed_avg_pool5_output_output =
        vec![
            vec![
                0u8;
                avg_pool5_output[0].len() * avg_pool5_output[0][0].len() * avg_pool5_output[0][0][0].len()
            ];
            avg_pool5_output.len()
        ];
    
        for i in 0..avg_pool5_output.len() {
            let mut counter = 0;
            for j in 0..avg_pool5_output[0].len() {
                for p in 0..avg_pool5_output[0][0].len() {
                    for q in 0..avg_pool5_output[0][0][0].len() {
                        transformed_avg_pool5_output_output[i][counter] = avg_pool5_output[i][j][p][q];
                        counter += 1;
                    }
                }
            }
        }

        //layer 4 :
        //FC1 -> relu
        let mut fc1_output = vec![vec![0u8; self.fc1_weights.len()];  // channels
        transformed_avg_pool5_output_output.len()]; //batch size
        let fc1_weight_ref: Vec<&[u8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

        for i in 0..transformed_avg_pool5_output_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            let (remainder_fc1, div_fc1) = vec_mat_mul_with_remainder_u8(
                &transformed_avg_pool5_output_output[i],
                fc1_weight_ref[..].as_ref(),
                &mut fc1_output[i],
                self.conv54_output_0,
                self.fc1_weights_0,
                self.fc1_output_0,
                &self.multiplier_fc1.clone(),
            );

            //because the vector dot product is too short. SIMD can not reduce the number of contsraints
            let fc1_circuit = FCWrapper {
                x: transformed_avg_pool5_output_output[i].clone(),
                weights: self.fc1_weights.clone(),
                y: fc1_output[i].clone(),
                remainder: remainder_fc1.clone(),
                div: div_fc1.clone(),
                x_0: self.conv54_output_0,
                weights_0: self.fc1_weights_0,
                y_0: self.fc1_output_0,

                multiplier: self.multiplier_fc1.clone(),
            };
            fc1_circuit.generate_constraints(cs.clone())?;

            println!(
                "Number of constraints FC1 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }

        // z commitment
        let flattened_z1d: Vec<u8> = fc1_output.clone().into_iter().flatten().collect();
        let z_com_circuit = PedersenComCircuit {
            param: self.params.clone(),
            input: flattened_z1d.clone(),
            open: self.z_open,
            commit: self.z_com,
        };
        z_com_circuit.generate_constraints(cs.clone())?;
        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for z commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        Ok(())
    }
}
