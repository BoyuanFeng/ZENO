use crate::argmax_circuit::*;
use crate::avg_pool_circuit::*;
use crate::conv_circuit::*;
use crate::cosine_circuit::*;
use crate::mul_circuit::*;
use crate::relu_circuit::*;
use crate::both_private_circuit::*;
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

fn convert_4d_vector_into_1d(vec: Vec<Vec<Vec<Vec<u8>>>>) -> Vec<u8> {
    let mut res = Vec::new();
    for i in 0..vec.len() {
        for j in 0..vec[0].len() {
            for k in 0..vec[0][0].len() {
                res.extend(&vec[i][j][k]);
            }
        }
    }
    res
}

fn padding_helper(input: Vec<Vec<Vec<Vec<u8>>>>, padding: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    // @input param:
    // input shape: [batch_size, channel_num, width, height]
    // padding: an integer.
    // output shape: [batch_size, channel_num, width+2*padding, height+2*padding]
    let mut padded_output: Vec<Vec<Vec<Vec<u8>>>> =
        vec![
            vec![
                vec![vec![0; input[0][0][0].len() + padding * 2]; input[0][0].len() + padding * 2];
                input[0].len()
            ];
            input.len()
        ];
    for i in 0..input.len() {
        for j in 0..input[0].len() {
            for k in padding..(padded_output[0][0].len() - padding) {
                for m in padding..(padded_output[0][0][0].len() - padding) {
                    padded_output[i][j][k][m] = input[i][j][k-padding][m-padding];
                }
            }
        }
    }

    padded_output
}

fn print_size(input: Vec<Vec<Vec<Vec<u8>>>>){
    let size1 = input.len();
    let size2 = input[0].len();
    let size3 = input[0][0].len();
    let size4 = input[0][0][0].len();
    println!("size:[{},{},{},{}]",size1,size2,size3,size4);
}

#[derive(Clone)]
pub struct VGGCircuitCircuitBothPrivate {
    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,

    pub conv11_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv11_open:PedersenRandomness,
    pub conv11_com_vec: Vec<PedersenCommitment>,

    pub conv12_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv12_open:PedersenRandomness,
    pub conv12_com_vec: Vec<PedersenCommitment>,

    pub conv21_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv21_open:PedersenRandomness,
    pub conv21_com_vec: Vec<PedersenCommitment>,

    pub conv22_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv22_open:PedersenRandomness,
    pub conv22_com_vec: Vec<PedersenCommitment>,

    pub conv31_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv31_open:PedersenRandomness,
    pub conv31_com_vec: Vec<PedersenCommitment>,

    pub conv32_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv32_open:PedersenRandomness,
    pub conv32_com_vec: Vec<PedersenCommitment>,

    pub conv33_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv33_open:PedersenRandomness,
    pub conv33_com_vec: Vec<PedersenCommitment>,

    pub conv41_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv41_open:PedersenRandomness,
    pub conv41_com_vec: Vec<PedersenCommitment>,

    pub conv42_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv42_open:PedersenRandomness,
    pub conv42_com_vec: Vec<PedersenCommitment>,

    pub conv43_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv43_open:PedersenRandomness,
    pub conv43_com_vec: Vec<PedersenCommitment>,

    pub conv51_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv51_open:PedersenRandomness,
    pub conv51_com_vec: Vec<PedersenCommitment>,

    pub conv52_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv52_open:PedersenRandomness,
    pub conv52_com_vec: Vec<PedersenCommitment>,

    pub conv53_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv53_open:PedersenRandomness,
    pub conv53_com_vec: Vec<PedersenCommitment>,

    pub fc1_weights: Vec<Vec<u8>>,
    pub fc1_weights_open:PedersenRandomness,
    pub fc1_weights_com_vec: Vec<PedersenCommitment>,

    pub fc2_weights: Vec<Vec<u8>>,
    pub fc2_weights_open:PedersenRandomness,
    pub fc2_weights_com_vec: Vec<PedersenCommitment>,

    pub fc3_weights: Vec<Vec<u8>>,
    pub fc3_weights_open:PedersenRandomness,
    pub fc3_weights_com_vec: Vec<PedersenCommitment>,

    //zero points for quantization.
    pub x_0: u8,
    pub conv11_output_0: u8,
    pub conv12_output_0: u8,

    pub conv21_output_0: u8,
    pub conv22_output_0: u8,

    pub conv31_output_0: u8,
    pub conv32_output_0: u8,
    pub conv33_output_0: u8,

    pub conv41_output_0: u8,
    pub conv42_output_0: u8,
    pub conv43_output_0: u8,

    pub conv51_output_0: u8,
    pub conv52_output_0: u8,
    pub conv53_output_0: u8,

    pub fc1_output_0: u8,
    pub fc2_output_0: u8, 
    pub fc3_output_0: u8, // which is also vgg output(z) zero point

    pub conv11_weights_0: u8,
    pub conv12_weights_0: u8,

    pub conv21_weights_0: u8,
    pub conv22_weights_0: u8,

    pub conv31_weights_0: u8,
    pub conv32_weights_0: u8,
    pub conv33_weights_0: u8,

    pub conv41_weights_0: u8,
    pub conv42_weights_0: u8,
    pub conv43_weights_0: u8,

    pub conv51_weights_0: u8,
    pub conv52_weights_0: u8,
    pub conv53_weights_0: u8,

    pub fc1_weights_0: u8,
    pub fc2_weights_0: u8,
    pub fc3_weights_0: u8,


    //multiplier for quantization
    pub multiplier_conv11: Vec<f32>,
    pub multiplier_conv12: Vec<f32>,

    pub multiplier_conv21: Vec<f32>,
    pub multiplier_conv22: Vec<f32>,

    pub multiplier_conv31: Vec<f32>,
    pub multiplier_conv32: Vec<f32>,
    pub multiplier_conv33: Vec<f32>,

    pub multiplier_conv41: Vec<f32>,
    pub multiplier_conv42: Vec<f32>,
    pub multiplier_conv43: Vec<f32>,

    pub multiplier_conv51: Vec<f32>,
    pub multiplier_conv52: Vec<f32>,
    pub multiplier_conv53: Vec<f32>,

    pub multiplier_fc1: Vec<f32>,
    pub multiplier_fc2: Vec<f32>,
    pub multiplier_fc3: Vec<f32>,

    //we do not need multiplier in relu and AvgPool layer
    pub z: Vec<Vec<u8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
    pub knit_encoding: bool,
}

impl ConstraintSynthesizer<Fq> for VGGCircuitCircuitBothPrivate {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("LeNet is setup mode: {}", cs.is_in_setup_mode());
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

        let output: Vec<Vec<u8>> = vgg_circuit_forward_u8(
            self.x.clone(),
            self.conv11_weights.clone(),
            self.conv12_weights.clone(),
            self.conv21_weights.clone(),
            self.conv22_weights.clone(),
            self.conv31_weights.clone(),
            self.conv32_weights.clone(),
            self.conv33_weights.clone(),
            self.conv41_weights.clone(),
            self.conv42_weights.clone(),
            self.conv43_weights.clone(),
            self.conv51_weights.clone(),
            self.conv52_weights.clone(),
            self.conv53_weights.clone(),

            self.fc1_weights.clone(),
            self.fc2_weights.clone(),
            self.fc3_weights.clone(),

            self.x_0,
            self.conv11_output_0,
            self.conv12_output_0,
            self.conv21_output_0,
            self.conv22_output_0,
            self.conv31_output_0,
            self.conv32_output_0,
            self.conv33_output_0,
            self.conv41_output_0,
            self.conv42_output_0,
            self.conv43_output_0,
            self.conv51_output_0,
            self.conv52_output_0,
            self.conv53_output_0,
            self.fc1_output_0,
            self.fc2_output_0,
            self.fc3_output_0,

            self.conv11_weights_0,
            self.conv12_weights_0,

            self.conv21_weights_0,
            self.conv22_weights_0,

            self.conv31_weights_0,
            self.conv32_weights_0,
            self.conv33_weights_0,

            self.conv41_weights_0,
            self.conv42_weights_0,
            self.conv43_weights_0,

            self.conv51_weights_0,
            self.conv52_weights_0,
            self.conv53_weights_0,

            self.fc1_weights_0,
            self.fc2_weights_0,
            self.fc3_weights_0,

            self.multiplier_conv11.clone(),
            self.multiplier_conv12.clone(),

            self.multiplier_conv21.clone(),
            self.multiplier_conv22.clone(),

            self.multiplier_conv31.clone(),
            self.multiplier_conv32.clone(),
            self.multiplier_conv33.clone(),

            self.multiplier_conv41.clone(),
            self.multiplier_conv42.clone(),
            self.multiplier_conv43.clone(),

            self.multiplier_conv51.clone(),
            self.multiplier_conv52.clone(),
            self.multiplier_conv53.clone(),

            self.multiplier_fc1.clone(),
            self.multiplier_fc2.clone(),
            self.multiplier_fc3.clone(),

        );
        // z commitment
        let flattened_z1d: Vec<u8> = output.clone().into_iter().flatten().collect();
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
        
        //==================================================================================================
        let len_per_commit = PERDERSON_WINDOW_NUM * PERDERSON_WINDOW_SIZE / 8; //for vec<u8> commitment

        let conv11_mat_1d = convert_4d_vector_into_1d(self.conv11_weights.clone());
        let num_of_commit_conv11 = conv11_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv11 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv11_mat_1d.len()) {
                tmp.push(conv11_mat_1d[j]);
            }
            let conv11_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv11_open.clone(),
                commit: self.conv11_com_vec[i],
            };
            conv11_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

       
        let conv12_mat_1d = convert_4d_vector_into_1d(self.conv12_weights.clone());
        let num_of_commit_conv12 = conv12_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_conv12 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, conv12_mat_1d.len()) {
                tmp.push(conv12_mat_1d[j]);
            }
            let conv12_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.conv12_open.clone(),
                commit: self.conv12_com_vec[i],
            };
            conv12_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );


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
            "Number of constraints for conv22 layer commitment {} accumulated constraints {}",
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
            "Number of constraints for conv31 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

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
            "Number of constraints for conv32 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

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
            "Number of constraints for conv33 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

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
            "Number of constraints for conv41 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

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
            "Number of constraints for conv42 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

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
            "Number of constraints for conv43 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

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
            "Number of constraints for conv51 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

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
            "Number of constraints for conv52 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );

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
            "Number of constraints for conv53 layer commitment {} accumulated constraints {}",
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

                
        // fc1 commit =========================================================================
        let fc2_mat_1d = convert_2d_vector_into_1d(self.fc2_weights.clone());
        let num_of_commit_fc2 = fc2_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_fc2 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, fc2_mat_1d.len()) {
                tmp.push(fc2_mat_1d[j]);
            }
            let fc2_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.fc2_weights_open.clone(),
                commit: self.fc2_weights_com_vec[i],
            };
            fc2_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );


        // fc1 commit =========================================================================
        let fc3_mat_1d = convert_2d_vector_into_1d(self.fc3_weights.clone());
        let num_of_commit_fc3 = fc3_mat_1d.len() / len_per_commit + 1;
        for i in 0..num_of_commit_fc3 {
            let mut tmp = Vec::new();
            for j in i * len_per_commit..min((i + 1) * len_per_commit, fc3_mat_1d.len()) {
                tmp.push(fc3_mat_1d[j]);
            }
            let fc3_com_circuit = PedersenComCircuit {
                param: self.params.clone(),
                input: tmp.clone(),
                open: self.fc3_weights_open.clone(),
                commit: self.fc3_weights_com_vec[i],
            };
            fc3_com_circuit.generate_constraints(cs.clone())?;
        }
        println!(
            "Number of constraints for conv21 layer commitment {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );


        //==================================================================================================
        //layer 1
        //conv1
        let padded_x = padding_helper(self.x.clone(), 1);
        let mut conv11_output = vec![vec![vec![vec![0u8; padded_x[0][0][0].len() - self.conv11_weights[0][0][0].len() + 1];  // w - kernel_size  + 1
                                            padded_x[0][0].len() - self.conv11_weights[0][0].len() + 1]; // h - kernel_size + 1
                                            self.conv11_weights.len()]; //number of conv kernels
                                            padded_x.len()]; //input (image) batch size
        let (remainder_conv11, div_conv11) = vec_conv_with_remainder_u8(
            &padded_x.clone(),
            &self.conv11_weights,
            &mut conv11_output,
            self.x_0,
            self.conv11_weights_0,
            self.conv11_output_0,
            &self.multiplier_conv11,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv11_circuit = ConvCircuitOp3Wrapper {
            x: padded_x.clone(),
            conv_kernel: self.conv11_weights.clone(),
            y: conv11_output.clone(),
            remainder: remainder_conv11.clone(),
            div: div_conv11.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv11_weights_0,
            y_0: self.conv11_output_0,

            multiplier: self.multiplier_conv11,
            padding: 0,
        };
        conv11_circuit.generate_constraints(cs.clone())?;

        let padded_conv11_output = padding_helper(conv11_output.clone(), 1);
        let mut conv12_output = vec![vec![vec![vec![0u8; padded_conv11_output[0][0][0].len() - self.conv12_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv11_output[0][0].len() - self.conv12_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv12_weights.len()]; //number of conv kernels
        padded_conv11_output.len()]; //input (image) batch size

        let (remainder_conv12, div_conv12) = vec_conv_with_remainder_u8(
        &padded_conv11_output.clone(),
        &self.conv12_weights,
        &mut conv12_output,
        self.conv11_output_0,
        self.conv12_weights_0,
        self.conv12_output_0,
        &self.multiplier_conv12,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv12_circuit = ConvCircuitOp3Wrapper {
            x: padded_conv11_output.clone(),
            conv_kernel: self.conv12_weights.clone(),
            y: conv12_output.clone(),
            remainder: remainder_conv12.clone(),
            div: div_conv12.clone(),

            x_0: self.conv11_output_0,
            conv_kernel_0: self.conv12_weights_0,
            y_0: self.conv12_output_0,

            multiplier: self.multiplier_conv12,
            padding: 0,
        };
        conv12_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv12 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg pool2 layer
        let (avg_pool1_output, avg1_remainder) = avg_pool_with_remainder_scala_u8(&conv12_output, 2);
        let avg_pool1_circuit = AvgPoolWrapper {
            x: conv12_output.clone(),
            y: avg_pool1_output.clone(),
            kernel_size: 2,
            remainder: avg1_remainder.clone(),
        };
        avg_pool1_circuit.generate_constraints(cs.clone())?;

//----------------------------------------------------------------------
        let padded_conv12_output = padding_helper(avg_pool1_output.clone(), 1);
        let mut conv21_output = vec![vec![vec![vec![0u8; padded_conv12_output[0][0][0].len() - self.conv21_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv12_output[0][0].len() - self.conv21_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv21_weights.len()]; //number of conv kernels
        padded_conv12_output.len()]; //input (image) batch size

        let (remainder_conv21, div_conv21) = vec_conv_with_remainder_u8(
        &padded_conv12_output.clone(),
        &self.conv21_weights,
        &mut conv21_output,
        self.conv12_output_0,
        self.conv21_weights_0,
        self.conv21_output_0,
        &self.multiplier_conv21,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv21_circuit = ConvCircuitOp3Wrapper {
        x: padded_conv12_output.clone(),
        conv_kernel: self.conv21_weights.clone(),
        y: conv21_output.clone(),
        remainder: remainder_conv21.clone(),
        div: div_conv21.clone(),

        x_0: self.conv12_output_0,
        conv_kernel_0: self.conv21_weights_0,
        y_0: self.conv21_output_0,

        multiplier: self.multiplier_conv21,
        padding: 0,
        };
        conv21_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv21 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //----------------------------------------------------------------------
        let padded_conv21_output = padding_helper(conv21_output.clone(), 1);
        let mut conv22_output = vec![vec![vec![vec![0u8; padded_conv12_output[0][0][0].len() - self.conv22_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv21_output[0][0].len() - self.conv22_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv22_weights.len()]; //number of conv kernels
        padded_conv21_output.len()]; //input (image) batch size

        let (remainder_conv22, div_conv22) = vec_conv_with_remainder_u8(
        &padded_conv21_output.clone(),
        &self.conv22_weights,
        &mut conv22_output,
        self.conv21_output_0,
        self.conv22_weights_0,
        self.conv22_output_0,
        &self.multiplier_conv22,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv22_circuit = ConvCircuitOp3Wrapper {
        x: padded_conv21_output.clone(),
        conv_kernel: self.conv22_weights.clone(),
        y: conv22_output.clone(),
        remainder: remainder_conv22.clone(),
        div: div_conv22.clone(),

        x_0: self.conv21_output_0,
        conv_kernel_0: self.conv22_weights_0,
        y_0: self.conv22_output_0,

        multiplier: self.multiplier_conv22,
        padding: 0,
        };
        conv22_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv22 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();



        //avg pool2 layer
        let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&conv22_output, 2);
        let avg_pool2_circuit = AvgPoolWrapper {
            x: conv22_output.clone(),
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

        //----------------------------------------------------------------------
        let padded_avg_pool2_output = padding_helper(avg_pool2_output.clone(), 1);
        let mut conv31_output = vec![vec![vec![vec![0u8; padded_avg_pool2_output[0][0][0].len() - self.conv21_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_avg_pool2_output[0][0].len() - self.conv31_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv31_weights.len()]; //number of conv kernels
        padded_avg_pool2_output.len()]; //input (image) batch size

        let (remainder_conv31, div_conv31) = vec_conv_with_remainder_u8(
        &padded_avg_pool2_output.clone(),
        &self.conv31_weights,
        &mut conv31_output,
        self.conv22_output_0,
        self.conv31_weights_0,
        self.conv31_output_0,
        &self.multiplier_conv31,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv31_circuit = ConvCircuitOp3Wrapper {
        x: padded_avg_pool2_output.clone(),
        conv_kernel: self.conv31_weights.clone(),
        y: conv31_output.clone(),
        remainder: remainder_conv31.clone(),
        div: div_conv31.clone(),

        x_0: self.conv22_output_0,
        conv_kernel_0: self.conv31_weights_0,
        y_0: self.conv31_output_0,

        multiplier: self.multiplier_conv31,
        padding: 0,

        };
        conv31_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv31 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();




        //----------------------------------------------------------------------
        let padded_conv31_output = padding_helper(conv31_output.clone(), 1);
        let mut conv32_output = vec![vec![vec![vec![0u8; padded_conv31_output[0][0][0].len() - self.conv32_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv31_output[0][0].len() - self.conv32_weights[0][0].len()+ 1]; // h - kernel_size+ 1
        self.conv32_weights.len()]; //number of conv kernels
        padded_conv31_output.len()]; //input (image) batch size

        let (remainder_conv32, div_conv32) = vec_conv_with_remainder_u8(
        &padded_conv31_output.clone(),
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

        multiplier: self.multiplier_conv32,
        padding: 0,
        };
        conv32_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv32 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        //----------------------------------------------------------------------
        let padded_conv32_output = padding_helper(conv32_output.clone(), 0);

        let mut conv33_output = vec![vec![vec![vec![0u8; padded_conv32_output[0][0][0].len()];  // w - kernel_size + 1
        padded_conv32_output[0][0].len()]; // h - kernel_size+ 1
        self.conv33_weights.len()]; //number of conv kernels
        padded_conv32_output.len()]; //input (image) batch size

        let (remainder_conv33, div_conv33) = vec_conv_with_remainder_u8(
        &padded_conv32_output.clone(),
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
        padding: 0,
        };
        conv33_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv33 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

    

        
    //avg pool3 layer
    let (avg_pool3_output, avg3_remainder) = avg_pool_with_remainder_scala_u8(&conv33_output, 2);
    let avg_pool3_circuit = AvgPoolWrapper {
        x: conv33_output.clone(),
        y: avg_pool3_output.clone(),
        kernel_size: 2,
        remainder: avg3_remainder.clone(),
    };
    avg_pool3_circuit.generate_constraints(cs.clone())?;

    //----------------------------------------------------------------------
    let padded_conv33_output = padding_helper(avg_pool3_output.clone(), 1);

    let mut conv41_output = vec![vec![vec![vec![0u8; padded_conv33_output[0][0][0].len() - self.conv41_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv33_output[0][0].len() - self.conv41_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv41_weights.len()]; //number of conv kernels
    padded_conv33_output.len()]; //input (image) batch size

    let (remainder_conv41, div_conv41) = vec_conv_with_remainder_u8(
    &padded_conv33_output.clone(),
    &self.conv41_weights,
    &mut conv41_output,
    self.conv33_output_0,
    self.conv41_weights_0,
    self.conv41_output_0,
    &self.multiplier_conv41,
    );

    //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
    let conv41_circuit = ConvCircuitOp3Wrapper {
    x: padded_conv33_output.clone(),
    conv_kernel: self.conv41_weights.clone(),
    y: conv41_output.clone(),
    remainder: remainder_conv41.clone(),
    div: div_conv41.clone(),

    x_0: self.conv33_output_0,
    conv_kernel_0: self.conv41_weights_0,
    y_0: self.conv41_output_0,

    multiplier: self.multiplier_conv41,
    padding: 0,
    };
    conv41_circuit.generate_constraints(cs.clone())?;

    // #[cfg(debug_assertion)]
    println!(
        "Number of constraints for Conv41 {} accumulated constraints {}",
        cs.num_constraints() - _cir_number,
        cs.num_constraints()
    );
    _cir_number = cs.num_constraints();



    //----------------------------------------------------------------------
    let padded_conv41_output = padding_helper(conv41_output.clone(), 1);

    let mut conv42_output = vec![vec![vec![vec![0u8; padded_conv41_output[0][0][0].len() - self.conv42_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv41_output[0][0].len() - self.conv42_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv42_weights.len()]; //number of conv kernels
    padded_conv41_output.len()]; //input (image) batch size

    let (remainder_conv42, div_conv42) = vec_conv_with_remainder_u8(
    &padded_conv41_output.clone(),
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

    multiplier: self.multiplier_conv42,
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


    //----------------------------------------------------------------------
    let padded_conv42_output = padding_helper(conv42_output.clone(), 0);

    let mut conv43_output = vec![vec![vec![vec![0u8; padded_conv42_output[0][0][0].len()];  // w - kernel_size + 1
    padded_conv42_output[0][0].len()]; // h - kernel_size+ 1
    self.conv43_weights.len()]; //number of conv kernels
    padded_conv42_output.len()]; //input (image) batch size

    let (remainder_conv43, div_conv43) = vec_conv_with_remainder_u8(
    &padded_conv42_output.clone(),
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
    padding: 0,
    };
    conv43_circuit.generate_constraints(cs.clone())?;

    //avg pool4 layer
    let (avg_pool4_output, avg4_remainder) = avg_pool_with_remainder_scala_u8(&conv43_output, 2);
    let avg_pool4_circuit = AvgPoolWrapper {
        x: conv43_output.clone(),
        y: avg_pool4_output.clone(),
        kernel_size: 2,
        remainder: avg4_remainder.clone(),
    };
    avg_pool4_circuit.generate_constraints(cs.clone())?;

    //----------------------------------------------------------------------
    let padded_conv43_output = padding_helper(avg_pool4_output.clone(), 1);

    let mut conv51_output = vec![vec![vec![vec![0u8; padded_conv43_output[0][0][0].len() - self.conv51_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv43_output[0][0].len() - self.conv51_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv51_weights.len()]; //number of conv kernels
    padded_conv43_output.len()]; //input (image) batch size

    let (remainder_conv51, div_conv51) = vec_conv_with_remainder_u8(
    &padded_conv43_output.clone(),
    &self.conv51_weights,
    &mut conv51_output,
    self.conv43_output_0,
    self.conv51_weights_0,
    self.conv51_output_0,
    &self.multiplier_conv51,
    );

    //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
    let conv51_circuit = ConvCircuitOp3Wrapper {
    x: padded_conv43_output.clone(),
    conv_kernel: self.conv51_weights.clone(),
    y: conv51_output.clone(),
    remainder: remainder_conv51.clone(),
    div: div_conv51.clone(),

    x_0: self.conv43_output_0,
    conv_kernel_0: self.conv51_weights_0,
    y_0: self.conv51_output_0,

    multiplier: self.multiplier_conv51,
    padding: 0,
    };
    conv51_circuit.generate_constraints(cs.clone())?;


    //----------------------------------------------------------------------
    let padded_conv51_output = padding_helper(conv51_output.clone(), 1);

    let mut conv52_output = vec![vec![vec![vec![0u8; padded_conv51_output[0][0][0].len() - self.conv52_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv51_output[0][0].len() - self.conv52_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    self.conv52_weights.len()]; //number of conv kernels
    padded_conv51_output.len()]; //input (image) batch size

    let (remainder_conv52, div_conv52) = vec_conv_with_remainder_u8(
    &padded_conv51_output.clone(),
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

    multiplier: self.multiplier_conv52,
    padding: 0,
    };
    conv52_circuit.generate_constraints(cs.clone())?;

    //----------------------------------------------------------------------
    let padded_conv52_output = padding_helper(conv52_output.clone(), 0);

    let mut conv53_output = vec![vec![vec![vec![0u8; padded_conv52_output[0][0][0].len()];  // w - kernel_size + 1
    padded_conv52_output[0][0].len()]; // h - kernel_size+ 1
    self.conv53_weights.len()]; //number of conv kernels
    padded_conv52_output.len()]; //input (image) batch size

    let (remainder_conv53, div_conv53) = vec_conv_with_remainder_u8(
    &padded_conv52_output.clone(),
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
    padding: 0,

    };
    conv53_circuit.generate_constraints(cs.clone())?;

    // #[cfg(debug_assertion)]
    println!(
        "Number of constraints for Conv42 {} accumulated constraints {}",
        cs.num_constraints() - _cir_number,
        cs.num_constraints()
    );
    _cir_number = cs.num_constraints();


    //avg pool5 layer
    let (avg_pool5_output, avg5_remainder) = avg_pool_with_remainder_scala_u8(&conv53_output, 2);
    let avg_pool5_circuit = AvgPoolWrapper {
        x: conv53_output.clone(),
        y: avg_pool5_output.clone(),
        kernel_size: 2,
        remainder: avg5_remainder.clone(),
    };
    avg_pool5_circuit.generate_constraints(cs.clone())?;

    println!("avg");
    print_size(avg_pool5_output.clone());
//---------------------------------------------------------------------

        let mut transformed_conv53_output =
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
                        transformed_conv53_output[i][counter] = avg_pool5_output[i][j][p][q];
                        counter += 1;
                    }
                }
            }
        }

        //layer 4 :
        //FC1 -> relu
        let mut fc1_output = vec![vec![0u8; self.fc1_weights.len()];  // channels
        transformed_conv53_output.len()]; //batch size
        let fc1_weight_ref: Vec<&[u8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

        for i in 0..transformed_conv53_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            let (remainder_fc1, div_fc1) = vec_mat_mul_with_remainder_u8(
                &transformed_conv53_output[i],
                fc1_weight_ref[..].as_ref(),
                &mut fc1_output[i],
                self.conv53_output_0,
                self.fc1_weights_0,
                self.fc1_output_0,
                &self.multiplier_fc1.clone(),
            );

            //because the vector dot product is too short. SIMD can not reduce the number of contsraints
            let fc1_circuit = FCWrapper {
                x: transformed_conv53_output[i].clone(),
                weights: self.fc1_weights.clone(),
                y: fc1_output[i].clone(),
                remainder: remainder_fc1.clone(),
                div: div_fc1.clone(),
                x_0: self.conv53_output_0,
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


        //layer 5 :
        //FC2 -> output
        let mut fc2_output = vec![vec![0u8; self.fc2_weights.len()]; // channels
                                            fc1_output.len()]; //batch size
        let fc2_weight_ref: Vec<&[u8]> = self.fc2_weights.iter().map(|x| x.as_ref()).collect();

        for i in 0..fc2_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            let (remainder_fc2, div_fc2) = vec_mat_mul_with_remainder_u8(
                &fc1_output[i],
                fc2_weight_ref[..].as_ref(),
                &mut fc2_output[i],
                self.fc1_output_0,
                self.fc2_weights_0,
                self.fc2_output_0,
                &self.multiplier_fc2.clone(),
            );

            //because the vector dot product is too short. SIMD can not reduce the number of contsraints
            let fc2_circuit = FCWrapper {
                x: fc1_output[i].clone(),
                weights: self.fc2_weights.clone(),
                y: fc2_output[i].clone(),

                x_0: self.fc1_output_0,
                weights_0: self.fc2_weights_0,
                y_0: self.fc2_output_0,

                multiplier: self.multiplier_fc2.clone(),

                remainder: remainder_fc2.clone(),
                div: div_fc2.clone(),
            };
            fc2_circuit.generate_constraints(cs.clone())?;

            println!(
                "Number of constraints FC2 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }



        let mut fc3_output = vec![vec![0u8; self.fc3_weights.len()]; // channels
                                            fc2_output.len()]; //batch size
        let fc3_weight_ref: Vec<&[u8]> = self.fc3_weights.iter().map(|x| x.as_ref()).collect();

        for i in 0..fc3_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            let (remainder_fc3, div_fc3) = vec_mat_mul_with_remainder_u8(
                &fc2_output[i],
                fc3_weight_ref[..].as_ref(),
                &mut fc3_output[i],
                self.fc2_output_0,
                self.fc3_weights_0,
                self.fc3_output_0,
                &self.multiplier_fc3.clone(),
            );

            //because the vector dot product is too short. SIMD can not reduce the number of contsraints
            let fc3_circuit = FCWrapper {
                // x: fc2_output[i].clone(),
                // l1_mat: self.fc3_weights.clone(),
                // y: fc3_output[i].clone(),

                // x_0: self.fc2_output_0,
                // l1_mat_0: self.fc3_weights_0,
                // y_0: self.fc3_output_0,

                // multiplier: self.multiplier_fc3.clone(),

                // remainder: remainder_fc3.clone(),
                // div: div_fc3.clone(),
                // knit_encoding: self.knit_encoding,
                x: fc2_output[i].clone(), 
                weights: self.fc3_weights.clone(), 
                y: fc3_output[i].clone(), 
                x_0: self.fc2_output_0, 
                weights_0: self.fc3_weights_0, 
                y_0: self.fc3_output_0, 
                multiplier: self.multiplier_fc3.clone(), 
                remainder: remainder_fc3.clone(), 
                div: div_fc3.clone(), 
            };
            fc3_circuit.generate_constraints(cs.clone())?;

            println!(
                "Number of constraints FC3 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }

        Ok(())
    }
}

