use crate::argmax_circuit::*;
use crate::avg_pool_circuit::*;
use crate::conv_circuit::*;
use crate::cosine_circuit::*;
use crate::mul_circuit::*;
use crate::relu_circuit::*;
use crate::vanilla::*;
use crate::*;

use algebra::ed_on_bls12_381::*;
use algebra_core::Zero;
use pedersen_commit::*;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::fields::fp::FpVar;

pub(crate) fn padding_helper(input: Vec<Vec<Vec<Vec<u8>>>>, padding: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
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


pub(crate) fn generate_fqvar4d(
    cs: ConstraintSystemRef<Fq>,
    input: Vec<Vec<Vec<Vec<u8>>>>,
) -> Vec<Vec<Vec<Vec<FqVar>>>> {
    let mut res: Vec<Vec<Vec<Vec<FqVar>>>> =
        vec![
            vec![
                vec![
                    vec![FpVar::<Fq>::Constant(Fq::zero()); input[0][0][0].len()];
                    input[0][0].len()
                ];
                input[0].len()
            ];
            input.len()
        ];
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            for k in 0..input[i][j].len() {
                for l in 0..input[i][j][k].len() {
                    let fq: Fq = input[i][j][k][l].into();
                    let tmp =
                        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "tmp"), || Ok(fq)).unwrap();
                    res[i][j][k][l] = tmp;
                }
            }
        }
    }
    res
}

fn generate_fqvar_witness2D(cs: ConstraintSystemRef<Fq>, input: Vec<Vec<u8>>) -> Vec<Vec<FqVar>> {
    let zero_var = FpVar::<Fq>::Constant(Fq::zero());
    let mut res: Vec<Vec<FqVar>> = vec![vec![zero_var; input[0].len()]; input.len()];
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            let fq: Fq = input[i][j].into();
            let tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "tmp"), || Ok(fq)).unwrap();
            res[i][j] = tmp;
        }
    }
    res
}


pub(crate) fn generate_fqvar(cs: ConstraintSystemRef<Fq>, input: Vec<u8>) -> Vec<FqVar> {
    let mut res: Vec<FqVar> = Vec::new();
    for i in 0..input.len() {
        let fq: Fq = input[i].into();
        let tmp = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "tmp"), || Ok(fq)).unwrap();
        res.push(tmp);
    }
    res
}


pub(crate) fn convert_4d_vector_into_1d(vec: Vec<Vec<Vec<Vec<u8>>>>) -> Vec<u8> {
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

pub fn convert_2d_vector_into_1d(vec: Vec<Vec<u8>>) -> Vec<u8> {
    let mut res = Vec::new();
    for i in 0..vec.len() {
        res.extend(&vec[i]);
    }
    res
}


#[derive(Debug, Clone)]
pub struct ConvCircuitOp3Wrapper{
    pub x: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    pub conv_kernel: Vec<Vec<Vec<Vec<u8>>>>, //[Num Kernel, Num Channel, kernel_size, kernel_size]
    pub y: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Kernel, Height - kernel_size + 1, Width - kernel_size + 1]

    pub remainder: Vec<Vec<Vec<Vec<u32>>>>,
    pub div: Vec<Vec<Vec<Vec<u32>>>>,

    //zero points for quantization
    pub x_0: u8,
    pub conv_kernel_0: u8,
    pub y_0: u8,

    //multiplier for quantization. s1*s2/s3
    pub multiplier: Vec<f32>,

    // Padding size
    pub padding: usize,
}

impl ConstraintSynthesizer<Fq> for ConvCircuitOp3Wrapper{
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        // let padded_x = padding_helper(self.x, self.padding);

        // let (remainder, div) = vec_conv_with_remainder_u8(
        //     &self.x,
        //     &self.conv_kernel,
        //     &mut self.y.clone(),
        //     self.x_0,
        //     self.conv_kernel_0,
        //     self.y_0,
        //     &self.multiplier,
        // );

        let x_fqvar = generate_fqvar4d(cs.clone(), self.x.clone());
        let y_fqvar = generate_fqvar4d(cs.clone(), self.y.clone());
        let weight_fqvar_input =
        generate_fqvar4d(cs.clone(), self.conv_kernel.clone());
        // y_0 and multiplier are both constants.
        let mut y_zeropoint_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier.len() {
            let m = (self.multiplier[i] * (2u64.pow(M_EXP)) as f32) as u64;
            y_zeropoint_converted
                .push((self.y_0 as u64 * 2u64.pow(M_EXP)) / m);
        }

        //use SIMD for reducing constraints
        let circuit = ConvCircuitOp3 {
            x: x_fqvar.clone(),
            conv_kernel: weight_fqvar_input.clone(),
            y: y_fqvar.clone(),
            remainder: self.remainder.clone(),
            div: self.div.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv_kernel_0,
            y_0: y_zeropoint_converted,

            multiplier: self.multiplier,
        };
        circuit.generate_constraints(cs.clone())?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ReLUWrapper{
    pub x: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    pub cmp_res: Vec<Vec<Vec<Vec<bool>>>>,

    //zero points for quantization
    pub y_0: u8,
}

impl ConstraintSynthesizer<Fq> for ReLUWrapper{
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        // let cmp_res = relu4d_u8(&mut self.x, self.x_0);
        let y_fqvar = generate_fqvar4d(cs.clone(), self.x.clone());

        let conv1_output_fqvar = generate_fqvar4d(cs.clone(), self.x.clone());
        let output_fqvar = generate_fqvar4d(cs.clone(), self.x.clone());

        let cmp_res_3d: Vec<Vec<Vec<bool>>> = self.cmp_res.into_iter().flatten().collect();
        let cmp_res_2d: Vec<Vec<bool>> = cmp_res_3d.into_iter().flatten().collect();
        let cmp_res_1d: Vec<bool> = cmp_res_2d.into_iter().flatten().collect();

        let flattened_input3d: Vec<Vec<Vec<FqVar>>> =
            conv1_output_fqvar.into_iter().flatten().collect();
        let flattened_input2d: Vec<Vec<FqVar>> =
            flattened_input3d.into_iter().flatten().collect();
        let flattened_input1d: Vec<FqVar> =
            flattened_input2d.into_iter().flatten().collect();

        let flattened_output3d: Vec<Vec<Vec<FqVar>>> =
            output_fqvar.clone().into_iter().flatten().collect();
        let flattened_output2d: Vec<Vec<FqVar>> =
            flattened_output3d.into_iter().flatten().collect();
        let flattened_output1d: Vec<FqVar> =
            flattened_output2d.into_iter().flatten().collect();

        let circuit = ReLUCircuitOp3 {
            y_in: flattened_input1d.clone(),
            y_out: flattened_output1d.clone(),
            y_zeropoint: self.y_0,
            cmp_res: cmp_res_1d.clone(),
        };
        circuit.generate_constraints(cs.clone())?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AvgPoolWrapper {
    pub x: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    pub y: Vec<Vec<Vec<Vec<u8>>>>,
    pub kernel_size: usize,
    pub remainder: Vec<Vec<Vec<Vec<u8>>>>,
}

impl ConstraintSynthesizer<Fq> for AvgPoolWrapper {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {

        let output_fqvar = generate_fqvar4d(cs.clone(), self.y.clone());
        let x_fqvar = generate_fqvar4d(cs.clone(), self.x.clone());
        let circuit = AvgPoolCircuitLv3 {
            x: x_fqvar.clone(),
            y: output_fqvar.clone(),
            kernel_size: self.kernel_size,
            remainder: self.remainder.clone(),
        };
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FCWrapper{
    pub x: Vec<u8>, // [Batch Size, CIN]
    pub weights: Vec<Vec<u8>>, //[CIN, COUT]
    pub y: Vec<u8>, // [Batch Size, COUT]

    pub remainder: Vec<u32>,
    pub div: Vec<u32>,
    //zero points for quantization
    pub x_0: u8,
    pub weights_0: u8,
    pub y_0: u8,

    //multiplier for quantization. s1*s2/s3
    pub multiplier: Vec<f32>,
}

impl ConstraintSynthesizer<Fq> for FCWrapper{
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        // let mut output = vec![vec![0u8; self.weights.len()];  // channels
        //                                     self.x.len()]; //batch size
        // let weight_ref: Vec<&[u8]> = self.weights.iter().map(|x| x.as_ref()).collect();

        // //assume we only do inference on one image
        // let (remainder, div) = vec_mat_mul_with_remainder_u8(
        //     &self.x[0],
        //     weight_ref[..].as_ref(),
        //     &mut output[0],
        //     self.x_0,
        //     self.weights_0,
        //     self.y_0,
        //     &self.multiplier,
        // );

        let output_fqvar = generate_fqvar(cs.clone(), self.y.clone());
        let mut output0_converted: Vec<u64> = Vec::new();
        for i in 0..self.multiplier.len() {
            let m = (self.multiplier[i] * (2u64.pow(M_EXP)) as f32) as u64;
            output0_converted.push((self.y_0 as u64 * 2u64.pow(M_EXP)) / m);
        }
        let weights_fqvar_input =
            generate_fqvar_witness2D(cs.clone(), self.weights.clone());

        
        let x_fqvar = generate_fqvar(cs.clone(), self.x.clone());
        let circuit = FCCircuitOp3 {
            x: x_fqvar.clone(),
            l1_mat: weights_fqvar_input.clone(),
            y: output_fqvar.clone(),
            remainder: self.remainder.clone(),
            div: self.div.clone(),
            x_0: self.x_0,
            l1_mat_0: self.weights_0,
            y_0: output0_converted,

            multiplier: self.multiplier.clone(),
        };

        circuit.generate_constraints(cs.clone())?;

        Ok(())
    }
}
