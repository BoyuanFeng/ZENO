use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::biginteger::*;
use algebra_core::Zero;
use num_traits::Pow;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::eq::EqGadget;
use r1cs_std::fields::fp::FpVar;
use r1cs_std::ToBitsGadget;
use std::cmp;

use std::ops::*;
//in knit encoding, we assume that the kernel weights are all public and constant. the prover does not need to commit the weights.

#[derive(Debug, Clone)]
pub struct ConvCircuitU8BitDecomposeOptimization {
    pub x: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    pub conv_kernel: Vec<Vec<Vec<Vec<u8>>>>, //[Num Kernel, Num Channel, kernel_size, kernel_size]
    pub y: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Kernel, Height - kernel_size + 1, Width - kernel_size + 1]

    //these two variables are used to restore the real y
    pub remainder: Vec<Vec<Vec<Vec<u32>>>>,
    pub div: Vec<Vec<Vec<Vec<u32>>>>,

    //zero points for quantization
    pub x_0: u8,
    pub conv_kernel_0: u8,
    pub y_0: u8,

    //multiplier for quantization. s1*s2/s3
    pub multiplier: Vec<f32>,
    pub knit_encoding: bool,
}

impl ConstraintSynthesizer<Fq> for ConvCircuitU8BitDecomposeOptimization {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let batch_size = self.x.len();
        let input_height = self.x[0][0].len();
        let input_width = self.x[0][0][0].len();

        let num_kernel = self.conv_kernel.len();
        let kernel_size = self.conv_kernel[0][0].len();
        let encoding_batch_size = 8; // this batch size should be related to the vector dot length. 254/()

        for k in 0..num_kernel {
            for n in 0..batch_size {
                for h in 0..(input_height - kernel_size + 1) {

                    for p in (0..(input_width - kernel_size + 1)).step_by(encoding_batch_size){
                        let mut left_side: Vec<FqVar> = Vec::new();
                        let mut right_side: Vec<FqVar> = Vec::new();
                        for w in p..(cmp::min(input_width - kernel_size + 1, p + encoding_batch_size)) {
                            let m = (self.multiplier[k] * (2.pow(M_EXP)) as f32) as u64;
    
                            let y_0_converted: u64 = (self.y_0 as u64 * 2u64.pow(22)) / m;
    
                            let multiplier_fq: Fq = m.into();
                            let multiplier_var = FpVar::Constant(multiplier_fq);
    
                            let tmp = multiplier_var
                                * conv_kernel_remainder_helper_u8(
                                    cs.clone(),
                                    self.x[n].clone(),
                                    self.conv_kernel[k].clone(),
                                    h,
                                    w,
                                    self.x_0,
                                    self.conv_kernel_0,
                                    y_0_converted,
                                );
                            //np.sum(self.x[n, :, h : h + kernel_size, w: w + kernel_size] * self.conv_kernel[k])
    
                            let yy: Fq = (self.y[n][k][h][w] as u64).into();
                            let yy_var =
                                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "yy gadget"), || Ok(yy))
                                    .unwrap();
                            let div: Fq = (self.div[n][k][h][w] as u64).into();
                            let div_var =
                                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div gadget"), || Ok(div))
                                    .unwrap();
                            let remainder: Fq = (self.remainder[n][k][h][w] as u64).into();
                            let remainder_var = FpVar::<Fq>::new_witness(
                                r1cs_core::ns!(cs, "remainder gadget"),
                                || Ok(remainder),
                            )
                            .unwrap();
                            let two_power_8: Fq = (2u64.pow(8)).into();
                            let two_power_8_constant = FpVar::<Fq>::Constant(two_power_8);
                            let two_power_22: Fq = (2u64.pow(22)).into();
                            let two_power_22_constant = FpVar::<Fq>::Constant(two_power_22);
    
                            let output_var = (yy_var + div_var * two_power_8_constant)
                                * two_power_22_constant
                                + remainder_var;
                            
                                left_side.push(tmp.clone());
                                right_side.push(output_var.clone()); 
                        }
                        if (self.knit_encoding) {
                            knit_encoding_check(cs.clone(), &left_side, &right_side);
                        } else {
                            for i in 0..left_side.len() {
                                left_side[i].enforce_equal(&right_side[i]).unwrap();
                            }
                        }
                    }
                    
                }
            }
        }

        Ok(())
    }
}

fn mul_cs_helper_u8(cs: ConstraintSystemRef<Fq>, a: u8, c: u8) -> FqVar {
    let aa: Fq = a.into();
    let cc: Fq = c.into();
    let a_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
    let c_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "c gadget"), || Ok(cc)).unwrap();
    a_var.mul(c_var)
}

fn constant_mul_cs_helper_u8(cs: ConstraintSystemRef<Fq>, a: u8, constant: u8) -> FqVar {
    let aa: Fq = a.into();
    let cc: Fq = constant.into();
    let a_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
    let c_var = FpVar::Constant(cc);
    a_var.mul(c_var)
}

fn constant_mul_constant_cs_helper_u8(_cs: ConstraintSystemRef<Fq>, c1: u8, c2: u8) -> FqVar {
    let aa: Fq = c1.into();
    let cc: Fq = c2.into();
    let a_var = FpVar::Constant(aa);
    let c_var = FpVar::Constant(cc);
    a_var.mul(c_var)
}

fn knit_encoding_check(cs: ConstraintSystemRef<Fq>, left_side: &[FqVar], right_side: &[FqVar]) {
    //encode here.
    let two_power_32 = 2u32.pow(32);
    let two_power_32_fq: Fq = two_power_32.into();
    let bit_shift = FpVar::Constant(two_power_32_fq);
    if (left_side.len() != right_side.len() || left_side.len() > 8) {
        // we only support at most 8 var encoded check.
        println!("error length in knit encoding! conv");
    }
    let mut left_encoded =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "left gadget"), || Ok(Fq::zero())).unwrap();
    let mut right_encoded =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "right gadget"), || Ok(Fq::zero())).unwrap();
    left_encoded += left_side[0].clone();
    right_encoded += right_side[0].clone();

    for i in 1..left_side.len() {
        left_encoded += bit_shift.clone() * left_side[i].clone();
        right_encoded += bit_shift.clone() * right_side[i].clone();
    }

    left_encoded.enforce_equal(&right_encoded).unwrap();
}

fn conv_kernel_remainder_helper_u8(
    cs: ConstraintSystemRef<Fq>,
    x: Vec<Vec<Vec<u8>>>,
    kernel: Vec<Vec<Vec<u8>>>,
    h_index: usize,
    w_index: usize,

    x_zeropoint: u8,
    kernel_zeropoint: u8,

    y_0_converted: u64,
) -> FqVar {
    let _no_cs = cs.num_constraints();

    let num_channels = kernel.len();
    let kernel_size = kernel[0].len();
    let mut tmp1 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    //println!("Conv channel * kernel_size * kernel_size {}", C * kernel_size * kernel_size);

    let y_zeropoint_fq: Fq = y_0_converted.into();
    let y_zeropoint_var = FpVar::<Fq>::Constant(y_zeropoint_fq);
    // println!("multiplier : {}", (multiplier * (2u64.pow(22)) as f32) as u32);
    // println!("y_0 {}, y_converted : {}", y_0, (y_0 as u64 * 2u64.pow(22)));
    for i in 0..num_channels {
        //iterate through all channels
        for j in h_index..(h_index + kernel_size) {
            for k in w_index..(w_index + kernel_size) {
                //x_zeropoint, kernel_zeropoints and y_zeropoints are all Constant wires because they are independent of input image
                tmp1 += constant_mul_cs_helper_u8(cs.clone(), x[i][j][k], kernel[i][j - h_index][k - w_index]);
                tmp2 += constant_mul_cs_helper_u8(cs.clone(), x[i][j][k], kernel_zeropoint);
                tmp3 += constant_mul_constant_cs_helper_u8(
                    cs.clone(),
                    kernel[i][j - h_index][k - w_index],
                    x_zeropoint,
                );
                tmp4 +=
                    constant_mul_constant_cs_helper_u8(cs.clone(), x_zeropoint, kernel_zeropoint);
            }
        }
    }

    let res = (tmp1 + tmp4 + y_zeropoint_var) - (tmp2 + tmp3);

    res
}
