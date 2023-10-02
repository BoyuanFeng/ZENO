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
pub struct ResidualAddCircuit {
    pub input1: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    pub input2: Vec<Vec<Vec<Vec<u8>>>>, //[Num Kernel, Num Channel, kernel_size, kernel_size]
    pub output: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Kernel, Height - kernel_size + 1, Width - kernel_size + 1]

    //these two variables are used to restore the real y
    pub remainder: Vec<Vec<Vec<Vec<u32>>>>,
    pub div: Vec<Vec<Vec<Vec<u32>>>>,

    //zero points for quantization
    pub input1_0: u8,
    pub input2_0: u8,
    pub output_0: u8,

    //multiplier for quantization.
    pub multiplier_1: Vec<f32>,
    pub multiplier_2: Vec<f32>,
    pub knit_encoding: bool,
}

impl ConstraintSynthesizer<Fq> for ResidualAddCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let batch_size = self.input1.len();
        let channel_num = self.input1[0].len();
        let input_height = self.input1[0][0].len();
        let input_width = self.input1[0][0][0].len();
    
        let encoding_batch_size = 8; // this batch size should be related to the vector dot length. 254/()
        for n in 0..batch_size {
            for k in 0..channel_num {
                for h in 0..input_height {
                    for p in (0..input_width)
                        .step_by(encoding_batch_size)
                    {
                        let mut left_side: Vec<FqVar> = Vec::new();
                        let mut right_side: Vec<FqVar> = Vec::new();
                        for w in
                            p..(cmp::min(input_width, p + encoding_batch_size))
                        {
                            let m1 = (self.multiplier_1[k] * 2u64.pow(22) as f32) as u64;
                            let m2 = (self.multiplier_2[k] * 2u64.pow(22) as f32) as u64;

                            let multiplier_fq1: Fq = m1.into();
                            let multiplier_fq2: Fq = m2.into();

                            let multiplier_var1 = FpVar::Constant(multiplier_fq1);
                            let multiplier_var2 = FpVar::Constant(multiplier_fq2);

                            let output_scaled_zero: u64 = (self.output_0 as u64 * 2u64.pow(22));
                            let output_scaled_zero_fq: Fq = output_scaled_zero.into();
                            let output_scaled_zero_var = FpVar::<Fq>::Constant(output_scaled_zero_fq);

                            let mut tmp1 = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "m1*input2 gadget"), || Ok(Fq::zero())).unwrap();
                            let mut tmp2 = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "m2*input1 gadget"), || Ok(Fq::zero())).unwrap();
                            let mut tmp3 = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "m1*input2_0 gadget"), || Ok(Fq::zero())).unwrap();
                            let mut tmp4 = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "m2*input1_0 gadget"), || Ok(Fq::zero())).unwrap();
                            
                            tmp1 += constant_mul_cs_helper_u64(cs.clone(), self.input2[n][k][h][w], m1);
                            tmp2 += constant_mul_cs_helper_u64(cs.clone(), self.input1[n][k][h][w], m2);
                            tmp3 += constant_mul_constant_cs_helper_u64(cs.clone(), self.input2_0, m1);
                            tmp4 += constant_mul_constant_cs_helper_u64(cs.clone(), self.input1_0, m2);
                        
                            let tmp = (tmp1+tmp4+output_scaled_zero_var) - (tmp2+tmp3);
                            
                            let yy: Fq = (self.output[n][k][h][w] as u64).into();
                            let yy_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "yy gadget"), || Ok(yy)).unwrap();
                            let div: Fq = (self.div[n][k][h][w] as u64).into();
                            let div_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div gadget"), || Ok(div)).unwrap();
                            let remainder: Fq = (self.remainder[n][k][h][w] as u64).into();
                            let remainder_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder gadget"), || Ok(remainder)).unwrap();

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

fn constant_mul_cs_helper_u64(cs: ConstraintSystemRef<Fq>, a: u8, constant: u64) -> FqVar {
    let aa: Fq = a.into();
    let cc: Fq = constant.into();
    let a_var = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
    let c_var = FpVar::Constant(cc);
    a_var.mul(c_var)
}

fn constant_mul_constant_cs_helper_u64(_cs: ConstraintSystemRef<Fq>, c1: u8, c2: u64) -> FqVar {
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
        println!("error length in knit encoding!");
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
