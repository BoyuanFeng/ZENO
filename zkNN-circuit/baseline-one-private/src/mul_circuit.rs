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
        println!("error length in knit encoding! fc");
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

fn scala_cs_helper_remainder_u8(
    cs: ConstraintSystemRef<Fq>,
    input: &[u8],
    weight: &[u8],
    input_zeropoint: u8,
    weight_zeropoint: u8,
    y_zeropoint_converted: u64,
) -> FqVar {
    let _no_cs = cs.num_constraints();
    if input.len() != weight.len() {
        panic!("scala mul: length not equal");
    }
    //println!("a {:?} \n b {:?}", a, b);
    let mut tmp1 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*q2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp2 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q1*z2 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp3 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "q2*z1 gadget"), || Ok(Fq::zero())).unwrap();
    let mut tmp4 =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "z1*z2 gadget"), || Ok(Fq::zero())).unwrap();

    //zero points of input, weight and y for quantization are all fixed after training, so they are Constant wires.
    let y_zeropoint_fq: Fq = y_zeropoint_converted.into();
    let y_zeropoint_var = FpVar::<Fq>::Constant(y_zeropoint_fq);

    //println!("multiplier : {}", (multiplier * (2u64.pow(22)) as f32) as u32);
    //println!("y_0 {}, y_converted : {}", y_zeropoint, (y_zeropoint as u64 * 2u64.pow(22)));

    for i in 0..input.len() {
        tmp1 += constant_mul_cs_helper_u8(cs.clone(), input[i], weight[i]);
        tmp2 += constant_mul_cs_helper_u8(cs.clone(), input[i], weight_zeropoint);
        tmp3 += constant_mul_constant_cs_helper_u8(cs.clone(), weight[i], input_zeropoint);
        tmp4 += constant_mul_constant_cs_helper_u8(cs.clone(), input_zeropoint, weight_zeropoint);
    }
    //println!("tmp1 {:?} \n tmp2 {:?} \n tmp3 {:?} \n tmp4 {:?}", tmp1.value().unwrap(), tmp2.value().unwrap(), tmp3.value().unwrap(), tmp4.value().unwrap());
    let res = (tmp1 + tmp4 + y_zeropoint_var) - (tmp2 + tmp3);
    #[cfg(debug_assertion)]
    println!(
        "number of constrants for scalar {}",
        cs.num_constraints() - _no_cs
    );

    res
}

#[derive(Debug, Clone)]
pub struct FCCircuitU8BitDecomposeOptimized {
    pub x: Vec<u8>,
    pub l1_mat: Vec<Vec<u8>>, // we assume that the kernel weights are public constants.
    pub y: Vec<u8>,

    //these two variables are used to restore the real y
    pub remainder: Vec<u32>,
    pub div: Vec<u32>,

    // we need enforce quality between:
    // (y - y_0) as u32 * div * 2^24 as u32 + remainder = [\sum(x - x_0)(l1_mat - l1_mat_0)] as u32 * (multiplier * 2^24 as f32) as u32

    //zero points for quantization
    pub x_0: u8,
    pub l1_mat_0: u8,
    pub y_0: u8,

    //multiplier for quantization. s1*s2/s3
    pub multiplier: Vec<f32>,
    pub knit_encoding: bool, //use this to trigger whether we enable knit encoding
}

impl ConstraintSynthesizer<Fq> for FCCircuitU8BitDecomposeOptimized {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("FCCircuitU8 is setup mode: {}", cs.is_in_setup_mode());

        let encoding_batch_size = 8; 
        for k in (0..self.y.len()).step_by(encoding_batch_size) {
            let mut left_side: Vec<FqVar> = Vec::new();
            let mut right_side: Vec<FqVar> = Vec::new();
            for i in k..(cmp::min(k + encoding_batch_size, self.y.len())) {
                // compute multiplier * <x, l1[i]>(dot product), store the result in tmp

                let m = (self.multiplier[i] * (2.pow(M_EXP)) as f32) as u64;

                let y_0_converted: u64 = (self.y_0 as u64 * 2u64.pow(M_EXP)) / m;
                //println!("y_0_converted {}", y_0_converted);

                //multipliers for quantization is fixed parameters after training the model.
                let multiplier_fq: Fq = m.into();
                let multiplier_var = FpVar::<Fq>::Constant(multiplier_fq);

                let tmp = multiplier_var
                    * scala_cs_helper_remainder_u8(
                        cs.clone(),
                        &self.x,
                        &self.l1_mat[i],
                        self.x_0,
                        self.l1_mat_0,
                        y_0_converted,
                    );
                // let multiplier: Fq = ((self.multiplier[i] * (2.pow(M_EXP)) as f32) as u32).into();
                // let multiplier_var =
                //     FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "multiplier gadget"), || Ok(multiplier))
                //         .unwrap();
                // let tmp = tmp * multiplier_var.clone();
                let yy: Fq = (self.y[i] as u64).into();
                let yy_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "yy gadget"), || Ok(yy)).unwrap();
                let div: Fq = (self.div[i] as u64).into();
                let div_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "div gadget"), || Ok(div)).unwrap();
                let remainder: Fq = (self.remainder[i] as u64).into();
                let remainder_var =
                    FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder gadget"), || {
                        Ok(remainder)
                    })
                    .unwrap();
                let two_power_8: Fq = (2u64.pow(8)).into();
                let two_power_8_constant = FpVar::<Fq>::Constant(two_power_8);
                let two_power_22: Fq = (2u64.pow(22)).into();
                let two_power_22_constant = FpVar::<Fq>::Constant(two_power_22);

                let output_var = (yy_var + div_var * two_power_8_constant) * two_power_22_constant
                    + remainder_var;

                left_side.push(tmp.clone());
                right_side.push(output_var.clone());

                //println!("left {} == right {}", u32_res ,(self.y[i] - self.y_0) as u32 * 2u32.pow(M_EXP) + self.remainder[i]);
                //println!("{} {}", (self.y[i] - self.y_0) as u32, self.remainder[i]);
                //assert_eq!(u32_res ,(self.y[i] - self.y_0) as u32 * 2u32.pow(M_EXP) + self.remainder[i]);
                //println!("left {:?}\nright{:?}\n\n\n\n", tmp.to_bits_le().unwrap().value().unwrap(), yy_var.to_bits_le().unwrap().value().unwrap());
                //assert_eq!(tmp.to_bits_le().unwrap().value().unwrap(), yy_var.to_bits_le().unwrap().value().unwrap());
            }

            if (self.knit_encoding) {
                knit_encoding_check(cs.clone(), &left_side, &right_side);
            } else {
                for i in 0..left_side.len() {
                    left_side[i].enforce_equal(&right_side[i]).unwrap();
                }
            }
        }

        Ok(())
    }
}
