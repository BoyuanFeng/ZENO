use crate::*;
use algebra::ed_on_bls12_381::*;
use algebra_core::Zero;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::eq::EqGadget;
use r1cs_std::fields::fp::FpVar;
use std::cmp;

#[derive(Debug, Clone)]
pub struct AvgPoolCircuitU8BitDecomposeOptimized {
    pub x: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    pub y: Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height/kernel_size, Width/kernel_size]
    pub kernel_size: usize,
    pub remainder: Vec<Vec<Vec<Vec<u8>>>>,
    pub knit_encoding: bool, //use this to trigger whether we enable knit encoding

    // we do not need the quantization parameters to calculate the avg pool output
}

impl ConstraintSynthesizer<Fq> for AvgPoolCircuitU8BitDecomposeOptimized {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "AvgPoolCircuitU8BitDecomposeOptimized is setup mode: {}",
            cs.is_in_setup_mode()
        );
        let encoding_batch_size = 8; 

        let batch_size = self.x.len();
        let num_channels = self.x[0].len();
        let input_height = self.x[0][0].len();
        let input_width = self.x[0][0][0].len();
        let kernel_size_fq: Fq = (self.kernel_size as u32).into();
        let kernel_size_const = FpVar::Constant(kernel_size_fq);
        for n in 0..batch_size {
            for c in 0..num_channels {
                for h in 0..(input_height / self.kernel_size) {
                    for k in (0..(input_width / self.kernel_size)).step_by(encoding_batch_size) {
                        let mut left_side: Vec<FqVar> = Vec::new();
                        let mut right_side: Vec<FqVar> = Vec::new();
                        for w in k..cmp::min((input_width / self.kernel_size), k+encoding_batch_size) {
                            // self.y[n][c][x][y] = np.mean(self.x[n][c][kernel_size*x:kernel_size*(x+1)][kernel_size*y:kernel_size*(y+1)])

                            let tmp = sum_helper_u8(
                                cs.clone(),
                                self.x[n][c].clone(),
                                self.kernel_size * h,
                                self.kernel_size * w,
                                self.kernel_size,
                            );
                            let yy: Fq = (self.y[n][c][h][w]  as u64).into();
                            let yy_var =
                                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "yy gadget"), || Ok(yy)).unwrap();
                            let remainder: Fq = (self.remainder[n][c][h][w] as u64).into();
                            let remainder_var =
                                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "remainder gadget"), || {
                                    Ok(remainder)
                                })
                                .unwrap();
                            let yy_var = yy_var * kernel_size_const.clone() * kernel_size_const.clone() + remainder_var;

                            left_side.push(tmp.clone());
                            right_side.push(yy_var.clone());
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

fn knit_encoding_check(cs: ConstraintSystemRef<Fq>, left_side: &[FqVar], right_side: &[FqVar]) {
    //encode here.
    let two_power_32 = 2u32.pow(32);
    let two_power_32_fq: Fq = two_power_32.into();
    let bit_shift = FpVar::Constant(two_power_32_fq);
    if (left_side.len() != right_side.len() || left_side.len() > 8) {
        // we only support at most 8 var encoded check.
        println!("error length {}-{} in knit encoding!", left_side.len(), right_side.len());
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

fn sum_helper_u8(
    cs: ConstraintSystemRef<Fq>,
    x: Vec<Vec<u8>>,
    h_index: usize,
    w_index: usize,
    kernel_size: usize,
) -> FqVar {
    //we don't need to multiply the multiplier. we can obtain the mean value directly on u8 type(accumulated using u32 to avoid overflow)
    let mut tmp =
        FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "avg pool gadget"), || Ok(Fq::zero())).unwrap();

    for i in h_index..(h_index + kernel_size) {
        for j in w_index..(w_index + kernel_size) {
            let aa: Fq = x[i][j].into();
            let a_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "a gadget"), || Ok(aa)).unwrap();
            tmp += a_var;
        }
    }

    tmp
}