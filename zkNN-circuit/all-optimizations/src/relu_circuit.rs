use crate::*;
use algebra::ed_on_bls12_381::*;
use r1cs_core::*;
use r1cs_std::alloc::AllocVar;
use r1cs_std::boolean::Boolean;
use r1cs_std::ed_on_bls12_381::FqVar;
use r1cs_std::eq::EqGadget;
use r1cs_std::fields::fp::FpVar;

// statement:
//  if y_in[i] < 0, y_out[i] = 0;
//  else y_out[i] = y_in[i]
#[derive(Debug, Clone)]
pub(crate) struct ReLUCircuitU8 {
    pub(crate) y_in: Vec<u8>,
    pub(crate) y_out: Vec<u8>,
    pub(crate) y_zeropoint: u8,
}

impl ConstraintSynthesizer<Fq> for ReLUCircuitU8 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        let zero: Fq = self.y_zeropoint.into();
        let zero_var = FqVar::Constant(zero);
        //zero point is constant wire in the circuit
        for (i, e) in self.y_in.iter().enumerate() {
            let mut cmp;

            let tmp_zero: Fq = (self.y_zeropoint as u32).into();
            let zero_var = FqVar::Constant(tmp_zero);
            // cast y_out[i] as a gadget
            let tmp: Fq = (self.y_out[i] as u32).into();
            let out_var = FpVar::<Fq>::new_witness(
                r1cs_core::ns!(cs, format!("input {}'s gadget", tmp)),
                || Ok(tmp),
            )
            .unwrap();

            let tmp_in: Fq = (*e as u32).into();
            let in_var =
                FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, format!("input 0 gadget",)), || {
                    Ok(tmp_in)
                })
                .unwrap();
            if (self.y_in[i] > self.y_zeropoint) {
                cmp =
                    Boolean::new_witness(r1cs_core::ns!(cs, format!("ReLU cmp_res {}", i)), || {
                        Ok(true)
                    })
                    .unwrap();
            } else {
                cmp =
                    Boolean::new_witness(r1cs_core::ns!(cs, format!("ReLU cmp_res {}", i)), || {
                        Ok(false)
                    })
                    .unwrap();
            }

            // enforce y_in[i] == y_out[i]
            out_var
                .enforce_equal(&cmp.select(&in_var, &zero_var).unwrap())
                .unwrap();
        }
        Ok(())
    }
}


#[derive(Debug, Clone)]
pub(crate) struct ReLUCircuitOp3 {
    pub(crate) y_in: Vec<FqVar>,
    pub(crate) y_out: Vec<FqVar>,
    pub(crate) y_zeropoint: u8,
    pub(crate) cmp_res: Vec<bool>,
}

impl ConstraintSynthesizer<Fq> for ReLUCircuitOp3 {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        //println!("ReLU zero point {}", self.y_zeropoint);
        let zero: Fq = self.y_zeropoint.into();
        let zero_var = FqVar::Constant(zero);
        //zero point is constant wire in the circuit

        for i in 0..self.y_in.len() {
            let cmp =
                Boolean::new_witness(r1cs_core::ns!(cs, format!("ReLU cmp_res {}", i)), || {
                    Ok(self.cmp_res[i])
                })
                .unwrap();
            self.y_out[i]
                .enforce_equal(&cmp.select(&self.y_in[i], &zero_var).unwrap())
                .unwrap();
        }
        Ok(())
    }
}