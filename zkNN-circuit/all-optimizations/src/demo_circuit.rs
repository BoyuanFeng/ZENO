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


#[derive(Debug, Clone)]
pub struct DemoCircuit {
    pub x: u8
}

impl ConstraintSynthesizer<Fq> for DemoCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("DemoCircuit is setup mode: {}", cs.is_in_setup_mode());

        let fq_x:Fq = self.x.into();
        let witness_x = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "x"), || Ok(fq_x)).unwrap(); 
        let result = witness_x.clone() * witness_x.clone() + witness_x.clone();
        // let var_result = cs.new_witness_variable(|| Ok(fq_x))?;

        let truth:u8 = 2;
        let fq_truth:Fq = truth.into();
        let witness_truth = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "x"), || Ok(fq_truth)).unwrap();

        // check result * one = truth
        cs.enforce_constraint(
            lc!() + result.variable(),
            lc!() + Variable::One,
            lc!() + witness_truth.variable(),
        )?;
        Ok(())
    }
}

/* 
impl ConstraintSynthesizer<Fq> for DemoCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!("DemoCircuit is setup mode: {}", cs.is_in_setup_mode());

        let fq_x:Fq = self.x.into();
        let fqvar_x = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "x"), || Ok(fq_x)).unwrap(); // or witness
        // let fq_const_5: Fq = 5u8.into();
        // let fqvar_const_5 = FpVar::<Fq>::Constant(fq_const_5);

        // let result = fqvar_x.clone() * fqvar_x.clone() * fqvar_x.clone() + fqvar_x.clone() + fqvar_const_5;
        let result = fqvar_x.clone() * fqvar_x.clone() + fqvar_x.clone();


        let truth:u8 = 2;
        let fq_truth:Fq = truth.into();
        let fqvar_truth = FpVar::<Fq>::new_witness(r1cs_core::ns!(cs, "truth"), || Ok(fq_truth)).unwrap(); // or witness
        
        // result.enforce_equal(&fqvar_truth);
        // result.is_eq(&fqvar_truth);
        let fq_one:Fq = 1u32.into();
        let fqvar_one = FpVar::<Fq>::Constant(fq_one);
        result.mul_equals(&fqvar_one,&fqvar_truth);

        cs.enforce_constraint();
        Ok(())
    }
}
*/
