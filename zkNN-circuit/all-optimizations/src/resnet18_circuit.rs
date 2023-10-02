use crate::argmax_circuit::*;
use crate::avg_pool_circuit::*;
use crate::conv_circuit::*;
use crate::cosine_circuit::*;
use crate::mul_circuit::*;
use crate::relu_circuit::*;
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
fn padding_helper(input: &Vec<Vec<Vec<Vec<u8>>>>, padding: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
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


fn conv_one_private_wrapper(
    input: &Vec<Vec<Vec<Vec<u8>>>>,  // [Batch Size, Num Channel, Height, Width]
    conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    input_0: u8,
    weight_0: u8,
    output_0: u8,
    multiplier: Vec<f32>,
    padding: usize,
    knit_encoding: bool,
) -> (ConvCircuitU8BitDecomposeOptimization, Vec<Vec<Vec<Vec<u8>>>>) {
    let padded_input = padding_helper(&input, padding);

    let mut output = vec![vec![vec![vec![0u8; padded_input[0][0][0].len() - conv_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_input[0][0].len() - conv_weights[0][0].len()+ 1]; // h - kernel_size+ 1
    conv_weights.len()]; //number of conv kernels
    padded_input.len()]; //input (image) batch size

    let (remainder, div) = vec_conv_with_remainder_u8(
        &padded_input,
        &conv_weights,
        &mut output,
        input_0,
        weight_0,
        output_0,
        &multiplier,
    );

    //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
    let conv_circuit = ConvCircuitU8BitDecomposeOptimization {
        x: padded_input.clone(),
        conv_kernel: conv_weights.clone(),
        y: output.clone(),

        remainder: remainder.clone(),
        div: div.clone(),

        x_0: input_0,
        conv_kernel_0: weight_0,
        y_0: output_0,

        multiplier: multiplier.clone(),
        knit_encoding: knit_encoding,
    };
    (conv_circuit, output)
}


fn residual_add_wrapper(
    input1: &Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    input2: &Vec<Vec<Vec<Vec<u8>>>>,

    input1_0: u8,
    input2_0: u8,
    output_0: u8,
    multiplier_1: Vec<f32>,
    multiplier_2: Vec<f32>,
    knit_encoding: bool,
) -> (ResidualAddCircuit, Vec<Vec<Vec<Vec<u8>>>>) {

    // residual add 1 =================================================================================================================
    let mut residual_output = vec![vec![vec![vec![0u8; input1[0][0][0].len()];  // w - kernel_size + 1
                input1[0][0].len()]; // h - kernel_size + 1
                input1[0].len()]; //number of conv kernels
                input1.len()]; //input (image) batch size
        
    // residual_add(&residual1,&conv22_output,&mut residual_0_output);  // Residual layer 1
    let (remainder, div) = residual_add_plaintext(
            &input1, 
            &input2,
            &mut residual_output,
            input1_0, 
            input2_0,
            output_0,
            &multiplier_1, 
            &multiplier_2
        );

    let residual_circuit = ResidualAddCircuit{
        input1: input1.clone(),
        input2: input2.clone(),
        output: residual_output.clone(),

        remainder: remainder,
        div: div,

        input1_0: input1_0,
        input2_0: input2_0,
        output_0: output_0,

        multiplier_1: multiplier_1,
        multiplier_2: multiplier_2,
        knit_encoding: knit_encoding,
    };

    (residual_circuit, residual_output)
}


fn resnet18_residual_block_one_private_wrapper(
    input: &Vec<Vec<Vec<Vec<u8>>>>,
    input_0: u8,

    first_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    first_conv_weight_0: u8,
    first_conv_weight_multiplier: Vec<f32>,
    first_conv_output_0: u8,

    second_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    second_conv_weight_0: u8,
    second_conv_weight_multiplier: Vec<f32>,
    second_conv_output_0: u8,

    first_residual_input: &Vec<Vec<Vec<Vec<u8>>>>,
    first_residual_input_0: u8,
    first_residual_input_multiplier: Vec<f32>,

    second_residual_input_multiplier: Vec<f32>,

    output_0: u8,
    padding: usize,
    knit_encoding: bool,
) -> (ConvCircuitU8BitDecomposeOptimization, ConvCircuitU8BitDecomposeOptimization, ResidualAddCircuit, Vec<Vec<Vec<Vec<u8>>>>) {
    let (first_conv_circuit, first_conv_output) = conv_one_private_wrapper(
        &input,
        &first_conv_weights,
        input_0,
        first_conv_weight_0,
        first_conv_output_0,
        first_conv_weight_multiplier.clone(),
        padding,
        knit_encoding,
    );

    let (second_conv_circuit, second_conv_output) = conv_one_private_wrapper(
        &first_conv_output,
        &second_conv_weights,
        first_conv_output_0,
        second_conv_weight_0,
        second_conv_output_0,
        second_conv_weight_multiplier.clone(),
        padding,
        knit_encoding,
    );

    // residual add 1 =================================================================================================================
    let (residual_circuit, residual_output) = residual_add_wrapper(
        &first_residual_input,
        &second_conv_output,
        first_residual_input_0,
        second_conv_output_0,
        output_0,
        first_residual_input_multiplier,
        second_residual_input_multiplier,
        knit_encoding,
    );

    (first_conv_circuit, second_conv_circuit, residual_circuit, residual_output)
}


#[derive(Clone)]
pub struct Resnet18CircuitU8OptimizedLv2PedersenPublicNNWeights {
    pub padding:usize,
    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,

    pub conv21_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv22_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv23_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv24_weights: Vec<Vec<Vec<Vec<u8>>>>,
    //------------------------------ --
    pub conv31_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv32_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv33_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv34_weights: Vec<Vec<Vec<Vec<u8>>>>,

    //------------------------------ --
    pub conv41_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv42_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv43_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv44_weights: Vec<Vec<Vec<Vec<u8>>>>,

    //------------------------------ --
    pub conv51_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv52_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv53_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv54_weights: Vec<Vec<Vec<Vec<u8>>>>,

    pub conv_residual1_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    pub conv_residual3_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    pub conv_residual5_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    pub conv_residual7_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1

    pub fc1_weights: Vec<Vec<u8>>,


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

    pub conv_residual1_output_0:u8,  // residual output_0 1
    pub conv_residual3_output_0:u8,  // residual output_0 1
    pub conv_residual5_output_0:u8,  // residual output_0 1
    pub conv_residual7_output_0:u8,  // residual output_0 1

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

    pub conv_residual1_weights_0:u8,  // residual weights_0 1
    pub conv_residual3_weights_0:u8,  // residual weights_0 1
    pub conv_residual5_weights_0:u8,  // residual weights_0 1
    pub conv_residual7_weights_0:u8,  // residual weights_0 1

    pub fc1_weights_0: u8,


    //multiplier for quantization


    pub conv21_multiplier: Vec<f32>,
    pub conv22_multiplier: Vec<f32>,
    pub conv23_multiplier: Vec<f32>,
    pub conv24_multiplier: Vec<f32>,

    pub conv31_multiplier: Vec<f32>,
    pub conv32_multiplier: Vec<f32>,
    pub conv33_multiplier: Vec<f32>,
    pub conv34_multiplier: Vec<f32>,

    pub conv41_multiplier: Vec<f32>,
    pub conv42_multiplier: Vec<f32>,
    pub conv43_multiplier: Vec<f32>,
    pub conv44_multiplier: Vec<f32>,

    pub conv51_multiplier: Vec<f32>,
    pub conv52_multiplier: Vec<f32>,
    pub conv53_multiplier: Vec<f32>,
    pub conv54_multiplier: Vec<f32>,

    pub conv_residual1_multiplier:Vec<f32>,  // residual multiplier 1
    pub conv_residual3_multiplier:Vec<f32>,  // residual multiplier 1
    pub conv_residual5_multiplier:Vec<f32>,  // residual multiplier 1
    pub conv_residual7_multiplier:Vec<f32>,  // residual multiplier 1

    pub add_residual1_output_0:u8,
    pub add_residual2_output_0:u8,
    pub add_residual3_output_0:u8,
    pub add_residual4_output_0:u8,
    pub add_residual5_output_0:u8,
    pub add_residual6_output_0:u8,
    pub add_residual7_output_0:u8,
    pub add_residual8_output_0:u8,

    pub add_residual1_first_multiplier:Vec<f32>,
    pub add_residual2_first_multiplier:Vec<f32>,
    pub add_residual3_first_multiplier:Vec<f32>,
    pub add_residual4_first_multiplier:Vec<f32>,
    pub add_residual5_first_multiplier:Vec<f32>,
    pub add_residual6_first_multiplier:Vec<f32>,
    pub add_residual7_first_multiplier:Vec<f32>,
    pub add_residual8_first_multiplier:Vec<f32>,

    pub add_residual1_second_multiplier:Vec<f32>,
    pub add_residual2_second_multiplier:Vec<f32>,
    pub add_residual3_second_multiplier:Vec<f32>,
    pub add_residual4_second_multiplier:Vec<f32>,
    pub add_residual5_second_multiplier:Vec<f32>,
    pub add_residual6_second_multiplier:Vec<f32>,
    pub add_residual7_second_multiplier:Vec<f32>,
    pub add_residual8_second_multiplier:Vec<f32>,

    pub multiplier_fc1: Vec<f32>,

    //we do not need multiplier in relu and AvgPool layer
    pub z: Vec<Vec<u8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
    pub knit_encoding: bool,
}

impl ConstraintSynthesizer<Fq> for Resnet18CircuitU8OptimizedLv2PedersenPublicNNWeights {
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

        // residual 1*1 conv 1 =================================================================================
        let (residual_conv1_circuit, residual_conv1_output) = conv_one_private_wrapper(
            &self.x,
            &self.conv_residual1_weight,
            self.x_0,
            self.conv_residual1_weights_0,
            self.conv_residual1_output_0,
            self.conv_residual1_multiplier,
            0,
            self.knit_encoding,            
        );

        residual_conv1_circuit.generate_constraints(cs.clone())?;

        //----------------------------------------------------------------------
        let (conv21_circuit, conv22_circuit, residual1_add_circuit, add_residual1_output) = resnet18_residual_block_one_private_wrapper(
            &self.x,
            self.x_0,
            &self.conv21_weights,
            self.conv21_weights_0,
            self.conv21_multiplier,
            self.conv21_output_0,
            &self.conv22_weights,
            self.conv22_weights_0,
            self.conv22_multiplier,
            self.conv22_output_0,
            &residual_conv1_output,
            self.conv_residual1_output_0,
            self.add_residual1_first_multiplier,
            self.add_residual1_second_multiplier,
            self.add_residual1_output_0,
            self.padding,
            self.knit_encoding,
        );

        conv21_circuit.generate_constraints(cs.clone())?;
        conv22_circuit.generate_constraints(cs.clone())?;
        residual1_add_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for res_block1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // =================================================================================================================================
        let (conv23_circuit, conv24_circuit, residual2_add_circuit, add_residual2_output) = resnet18_residual_block_one_private_wrapper(
            &add_residual1_output,
            self.add_residual1_output_0,
            &self.conv23_weights,
            self.conv23_weights_0,
            self.conv23_multiplier,
            self.conv23_output_0,
            &self.conv24_weights,
            self.conv24_weights_0,
            self.conv24_multiplier,
            self.conv24_output_0,
            &residual_conv1_output,
            self.add_residual1_output_0,
            self.add_residual2_first_multiplier,
            self.add_residual2_second_multiplier,
            self.add_residual2_output_0,
            self.padding,
            self.knit_encoding,
        );

        conv23_circuit.generate_constraints(cs.clone())?;
        conv24_circuit.generate_constraints(cs.clone())?;
        residual2_add_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for res_block2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        
        // =================================================================================================================================
        let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&add_residual2_output.clone(), 2);

        let avg_pool2_circuit = AvgPoolCircuitU8BitDecomposeOptimized {
            // x: conv24_output.clone(),
            x: add_residual2_output.clone(),
            y: avg_pool2_output.clone(),
            kernel_size: 2,
            remainder: avg2_remainder.clone(),
            knit_encoding: self.knit_encoding,
        };
        avg_pool2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // residual 1*1 conv 2 =================================================================================
        let (residual_conv3_circuit, residual_conv3_output) = conv_one_private_wrapper(
            &avg_pool2_output,
            &self.conv_residual3_weight,
            self.add_residual2_output_0,
            self.conv_residual3_weights_0,
            self.conv_residual3_output_0,
            self.conv_residual3_multiplier,
            0,
            self.knit_encoding,            
        );

        residual_conv3_circuit.generate_constraints(cs.clone())?;

        //----------------------------------------------------------------------
        let (conv31_circuit, conv32_circuit, residual3_add_circuit, add_residual3_output) = resnet18_residual_block_one_private_wrapper(
            &avg_pool2_output,
            self.add_residual2_output_0,
            &self.conv31_weights,
            self.conv31_weights_0,
            self.conv31_multiplier,
            self.conv31_output_0,
            &self.conv32_weights,
            self.conv32_weights_0,
            self.conv32_multiplier,
            self.conv32_output_0,
            &residual_conv3_output,
            self.conv_residual3_output_0,
            self.add_residual3_first_multiplier,
            self.add_residual3_second_multiplier,
            self.add_residual3_output_0,
            self.padding,
            self.knit_encoding,
        );

        conv31_circuit.generate_constraints(cs.clone())?;
        conv32_circuit.generate_constraints(cs.clone())?;
        residual3_add_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for res_block3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // =================================================================================================================================
        let (conv33_circuit, conv34_circuit, residual4_add_circuit, add_residual4_output) = resnet18_residual_block_one_private_wrapper(
            &add_residual3_output,
            self.add_residual3_output_0,
            &self.conv33_weights,
            self.conv33_weights_0,
            self.conv33_multiplier,
            self.conv33_output_0,
            &self.conv34_weights,
            self.conv34_weights_0,
            self.conv34_multiplier,
            self.conv34_output_0,
            &add_residual3_output,
            self.add_residual3_output_0,
            self.add_residual4_first_multiplier,
            self.add_residual4_second_multiplier,
            self.add_residual4_output_0,
            self.padding,
            self.knit_encoding,
        );

        conv33_circuit.generate_constraints(cs.clone())?;
        conv34_circuit.generate_constraints(cs.clone())?;
        residual4_add_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for res_block4 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // =================================================================================================================================
        let (avg_pool3_output, avg3_remainder) = avg_pool_with_remainder_scala_u8(&add_residual4_output.clone(), 2);

        let avg_pool3_circuit = AvgPoolCircuitU8BitDecomposeOptimized {
            // x: conv24_output.clone(),
            x: add_residual4_output.clone(),
            y: avg_pool3_output.clone(),
            kernel_size: 2,
            remainder: avg3_remainder.clone(),
            knit_encoding: self.knit_encoding,
        };
        avg_pool3_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // residual 1*1 conv 2 =================================================================================
        let (residual_conv5_circuit, residual_conv5_output) = conv_one_private_wrapper(
            &avg_pool3_output,
            &self.conv_residual5_weight,
            self.add_residual4_output_0,
            self.conv_residual5_weights_0,
            self.conv_residual5_output_0,
            self.conv_residual5_multiplier,
            0,
            self.knit_encoding,            
        );

        residual_conv5_circuit.generate_constraints(cs.clone())?;

        //----------------------------------------------------------------------
        let (conv41_circuit, conv42_circuit, residual5_add_circuit, add_residual5_output) = resnet18_residual_block_one_private_wrapper(
            &avg_pool3_output,
            self.add_residual4_output_0,
            &self.conv41_weights,
            self.conv41_weights_0,
            self.conv41_multiplier,
            self.conv41_output_0,
            &self.conv42_weights,
            self.conv42_weights_0,
            self.conv42_multiplier,
            self.conv42_output_0,
            &residual_conv5_output,
            self.conv_residual5_output_0,
            self.add_residual5_first_multiplier,
            self.add_residual5_second_multiplier,
            self.add_residual5_output_0,
            self.padding,
            self.knit_encoding,
        );

        conv41_circuit.generate_constraints(cs.clone())?;
        conv42_circuit.generate_constraints(cs.clone())?;
        residual5_add_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for res_block5 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // =================================================================================================================================
        let (conv43_circuit, conv44_circuit, residual6_add_circuit, add_residual6_output) = resnet18_residual_block_one_private_wrapper(
            &add_residual5_output,
            self.add_residual5_output_0,
            &self.conv43_weights,
            self.conv43_weights_0,
            self.conv43_multiplier,
            self.conv43_output_0,
            &self.conv44_weights,
            self.conv44_weights_0,
            self.conv44_multiplier,
            self.conv44_output_0,
            &add_residual5_output,
            self.add_residual5_output_0,
            self.add_residual6_first_multiplier,
            self.add_residual6_second_multiplier,
            self.add_residual6_output_0,
            self.padding,
            self.knit_encoding,
        );

        conv43_circuit.generate_constraints(cs.clone())?;
        conv44_circuit.generate_constraints(cs.clone())?;
        residual6_add_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for res_block6 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // =================================================================================================================================
        let (avg_pool4_output, avg4_remainder) = avg_pool_with_remainder_scala_u8(&add_residual6_output.clone(), 2);

        let avg_pool4_circuit = AvgPoolCircuitU8BitDecomposeOptimized {
            // x: conv24_output.clone(),
            x: add_residual6_output.clone(),
            y: avg_pool4_output.clone(),
            kernel_size: 2,
            remainder: avg4_remainder.clone(),
            knit_encoding: self.knit_encoding,
        };
        avg_pool4_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool4 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();


        // residual 1*1 conv 2 =================================================================================
        let (residual_conv7_circuit, residual_conv7_output) = conv_one_private_wrapper(
            &avg_pool4_output,
            &self.conv_residual7_weight,
            self.add_residual6_output_0,
            self.conv_residual7_weights_0,
            self.conv_residual7_output_0,
            self.conv_residual7_multiplier,
            0,
            self.knit_encoding,            
        );

        residual_conv7_circuit.generate_constraints(cs.clone())?;

        //----------------------------------------------------------------------
        let (conv51_circuit, conv52_circuit, residual7_add_circuit, add_residual7_output) = resnet18_residual_block_one_private_wrapper(
            &avg_pool4_output,
            self.add_residual6_output_0,
            &self.conv51_weights,
            self.conv51_weights_0,
            self.conv51_multiplier,
            self.conv51_output_0,
            &self.conv52_weights,
            self.conv52_weights_0,
            self.conv52_multiplier,
            self.conv52_output_0,
            &residual_conv7_output,
            self.conv_residual7_output_0,
            self.add_residual7_first_multiplier,
            self.add_residual7_second_multiplier,
            self.add_residual7_output_0,
            self.padding,
            self.knit_encoding,
        );

        conv51_circuit.generate_constraints(cs.clone())?;
        conv52_circuit.generate_constraints(cs.clone())?;
        residual7_add_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for res_block7 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // =================================================================================================================================
        let (conv53_circuit, conv54_circuit, residual8_add_circuit, add_residual8_output) = resnet18_residual_block_one_private_wrapper(
            &add_residual7_output,
            self.add_residual7_output_0,
            &self.conv53_weights,
            self.conv53_weights_0,
            self.conv53_multiplier,
            self.conv53_output_0,
            &self.conv54_weights,
            self.conv54_weights_0,
            self.conv54_multiplier,
            self.conv54_output_0,
            &add_residual7_output,
            self.add_residual7_output_0,
            self.add_residual8_first_multiplier,
            self.add_residual8_second_multiplier,
            self.add_residual8_output_0,
            self.padding,
            self.knit_encoding,
        );

        conv53_circuit.generate_constraints(cs.clone())?;
        conv54_circuit.generate_constraints(cs.clone())?;
        residual8_add_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for res_block8 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        // =================================================================================================================================
        let (avg_pool5_output, avg5_remainder) = avg_pool_with_remainder_scala_u8(&add_residual8_output.clone(), 4);

        let avg_pool5_circuit = AvgPoolCircuitU8BitDecomposeOptimized {
            // x: conv24_output.clone(),
            x: add_residual8_output.clone(),
            y: avg_pool5_output.clone(),
            kernel_size: 4,
            remainder: avg5_remainder.clone(),
            knit_encoding: self.knit_encoding,
        };
        avg_pool5_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool5 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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
            let fc1_circuit = FCCircuitU8BitDecomposeOptimized {
                x: transformed_avg_pool5_output_output[i].clone(),
                l1_mat: self.fc1_weights.clone(),
                y: fc1_output[i].clone(),
                remainder: remainder_fc1.clone(),
                div: div_fc1.clone(),
                x_0: self.conv54_output_0,
                l1_mat_0: self.fc1_weights_0,
                y_0: self.fc1_output_0,

                multiplier: self.multiplier_fc1.clone(),
                knit_encoding: self.knit_encoding,
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
