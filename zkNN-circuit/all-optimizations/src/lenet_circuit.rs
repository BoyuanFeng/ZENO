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


#[derive(Clone)]
pub struct LeNetCircuitU8OptimizedLv2PedersenRecognition {
    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,
    pub conv1_weights: Vec<Vec<Vec<Vec<u8>>>>,

    pub conv2_weights: Vec<Vec<Vec<Vec<u8>>>>,

    pub conv3_weights: Vec<Vec<Vec<Vec<u8>>>>,

    pub fc1_weights: Vec<Vec<u8>>,

    pub fc2_weights: Vec<Vec<u8>>,


    //zero points for quantization.
    pub x_0: u8,
    pub conv1_output_0: u8,
    pub conv2_output_0: u8,
    pub conv3_output_0: u8,
    pub fc1_output_0: u8,
    pub fc2_output_0: u8, // which is also lenet output(z) zero point

    pub conv1_weights_0: u8,
    pub conv2_weights_0: u8,
    pub conv3_weights_0: u8,
    pub fc1_weights_0: u8,
    pub fc2_weights_0: u8,

    //multiplier for quantization
    pub multiplier_conv1: Vec<f32>,
    pub multiplier_conv2: Vec<f32>,
    pub multiplier_conv3: Vec<f32>,
    pub multiplier_fc1: Vec<f32>,
    pub multiplier_fc2: Vec<f32>,
    //we do not need multiplier in relu and AvgPool layer
    pub z: Vec<Vec<u8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,

    pub person_feature_vector: Vec<u8>,
    pub threshold: u8,
    pub result: bool,
    pub knit_encoding: bool,
}

impl ConstraintSynthesizer<Fq> for LeNetCircuitU8OptimizedLv2PedersenRecognition {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "LeNetCircuitU8OptimizedLv3PedersenRecognition is setup mode: {}",
            cs.is_in_setup_mode()
        );

        let full_circuit = LeNetCircuitU8OptimizedLv2Pedersen {
            params: self.params.clone(),
            x: self.x.clone(),
            x_com: self.x_com.clone(),
            x_open: self.x_open,

            conv1_weights: self.conv1_weights.clone(),

            conv2_weights: self.conv2_weights.clone(),

            conv3_weights: self.conv3_weights.clone(),

            fc1_weights: self.fc1_weights.clone(),

            fc2_weights: self.fc2_weights.clone(),


            //zero points for quantization.
            x_0: self.x_0,
            conv1_output_0: self.conv1_output_0,
            conv2_output_0: self.conv2_output_0,
            conv3_output_0: self.conv3_output_0,
            fc1_output_0: self.fc1_output_0,
            fc2_output_0: self.fc2_output_0, // which is also lenet output(z) zero point

            conv1_weights_0: self.conv1_weights_0,
            conv2_weights_0: self.conv2_weights_0,
            conv3_weights_0: self.conv3_weights_0,
            fc1_weights_0: self.fc1_weights_0,
            fc2_weights_0: self.fc2_weights_0,

            //multiplier for quantization
            multiplier_conv1: self.multiplier_conv1.clone(),
            multiplier_conv2: self.multiplier_conv2.clone(),
            multiplier_conv3: self.multiplier_conv3.clone(),
            multiplier_fc1: self.multiplier_fc1.clone(),
            multiplier_fc2: self.multiplier_fc2.clone(),

            z: self.z.clone(),
            z_open: self.z_open,
            z_com: self.z_com,
            knit_encoding: self.knit_encoding,
        };

        //we only do one image in zk proof.
        let similarity_circuit = CosineSimilarityCircuitU8 {
            vec1: self.z[0].clone(),
            vec2: self.person_feature_vector.clone(),
            threshold: self.threshold,
            result: self.result,
        };

        full_circuit
            .clone()
            .generate_constraints(cs.clone())
            .unwrap();
        similarity_circuit
            .clone()
            .generate_constraints(cs.clone())
            .unwrap();

        Ok(())
    }
}


#[derive(Clone)]
pub struct LeNetCircuitU8OptimizedLv2PedersenClassification {
    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,

    pub conv1_weights: Vec<Vec<Vec<Vec<u8>>>>,

    pub conv2_weights: Vec<Vec<Vec<Vec<u8>>>>,

    pub conv3_weights: Vec<Vec<Vec<Vec<u8>>>>,

    pub fc1_weights: Vec<Vec<u8>>,

    pub fc2_weights: Vec<Vec<u8>>,

    //zero points for quantization.
    pub x_0: u8,
    pub conv1_output_0: u8,
    pub conv2_output_0: u8,
    pub conv3_output_0: u8,
    pub fc1_output_0: u8,
    pub fc2_output_0: u8, // which is also lenet output(z) zero point

    pub conv1_weights_0: u8,
    pub conv2_weights_0: u8,
    pub conv3_weights_0: u8,
    pub fc1_weights_0: u8,
    pub fc2_weights_0: u8,

    //multiplier for quantization
    pub multiplier_conv1: Vec<f32>,
    pub multiplier_conv2: Vec<f32>,
    pub multiplier_conv3: Vec<f32>,
    pub multiplier_fc1: Vec<f32>,
    pub multiplier_fc2: Vec<f32>,
    //we do not need multiplier in relu and AvgPool layer
    pub z: Vec<Vec<u8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
    pub argmax_res: usize,
    pub knit_encoding: bool,
}

impl ConstraintSynthesizer<Fq> for LeNetCircuitU8OptimizedLv2PedersenClassification {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        #[cfg(debug_assertion)]
        println!(
            "LeNetCircuitU8OptimizedLv3PedersenClassification is setup mode: {}",
            cs.is_in_setup_mode()
        );

        let full_circuit = LeNetCircuitU8OptimizedLv2Pedersen {
            params: self.params.clone(),
            x: self.x.clone(),
            x_com: self.x_com.clone(),
            x_open: self.x_open,

            conv1_weights: self.conv1_weights.clone(),

            conv2_weights: self.conv2_weights.clone(),

            conv3_weights: self.conv3_weights.clone(),

            fc1_weights: self.fc1_weights.clone(),

            fc2_weights: self.fc2_weights.clone(),

            //zero points for quantization.
            x_0: self.x_0,
            conv1_output_0: self.conv1_output_0,
            conv2_output_0: self.conv2_output_0,
            conv3_output_0: self.conv3_output_0,
            fc1_output_0: self.fc1_output_0,
            fc2_output_0: self.fc2_output_0, // which is also lenet output(z) zero point

            conv1_weights_0: self.conv1_weights_0,
            conv2_weights_0: self.conv2_weights_0,
            conv3_weights_0: self.conv3_weights_0,
            fc1_weights_0: self.fc1_weights_0,
            fc2_weights_0: self.fc2_weights_0,

            //multiplier for quantization
            multiplier_conv1: self.multiplier_conv1.clone(),
            multiplier_conv2: self.multiplier_conv2.clone(),
            multiplier_conv3: self.multiplier_conv3.clone(),
            multiplier_fc1: self.multiplier_fc1.clone(),
            multiplier_fc2: self.multiplier_fc2.clone(),

            z: self.z.clone(),
            z_open: self.z_open,
            z_com: self.z_com,
            knit_encoding: self.knit_encoding,
        };

        //we only do one image in zk proof.
        let argmax_circuit = ArgmaxCircuitU8 {
            input: self.z[0].clone(),
            argmax_res: self.argmax_res.clone(),
        };

        full_circuit
            .clone()
            .generate_constraints(cs.clone())
            .unwrap();
        argmax_circuit
            .clone()
            .generate_constraints(cs.clone())
            .unwrap();

        Ok(())
    }
}






#[derive(Clone)]
pub struct LeNetCircuitU8OptimizedLv2Pedersen {
    pub x: Vec<Vec<Vec<Vec<u8>>>>,
    pub x_open: PedersenRandomness,
    pub x_com: PedersenCommitment,
    pub params: PedersenParam,
    pub conv1_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv2_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub conv3_weights: Vec<Vec<Vec<Vec<u8>>>>,
    pub fc1_weights: Vec<Vec<u8>>,
    pub fc2_weights: Vec<Vec<u8>>,

    //zero points for quantization.
    pub x_0: u8,
    pub conv1_output_0: u8,
    pub conv2_output_0: u8,
    pub conv3_output_0: u8,
    pub fc1_output_0: u8,
    pub fc2_output_0: u8, // which is also lenet output(z) zero point

    pub conv1_weights_0: u8,
    pub conv2_weights_0: u8,
    pub conv3_weights_0: u8,
    pub fc1_weights_0: u8,
    pub fc2_weights_0: u8,

    //multiplier for quantization
    pub multiplier_conv1: Vec<f32>,
    pub multiplier_conv2: Vec<f32>,
    pub multiplier_conv3: Vec<f32>,
    pub multiplier_fc1: Vec<f32>,
    pub multiplier_fc2: Vec<f32>,
    //we do not need multiplier in relu and AvgPool layer
    pub z: Vec<Vec<u8>>,
    pub z_open: PedersenRandomness,
    pub z_com: PedersenCommitment,
    pub knit_encoding: bool,
}

impl ConstraintSynthesizer<Fq> for LeNetCircuitU8OptimizedLv2Pedersen {
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

        let output: Vec<Vec<u8>> = lenet_circuit_forward_u8(
            self.x.clone(),
            self.conv1_weights.clone(),
            self.conv2_weights.clone(),
            self.conv3_weights.clone(),
            self.fc1_weights.clone(),
            self.fc2_weights.clone(),
            self.x_0,
            self.conv1_output_0,
            self.conv2_output_0,
            self.conv3_output_0,
            self.fc1_output_0,
            self.fc2_output_0,
            self.conv1_weights_0,
            self.conv2_weights_0,
            self.conv3_weights_0,
            self.fc1_weights_0,
            self.fc2_weights_0,
            self.multiplier_conv1.clone(),
            self.multiplier_conv2.clone(),
            self.multiplier_conv3.clone(),
            self.multiplier_fc1.clone(),
            self.multiplier_fc2.clone(),
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

        //layer 1
        //conv1
        let mut conv1_output = vec![vec![vec![vec![0u8; self.x[0][0][0].len() - self.conv1_weights[0][0][0].len() + 1];  // w - kernel_size  + 1
                                            self.x[0][0].len() - self.conv1_weights[0][0].len() + 1]; // h - kernel_size + 1
                                            self.conv1_weights.len()]; //number of conv kernels
                                            self.x.len()]; //input (image) batch size
        let (remainder_conv1, div_conv1) = vec_conv_with_remainder_u8(
            &self.x,
            &self.conv1_weights,
            &mut conv1_output,
            self.x_0,
            self.conv1_weights_0,
            self.conv1_output_0,
            &self.multiplier_conv1,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv1_circuit = ConvCircuitU8BitDecomposeOptimization {
            x: self.x.clone(),
            conv_kernel: self.conv1_weights.clone(),
            y: conv1_output.clone(),
            remainder: remainder_conv1.clone(),
            div: div_conv1.clone(),

            x_0: self.x_0,
            conv_kernel_0: self.conv1_weights_0,
            y_0: self.conv1_output_0,

            multiplier: self.multiplier_conv1,
            knit_encoding: self.knit_encoding,

        };
        conv1_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        //println!("conv1 {:?}", conv1_output);

        //relu1
        let relu_input = conv1_output.clone(); //save the input for verification
        relu4d_u8(&mut conv1_output, self.conv1_output_0);

        let flattened_relu_input3d: Vec<Vec<Vec<u8>>> = relu_input.into_iter().flatten().collect();
        let flattened_relu_input2d: Vec<Vec<u8>> =
            flattened_relu_input3d.into_iter().flatten().collect();
        let flattened_relu_input1d: Vec<u8> =
            flattened_relu_input2d.into_iter().flatten().collect();

        let flattened_conv1_output3d: Vec<Vec<Vec<u8>>> =
            conv1_output.clone().into_iter().flatten().collect();
        let flattened_conv1_output2d: Vec<Vec<u8>> =
            flattened_conv1_output3d.into_iter().flatten().collect();
        let flattened_conv1_output1d: Vec<u8> =
            flattened_conv1_output2d.into_iter().flatten().collect();

        let relu1_circuit = ReLUCircuitU8 {
            y_in: flattened_relu_input1d,
            y_out: flattened_conv1_output1d,
            y_zeropoint: self.conv1_output_0,
        };
        relu1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg_pool1

        let (avg_pool1_output, avg1_remainder) = avg_pool_with_remainder_scala_u8(&conv1_output, 2);
        let avg_pool1_circuit = AvgPoolCircuitU8BitDecomposeOptimized {
            x: conv1_output.clone(),
            y: avg_pool1_output.clone(),
            kernel_size: 2,
            remainder: avg1_remainder.clone(),
            knit_encoding: self.knit_encoding,

        };
        avg_pool1_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for AvgPool1 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //layer 2 :
        //Conv2 -> relu -> AvgPool
        let mut conv2_output = vec![vec![vec![vec![0u8; avg_pool1_output[0][0][0].len() - self.conv2_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                        avg_pool1_output[0][0].len() - self.conv2_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                        self.conv2_weights.len()]; //number of conv kernels
                                                                        avg_pool1_output.len()]; //input (image) batch size

        let (remainder_conv2, div_conv2) = vec_conv_with_remainder_u8(
            &avg_pool1_output,
            &self.conv2_weights,
            &mut conv2_output,
            self.conv1_output_0,
            self.conv2_weights_0,
            self.conv2_output_0,
            &self.multiplier_conv2,
        );

        //vector dot product in conv1 is too short! SIMD bit decompostion overhead is more than the benefits of embedding four vectors dot product!
        let conv2_circuit = ConvCircuitU8BitDecomposeOptimization {
            x: avg_pool1_output.clone(),
            conv_kernel: self.conv2_weights.clone(),
            y: conv2_output.clone(),
            remainder: remainder_conv2.clone(),
            div: div_conv2.clone(),

            x_0: self.conv1_output_0,
            conv_kernel_0: self.conv2_weights_0,
            y_0: self.conv2_output_0,

            multiplier: self.multiplier_conv2,
            knit_encoding: self.knit_encoding,

        };
        conv2_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //println!("conv2 {:?}", conv2_output);

        //relu2 layer
        let relu2_input = conv2_output.clone(); //save the input for verification
        relu4d_u8(&mut conv2_output, self.conv2_output_0);
        let flattened_relu2_input3d: Vec<Vec<Vec<u8>>> =
            relu2_input.into_iter().flatten().collect();
        let flattened_relu2_input2d: Vec<Vec<u8>> =
            flattened_relu2_input3d.into_iter().flatten().collect();
        let flattened_relu2_input1d: Vec<u8> =
            flattened_relu2_input2d.into_iter().flatten().collect();

        let flattened_conv2_output3d: Vec<Vec<Vec<u8>>> =
            conv2_output.clone().into_iter().flatten().collect();
        let flattened_conv2_output2d: Vec<Vec<u8>> =
            flattened_conv2_output3d.into_iter().flatten().collect();
        let flattened_conv2_output1d: Vec<u8> =
            flattened_conv2_output2d.into_iter().flatten().collect();
        let relu2_circuit = ReLUCircuitU8 {
            y_in: flattened_relu2_input1d,
            y_out: flattened_conv2_output1d,
            y_zeropoint: self.conv2_output_0,
        };
        relu2_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu2 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        //avg pool2 layer
        //let avg_pool2_output = avg_pool_scala_u8(&conv2_output, self.conv2_weights.len());
        let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&conv2_output, 2);
        let avg_pool2_circuit = AvgPoolCircuitU8BitDecomposeOptimized {
            x: conv2_output.clone(),
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

        //layer 3 :
        //Conv3 -> relu -> reshape output for following FC layer
        let mut conv3_output = vec![vec![vec![vec![0u8; avg_pool2_output[0][0][0].len() - self.conv3_weights[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                            avg_pool2_output[0][0].len() - self.conv3_weights[0][0].len()+ 1]; // h - kernel_size+ 1
                                                                            self.conv3_weights.len()]; //number of conv kernels
                                                                            avg_pool2_output.len()]; //input (image) batch size
                                                                                                     //conv3 layer
        let (remainder_conv3, div_conv3) = vec_conv_with_remainder_u8(
            &avg_pool2_output,
            &self.conv3_weights,
            &mut conv3_output,
            self.conv2_output_0,
            self.conv3_weights_0,
            self.conv3_output_0,
            &self.multiplier_conv3,
        );

        let conv3_circuit = ConvCircuitU8BitDecomposeOptimization {
            x: avg_pool2_output.clone(),
            conv_kernel: self.conv3_weights,
            y: conv3_output.clone(),
            remainder: remainder_conv3.clone(),
            div: div_conv3.clone(),

            x_0: self.conv2_output_0,
            conv_kernel_0: self.conv3_weights_0,
            y_0: self.conv3_output_0,

            multiplier: self.multiplier_conv3,
            knit_encoding: self.knit_encoding,

        };
        conv3_circuit.generate_constraints(cs.clone())?;

        // #[cfg(debug_assertion)]
        println!(
            "Number of constraints for Conv3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();
        //println!("conv3 {:?}", conv3_output);

        //relu3 layer
        let relu3_input = conv3_output.clone(); //save the input for verification
        relu4d_u8(&mut conv3_output, self.conv3_output_0);
        let flattened_relu3_input3d: Vec<Vec<Vec<u8>>> =
            relu3_input.into_iter().flatten().collect();
        let flattened_relu3_input2d: Vec<Vec<u8>> =
            flattened_relu3_input3d.into_iter().flatten().collect();
        let flattened_relu3_input1d: Vec<u8> =
            flattened_relu3_input2d.into_iter().flatten().collect();

        let flattened_conv3_output3d: Vec<Vec<Vec<u8>>> =
            conv3_output.clone().into_iter().flatten().collect();
        let flattened_conv3_output2d: Vec<Vec<u8>> =
            flattened_conv3_output3d.into_iter().flatten().collect();
        let flattened_conv3_output1d: Vec<u8> =
            flattened_conv3_output2d.into_iter().flatten().collect();
        let relu3_circuit = ReLUCircuitU8 {
            y_in: flattened_relu3_input1d,
            y_out: flattened_conv3_output1d,
            y_zeropoint: self.conv3_output_0,
        };
        relu3_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu3 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

        let mut transformed_conv3_output =
            vec![
                vec![
                    0u8;
                    conv3_output[0].len() * conv3_output[0][0].len() * conv3_output[0][0][0].len()
                ];
                conv3_output.len()
            ];
        for i in 0..conv3_output.len() {
            let mut counter = 0;
            for j in 0..conv3_output[0].len() {
                for p in 0..conv3_output[0][0].len() {
                    for q in 0..conv3_output[0][0][0].len() {
                        transformed_conv3_output[i][counter] = conv3_output[i][j][p][q];
                        counter += 1;
                    }
                }
            }
        }

        //layer 4 :
        //FC1 -> relu
        let mut fc1_output = vec![vec![0u8; self.fc1_weights.len()];  // channels
                                            transformed_conv3_output.len()]; //batch size
        let fc1_weight_ref: Vec<&[u8]> = self.fc1_weights.iter().map(|x| x.as_ref()).collect();

        //println!("FC1 {} == {}?", self.fc1_weights[0].len(), transformed_conv3_output[0].len());
        for i in 0..transformed_conv3_output.len() {
            //iterate through each image in the batch
            //in the zkp nn system, we feed one image in each batch to reduce the overhead.
            let (remainder_fc1, div_fc1) = vec_mat_mul_with_remainder_u8(
                &transformed_conv3_output[i],
                fc1_weight_ref[..].as_ref(),
                &mut fc1_output[i],
                self.conv3_output_0,
                self.fc1_weights_0,
                self.fc1_output_0,
                &self.multiplier_fc1.clone(),
            );

            //because the vector dot product is too short. SIMD can not reduce the number of contsraints
            let fc1_circuit = FCCircuitU8BitDecomposeOptimized {
                x: transformed_conv3_output[i].clone(),
                l1_mat: self.fc1_weights.clone(),
                y: fc1_output[i].clone(),
                remainder: remainder_fc1.clone(),
                div: div_fc1.clone(),
                x_0: self.conv3_output_0,
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

        //relu4 layer
        let relu4_input = fc1_output.clone();
        relu2d_u8(&mut fc1_output, self.fc1_output_0);
        let flattened_relu4_input1d: Vec<u8> = relu4_input.into_iter().flatten().collect();
        let flattened_relu4_output1d: Vec<u8> = fc1_output.clone().into_iter().flatten().collect();
        let relu4_circuit = ReLUCircuitU8 {
            y_in: flattened_relu4_input1d,
            y_out: flattened_relu4_output1d,
            y_zeropoint: self.fc1_output_0,
        };
        relu4_circuit.generate_constraints(cs.clone())?;
        println!(
            "Number of constraints for Relu4 {} accumulated constraints {}",
            cs.num_constraints() - _cir_number,
            cs.num_constraints()
        );
        _cir_number = cs.num_constraints();

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
            let fc2_circuit = FCCircuitU8BitDecomposeOptimized {
                x: fc1_output[i].clone(),
                l1_mat: self.fc2_weights.clone(),
                y: fc2_output[i].clone(),

                x_0: self.fc1_output_0,
                l1_mat_0: self.fc2_weights_0,
                y_0: self.fc2_output_0,

                multiplier: self.multiplier_fc2.clone(),

                remainder: remainder_fc2.clone(),
                div: div_fc2.clone(),
                knit_encoding: self.knit_encoding,
            };
            fc2_circuit.generate_constraints(cs.clone())?;

            println!(
                "Number of constraints FC2 {} accumulated constraints {}",
                cs.num_constraints() - _cir_number,
                cs.num_constraints()
            );
            _cir_number = cs.num_constraints();
        }

        Ok(())
    }
}
