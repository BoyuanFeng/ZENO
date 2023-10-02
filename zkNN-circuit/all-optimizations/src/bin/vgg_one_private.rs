use algebra::ed_on_bls12_381::*;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::time::Instant;
use zk_ml_knit_encoding::vgg_circuit::*;
use zk_ml_knit_encoding::pedersen_commit::*;
use zk_ml_knit_encoding::read_inputs::*;
use zk_ml_knit_encoding::vanilla::*;

fn rand_4d_vec_generator(a : usize, b : usize, c : usize, d:usize) -> Vec<Vec<Vec<Vec<u8>>>>{
    let mut res = vec![vec![vec![vec![2; d]; c]; b]; a];

    res
}

fn rand_2d_vec_generator(a : usize, b : usize) -> Vec<Vec<u8>>{
    let mut res = vec![vec![2; b]; a];

    res
}

fn main() {
    let mut rng = rand::thread_rng();

    println!("VGG16 testing. use null parameters and input just to benchmark performance");
 
    let x = rand_4d_vec_generator(1, 3, 32, 32);
    // let x = rand_4d_vec_generator(1, 1, 64, 64);

    // let len1 = 16;
    // let len2 = 32;
    // let len3 = 64;
    // let len4 = 128;
    // let len5 = 256;

    let conv11_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16,3, 3,3);
    let conv12_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16,16, 3,3);

    let conv21_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32,16, 3,3);
    let conv22_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32,32, 3,3);

    let conv31_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64,32, 3,3);
    let conv32_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64,64, 3,3);
    let conv33_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64,64, 1,1);

    let conv41_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,64, 3,3);
    let conv42_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 3,3);
    let conv43_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 1,1);

    let conv51_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 3,3);
    let conv52_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 3,3);
    let conv53_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128,128, 1,1);

    // let fc1_w: Vec<Vec<u8>> = rand_2d_vec_generator(1024, 1024);
    // let fc2_w: Vec<Vec<u8>> = rand_2d_vec_generator(256, 1024);
    // let fc3_w: Vec<Vec<u8>> = rand_2d_vec_generator(10, 256);

    let fc1_w: Vec<Vec<u8>> = rand_2d_vec_generator(64, 128);
    let fc2_w: Vec<Vec<u8>> = rand_2d_vec_generator(32, 64);
    let fc3_w: Vec<Vec<u8>> = rand_2d_vec_generator(10, 32);

    let multiplier_conv11: Vec<f32> = vec![1.5; 16];
    let multiplier_conv12: Vec<f32> = vec![1.5; 16];

    let multiplier_conv21: Vec<f32> = vec![1.5; 32];
    let multiplier_conv22: Vec<f32> = vec![1.5; 32];

    let multiplier_conv31: Vec<f32> = vec![1.5; 64];
    let multiplier_conv32: Vec<f32> = vec![1.5; 64];
    let multiplier_conv33: Vec<f32> = vec![1.5; 64];

    let multiplier_conv41: Vec<f32> = vec![1.5; 128];
    let multiplier_conv42: Vec<f32> = vec![1.5; 128];
    let multiplier_conv43: Vec<f32> = vec![1.5; 128];

    let multiplier_conv51: Vec<f32> = vec![1.5; 128];
    let multiplier_conv52: Vec<f32> = vec![1.5; 128];
    let multiplier_conv53: Vec<f32> = vec![1.5; 128];

    let multiplier_fc1: Vec<f32> = vec![1.5; 64];
    let multiplier_fc2: Vec<f32> = vec![1.5; 32];
    let multiplier_fc3: Vec<f32> = vec![1.5; 10];


    let length = 128;
    let x_0: Vec<u8> = vec![length];
    let conv11_output_0: Vec<u8> = vec![length];
    let conv12_output_0: Vec<u8> = vec![length];
    let conv21_output_0: Vec<u8> = vec![length];
    let conv22_output_0: Vec<u8> = vec![length];

    let conv31_output_0: Vec<u8> = vec![length];
    let conv32_output_0: Vec<u8> = vec![length];
    let conv33_output_0: Vec<u8> = vec![length];
    let conv41_output_0: Vec<u8> = vec![length];
    let conv42_output_0: Vec<u8> = vec![length];
    let conv43_output_0: Vec<u8> = vec![length];
    let conv51_output_0: Vec<u8> = vec![length];
    let conv52_output_0: Vec<u8> = vec![length];
    let conv53_output_0: Vec<u8> = vec![length];

    let fc1_output_0: Vec<u8> = vec![length];
    let fc2_output_0: Vec<u8> = vec![length];
    let fc3_output_0: Vec<u8> = vec![length];

    let conv11_weights_0: Vec<u8> = vec![length];
    let conv12_weights_0: Vec<u8> = vec![length];
    let conv21_weights_0: Vec<u8> = vec![length];
    let conv22_weights_0: Vec<u8> = vec![length];
    let conv31_weights_0: Vec<u8> = vec![length];
    let conv32_weights_0: Vec<u8> = vec![length];
    let conv33_weights_0: Vec<u8> = vec![length];
    let conv41_weights_0: Vec<u8> = vec![length];
    let conv42_weights_0: Vec<u8> = vec![length];
    let conv43_weights_0: Vec<u8> = vec![length];
    let conv51_weights_0: Vec<u8> = vec![length];
    let conv52_weights_0: Vec<u8> = vec![length];
    let conv53_weights_0: Vec<u8> = vec![length];
    let fc1_weights_0: Vec<u8> = vec![length];
    let fc2_weights_0: Vec<u8> = vec![length];
    let fc3_weights_0: Vec<u8> = vec![length];

    println!("finish reading parameters");

    let z: Vec<Vec<u8>> = vgg_circuit_forward_u8(
        x.clone(),
        conv11_w.clone(),
        conv12_w.clone(),
        conv21_w.clone(),
        conv22_w.clone(),
        conv31_w.clone(),
        conv32_w.clone(),
        conv33_w.clone(),
        conv41_w.clone(),
        conv42_w.clone(),
        conv43_w.clone(),
        conv51_w.clone(),
        conv52_w.clone(),
        conv53_w.clone(),
        fc1_w.clone(),
        fc2_w.clone(),
        fc3_w.clone(),

        x_0[0],
        conv11_output_0[0],
        conv12_output_0[0],
        conv21_output_0[0],
        conv22_output_0[0],
        conv31_output_0[0],
        conv32_output_0[0],
        conv33_output_0[0],
        conv41_output_0[0],
        conv42_output_0[0],
        conv43_output_0[0],
        conv51_output_0[0],
        conv52_output_0[0],
        conv53_output_0[0],
        fc1_output_0[0],
        fc2_output_0[0], 
        fc3_output_0[0], 

        conv11_weights_0[0],
        conv12_weights_0[0],
        conv21_weights_0[0],
        conv22_weights_0[0],
        conv31_weights_0[0],
        conv32_weights_0[0],
        conv33_weights_0[0],
        conv41_weights_0[0],
        conv42_weights_0[0],
        conv43_weights_0[0],
        conv51_weights_0[0],
        conv52_weights_0[0],
        conv53_weights_0[0],
        fc1_weights_0[0],
        fc2_weights_0[0],
        fc3_weights_0[0],

        multiplier_conv11.clone(),
        multiplier_conv12.clone(),
        multiplier_conv21.clone(),
        multiplier_conv22.clone(),
        multiplier_conv31.clone(),
        multiplier_conv32.clone(),
        multiplier_conv33.clone(),
        multiplier_conv41.clone(),
        multiplier_conv42.clone(),
        multiplier_conv43.clone(),
        multiplier_conv51.clone(),
        multiplier_conv52.clone(),
        multiplier_conv53.clone(),
        multiplier_fc1.clone(),
        multiplier_fc2.clone(),
        multiplier_fc3.clone(),

    );

    println!("finish forwarding");

    //batch size is only one for faster calculation of total constraints
    let flattened_x3d: Vec<Vec<Vec<u8>>> = x.clone().into_iter().flatten().collect();
    let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
    let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();

    let flattened_z1d: Vec<u8> = z.clone().into_iter().flatten().collect();

    //println!("x outside {:?}", x.clone());
    println!("z outside {:?}", flattened_z1d.clone());
    let begin = Instant::now();
    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&flattened_x1d, &param, &x_open);

    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&flattened_z1d, &param, &z_open);
    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));
    //we only do one image in zk proof.

    let full_circuit = VGGCircuitU8OptimizedLv2PedersenPublicNNWeights {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,

        conv11_weights: conv11_w.clone(),
        conv12_weights: conv12_w.clone(),
        conv21_weights: conv21_w.clone(),
        conv22_weights: conv22_w.clone(),

        conv31_weights: conv31_w.clone(),
        conv32_weights: conv32_w.clone(),
        conv33_weights: conv33_w.clone(),

        conv41_weights: conv41_w.clone(),
        conv42_weights: conv42_w.clone(),
        conv43_weights: conv43_w.clone(),

        conv51_weights: conv51_w.clone(),
        conv52_weights: conv52_w.clone(),
        conv53_weights: conv53_w.clone(),

        fc1_weights: fc1_w.clone(),
        fc2_weights: fc2_w.clone(),
        fc3_weights: fc3_w.clone(),

        //zero points for quantization.
        x_0: x_0[0],
        conv11_output_0: conv11_output_0[0],
        conv12_output_0: conv12_output_0[0],
        conv21_output_0: conv21_output_0[0],
        conv22_output_0: conv22_output_0[0],

        conv31_output_0: conv31_output_0[0],
        conv32_output_0: conv32_output_0[0],
        conv33_output_0: conv33_output_0[0],

        conv41_output_0: conv41_output_0[0],
        conv42_output_0: conv42_output_0[0],
        conv43_output_0: conv43_output_0[0],
        
        conv51_output_0: conv51_output_0[0],
        conv52_output_0: conv52_output_0[0],
        conv53_output_0: conv53_output_0[0],

        fc1_output_0: fc1_output_0[0],
        fc2_output_0: fc2_output_0[0],
        fc3_output_0: fc3_output_0[0],

        conv11_weights_0: conv11_weights_0[0],
        conv12_weights_0: conv12_weights_0[0],
        conv21_weights_0: conv21_weights_0[0],
        conv22_weights_0: conv22_weights_0[0],
        conv31_weights_0: conv31_weights_0[0],
        conv32_weights_0: conv32_weights_0[0],
        conv33_weights_0: conv33_weights_0[0],

        conv41_weights_0: conv41_weights_0[0],
        conv42_weights_0: conv42_weights_0[0],
        conv43_weights_0: conv43_weights_0[0],
        conv51_weights_0: conv51_weights_0[0],
        conv52_weights_0: conv52_weights_0[0],
        conv53_weights_0: conv53_weights_0[0],
        fc1_weights_0: fc1_weights_0[0],
        fc2_weights_0: fc2_weights_0[0],
        fc3_weights_0: fc3_weights_0[0],

        //multiplier for quantization
        multiplier_conv11: multiplier_conv11.clone(),
        multiplier_conv12: multiplier_conv12.clone(),
        multiplier_conv21: multiplier_conv21.clone(),
        multiplier_conv22: multiplier_conv22.clone(),
        
        multiplier_conv31: multiplier_conv31.clone(),
        multiplier_conv32: multiplier_conv32.clone(),
        multiplier_conv33: multiplier_conv33.clone(),
        
        multiplier_conv41: multiplier_conv41.clone(),
        multiplier_conv42: multiplier_conv42.clone(),
        multiplier_conv43: multiplier_conv43.clone(),
        

        multiplier_conv51: multiplier_conv51.clone(),
        multiplier_conv52: multiplier_conv52.clone(),
        multiplier_conv53: multiplier_conv53.clone(),
        

        multiplier_fc1: multiplier_fc1.clone(),
        multiplier_fc2: multiplier_fc2.clone(),
        multiplier_fc3: multiplier_fc3.clone(),

        z: z.clone(),
        z_open: z_open,
        z_com: z_com,
        knit_encoding: true,

    };

    println!("start generating random parameters");
    let begin = Instant::now();

    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let end = Instant::now();

    println!("setup time {:?}", end.duration_since(begin));

    // let mut buf = vec![];
    // param.serialize(&mut buf).unwrap();
    // println!("crs size: {}", buf.len());

    let pvk = prepare_verifying_key(&param.vk);
    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));

    // let commitment = [x_com.x, x_com.y, z_com.x, z_com.y].to_vec();

    // let inputs: Vec<Fq> = [
    //     commitment[..].as_ref(),
    // ]
    // .concat();

    // let begin = Instant::now();
    // assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());
    // let end = Instant::now();
    // println!("verification time {:?}", end.duration_since(begin));
}