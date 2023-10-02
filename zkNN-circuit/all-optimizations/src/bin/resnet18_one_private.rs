use algebra::ed_on_bls12_381::*;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::time::Instant;
use zk_ml_knit_encoding::pedersen_commit::*;
use zk_ml_knit_encoding::read_inputs::*;
use zk_ml_knit_encoding::vanilla::*;
use zk_ml_knit_encoding::resnet18_circuit::*;

fn rand_4d_vec_generator(a: usize, b: usize, c: usize, d: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    let mut res = vec![vec![vec![vec![0; d]; c]; b]; a];

    res
}

fn rand_2d_vec_generator(a: usize, b: usize) -> Vec<Vec<u8>> {
    let mut res = vec![vec![2; b]; a];

    res
}

fn main() {
    let mut rng = rand::thread_rng();

    println!("RESNET18 testing. use null parameters and input just to benchmark performance");

    let x = rand_4d_vec_generator(1, 3, 32, 32);


    let conv21_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 3, 3, 3);
    let conv22_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 16, 3, 3);
    let conv23_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 16, 3, 3);
    let conv24_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 16, 3, 3);
    let multiplier_conv21: Vec<f32> = vec![1.5; 16];
    let multiplier_conv22: Vec<f32> = vec![1.5; 16];
    let multiplier_conv23: Vec<f32> = vec![1.5; 16];
    let multiplier_conv24: Vec<f32> = vec![1.5; 16];

    let conv31_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 16, 3, 3);
    let conv32_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 32, 3, 3);
    let conv33_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 32, 3, 3);
    let conv34_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 32, 3, 3);
    let multiplier_conv31: Vec<f32> = vec![1.5; 32];
    let multiplier_conv32: Vec<f32> = vec![1.5; 32];
    let multiplier_conv33: Vec<f32> = vec![1.5; 32];
    let multiplier_conv34: Vec<f32> = vec![1.5; 32];

    let conv41_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 32, 3, 3);
    let conv42_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 64, 3, 3);
    let conv43_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 64, 3, 3);
    let conv44_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 64, 3, 3);
    let multiplier_conv41: Vec<f32> = vec![1.5; 64];
    let multiplier_conv42: Vec<f32> = vec![1.5; 64];
    let multiplier_conv43: Vec<f32> = vec![1.5; 64];
    let multiplier_conv44: Vec<f32> = vec![1.5; 64];

    let conv51_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 64, 3, 3);
    let conv52_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 128, 3, 3);
    let conv53_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 128, 3, 3);
    let conv54_w: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 128, 3, 3);
    let multiplier_conv51: Vec<f32> = vec![1.5; 128];
    let multiplier_conv52: Vec<f32> = vec![1.5; 128];
    let multiplier_conv53: Vec<f32> = vec![1.5; 128];
    let multiplier_conv54: Vec<f32> = vec![1.5; 128];

    // let fc1_w: Vec<Vec<u8>> = rand_2d_vec_generator(1024, 1024);
    let fc1_w: Vec<Vec<u8>> = rand_2d_vec_generator(10, 128);
    // let multiplier_fc1: Vec<f32> = vec![1.5; 1024];
    let multiplier_fc1: Vec<f32> = vec![1.5; 128];


    let x_0: Vec<u8> = vec![128];

    let conv21_output_0: Vec<u8> = vec![128];
    let conv22_output_0: Vec<u8> = vec![128];
    let conv23_output_0: Vec<u8> = vec![128];
    let conv24_output_0: Vec<u8> = vec![128];

    let conv31_output_0: Vec<u8> = vec![128];
    let conv32_output_0: Vec<u8> = vec![128];
    let conv33_output_0: Vec<u8> = vec![128];
    let conv34_output_0: Vec<u8> = vec![128];

    let conv41_output_0: Vec<u8> = vec![128];
    let conv42_output_0: Vec<u8> = vec![128];
    let conv43_output_0: Vec<u8> = vec![128];
    let conv44_output_0: Vec<u8> = vec![128];

    let conv51_output_0: Vec<u8> = vec![128];
    let conv52_output_0: Vec<u8> = vec![128];
    let conv53_output_0: Vec<u8> = vec![128];
    let conv54_output_0: Vec<u8> = vec![128];

    let fc1_output_0: Vec<u8> = vec![128];


    let conv21_weights_0: Vec<u8> = vec![128];
    let conv22_weights_0: Vec<u8> = vec![128];
    let conv23_weights_0: Vec<u8> = vec![128];
    let conv24_weights_0: Vec<u8> = vec![128];

    let conv31_weights_0: Vec<u8> = vec![128];
    let conv32_weights_0: Vec<u8> = vec![128];
    let conv33_weights_0: Vec<u8> = vec![128];
    let conv34_weights_0: Vec<u8> = vec![128];

    let conv41_weights_0: Vec<u8> = vec![128];
    let conv42_weights_0: Vec<u8> = vec![128];
    let conv43_weights_0: Vec<u8> = vec![128];
    let conv44_weights_0: Vec<u8> = vec![128];

    let conv51_weights_0: Vec<u8> = vec![128];
    let conv52_weights_0: Vec<u8> = vec![128];
    let conv53_weights_0: Vec<u8> = vec![128];
    let conv54_weights_0: Vec<u8> = vec![128];

    let fc1_weights_0: Vec<u8> = vec![128];


    println!("finish reading parameters");

    let conv_residual1_weight: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(16, 3, 1, 1);
    let conv_residual1_output_0:Vec<u8> = vec![128];
    let conv_residual1_weights_0: Vec<u8> = vec![128];
    let conv_residual1_multiplier: Vec<f32> = vec![1.5; 16];

    let conv_residual3_weight: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(32, 16, 1, 1);
    let conv_residual3_output_0:Vec<u8> = vec![128];
    let conv_residual3_weights_0: Vec<u8> = vec![128];
    let conv_residual3_multiplier: Vec<f32> = vec![1.5; 32];

    let conv_residual5_weight: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(64, 32, 1, 1);
    let conv_residual5_output_0:Vec<u8> = vec![128];
    let conv_residual5_weights_0: Vec<u8> = vec![128];
    let conv_residual5_multiplier: Vec<f32> = vec![1.5; 64];

    let conv_residual7_weight: Vec<Vec<Vec<Vec<u8>>>> = rand_4d_vec_generator(128, 64, 1, 1);
    let conv_residual7_output_0:Vec<u8> = vec![128];
    let conv_residual7_weights_0: Vec<u8> = vec![128];
    let conv_residual7_multiplier: Vec<f32> = vec![1.5; 128];

    let add_residual1_output_0:Vec<u8> = vec![128];
    let add_residual2_output_0:Vec<u8> = vec![128];
    let add_residual3_output_0:Vec<u8> = vec![128];
    let add_residual4_output_0:Vec<u8> = vec![128];
    let add_residual5_output_0:Vec<u8> = vec![128];
    let add_residual6_output_0:Vec<u8> = vec![128];
    let add_residual7_output_0:Vec<u8> = vec![128];
    let add_residual8_output_0:Vec<u8> = vec![128];

    let add_residual1_first_multiplier: Vec<f32> = vec![1.5; 32];
    let add_residual2_first_multiplier: Vec<f32> = vec![1.5; 32];
    let add_residual3_first_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual4_first_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual5_first_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual6_first_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual7_first_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual8_first_multiplier: Vec<f32> = vec![1.5; 256];

    let add_residual1_second_multiplier: Vec<f32> = vec![1.5; 32];
    let add_residual2_second_multiplier: Vec<f32> = vec![1.5; 32];
    let add_residual3_second_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual4_second_multiplier: Vec<f32> = vec![1.5; 64];
    let add_residual5_second_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual6_second_multiplier: Vec<f32> = vec![1.5; 128];
    let add_residual7_second_multiplier: Vec<f32> = vec![1.5; 256];
    let add_residual8_second_multiplier: Vec<f32> = vec![1.5; 256];
    
    println!("before forward");

    let z: Vec<Vec<u8>> = resnet18_circuit_forward_u8(
        1, //padding
        x.clone(),

        conv21_w.clone(),
        conv22_w.clone(),
        conv23_w.clone(),
        conv24_w.clone(),

        conv31_w.clone(),
        conv32_w.clone(),
        conv33_w.clone(),
        conv34_w.clone(),

        conv41_w.clone(),
        conv42_w.clone(),
        conv43_w.clone(),
        conv44_w.clone(),

        conv51_w.clone(),
        conv52_w.clone(),
        conv53_w.clone(),
        conv54_w.clone(),

        conv_residual1_weight.clone(),  //residual kernel 1
        conv_residual3_weight.clone(),  //residual kernel 1
        conv_residual5_weight.clone(),  //residual kernel 1
        conv_residual7_weight.clone(),  //residual kernel 1

        fc1_w.clone(),

        x_0[0],

        conv21_output_0[0],
        conv22_output_0[0],
        conv23_output_0[0],
        conv24_output_0[0],

        conv31_output_0[0],
        conv32_output_0[0],
        conv33_output_0[0],
        conv34_output_0[0],

        conv41_output_0[0],
        conv42_output_0[0],
        conv43_output_0[0],
        conv44_output_0[0],


        conv51_output_0[0],
        conv52_output_0[0],
        conv53_output_0[0],
        conv54_output_0[0],

        conv_residual1_output_0[0], //residual output_0 1
        conv_residual3_output_0[0], //residual output_0 1
        conv_residual5_output_0[0], //residual output_0 1
        conv_residual7_output_0[0], //residual output_0 1

        fc1_output_0[0],

        conv21_weights_0[0],
        conv22_weights_0[0],
        conv23_weights_0[0],
        conv24_weights_0[0],

        conv31_weights_0[0],
        conv32_weights_0[0],
        conv33_weights_0[0],
        conv34_weights_0[0],

        conv41_weights_0[0],
        conv42_weights_0[0],
        conv43_weights_0[0],
        conv44_weights_0[0],

        conv51_weights_0[0],
        conv52_weights_0[0],
        conv53_weights_0[0],
        conv54_weights_0[0],

        conv_residual1_weights_0[0], //residual weights_0 1
        conv_residual3_weights_0[0], //residual weights_0 1
        conv_residual5_weights_0[0], //residual weights_0 1
        conv_residual7_weights_0[0], //residual weights_0 1

        fc1_weights_0[0],

        multiplier_conv21.clone(),
        multiplier_conv22.clone(),
        multiplier_conv23.clone(),
        multiplier_conv24.clone(),

        multiplier_conv31.clone(),
        multiplier_conv32.clone(),
        multiplier_conv33.clone(),
        multiplier_conv34.clone(),

        multiplier_conv41.clone(),
        multiplier_conv42.clone(),
        multiplier_conv43.clone(),
        multiplier_conv44.clone(),

        multiplier_conv51.clone(),
        multiplier_conv52.clone(),
        multiplier_conv53.clone(),
        multiplier_conv54.clone(),

        conv_residual1_multiplier.clone(),  //residual multiplier_0 1
        conv_residual3_multiplier.clone(),  //residual multiplier_0 1
        conv_residual5_multiplier.clone(),  //residual multiplier_0 1
        conv_residual7_multiplier.clone(),  //residual multiplier_0 1

        add_residual1_output_0[0],
        add_residual2_output_0[0],
        add_residual3_output_0[0],
        add_residual4_output_0[0],
        add_residual5_output_0[0],
        add_residual6_output_0[0],
        add_residual7_output_0[0],
        add_residual8_output_0[0],

        add_residual1_first_multiplier.clone(),
        add_residual2_first_multiplier.clone(),
        add_residual3_first_multiplier.clone(),
        add_residual4_first_multiplier.clone(),
        add_residual5_first_multiplier.clone(),
        add_residual6_first_multiplier.clone(),
        add_residual7_first_multiplier.clone(),
        add_residual8_first_multiplier.clone(),

        add_residual1_second_multiplier.clone(),
        add_residual2_second_multiplier.clone(),
        add_residual3_second_multiplier.clone(),
        add_residual4_second_multiplier.clone(),
        add_residual5_second_multiplier.clone(),
        add_residual6_second_multiplier.clone(),
        add_residual7_second_multiplier.clone(),
        add_residual8_second_multiplier.clone(),

        multiplier_fc1.clone(),
    );
    
    println!("finish forwarding");

    //batch size is only one for faster calculation of total constraints
    let flattened_x3d: Vec<Vec<Vec<u8>>> = x.clone().into_iter().flatten().collect();
    let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
    let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();

    let flattened_z1d: Vec<u8> = z.clone().into_iter().flatten().collect();

    //println!("x outside {:?}", x.clone());
    // println!("z outside {:?}", flattened_z1d.clone());
    let begin = Instant::now();
    let param = setup(&[0; 32]);
    let x_open = Randomness(Fr::rand(&mut rng));
    let x_com = pedersen_commit(&flattened_x1d, &param, &x_open);

    println!("finish pedersen");

    let z_open = Randomness(Fr::rand(&mut rng));
    let z_com = pedersen_commit(&flattened_z1d, &param, &z_open);
    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));
    //we only do one image in zk proof.

    let full_circuit = Resnet18CircuitU8OptimizedLv2PedersenPublicNNWeights {
        padding:1,
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,


        conv21_weights: conv21_w.clone(),
        conv22_weights: conv22_w.clone(),
        conv23_weights: conv23_w.clone(),
        conv24_weights: conv24_w.clone(),

        conv31_weights: conv31_w.clone(),
        conv32_weights: conv32_w.clone(),
        conv33_weights: conv33_w.clone(),
        conv34_weights: conv34_w.clone(),

        conv41_weights: conv41_w.clone(),
        conv42_weights: conv42_w.clone(),
        conv43_weights: conv43_w.clone(),
        conv44_weights: conv44_w.clone(),

        conv51_weights: conv51_w.clone(),
        conv52_weights: conv52_w.clone(),
        conv53_weights: conv53_w.clone(),
        conv54_weights: conv54_w.clone(),

        conv_residual1_weight: conv_residual1_weight.clone(),  //residual kernel 1
        conv_residual3_weight: conv_residual3_weight.clone(),  //residual kernel 1
        conv_residual5_weight: conv_residual5_weight.clone(),  //residual kernel 1
        conv_residual7_weight: conv_residual7_weight.clone(),  //residual kernel 1

        fc1_weights: fc1_w.clone(),


        //zero points for quantization.
        x_0: x_0[0],

        conv21_output_0: conv21_output_0[0],
        conv22_output_0: conv22_output_0[0],
        conv23_output_0: conv23_output_0[0],
        conv24_output_0: conv24_output_0[0],

        conv31_output_0: conv31_output_0[0],
        conv32_output_0: conv32_output_0[0],
        conv33_output_0: conv33_output_0[0],
        conv34_output_0: conv34_output_0[0],

        conv41_output_0: conv41_output_0[0],
        conv42_output_0: conv42_output_0[0],
        conv43_output_0: conv43_output_0[0],
        conv44_output_0: conv44_output_0[0],

        conv51_output_0: conv51_output_0[0],
        conv52_output_0: conv52_output_0[0],
        conv53_output_0: conv53_output_0[0],
        conv54_output_0: conv54_output_0[0],

        conv_residual1_output_0: conv_residual1_output_0[0], //residual output_0 1
        conv_residual3_output_0: conv_residual3_output_0[0], //residual output_0 1
        conv_residual5_output_0: conv_residual5_output_0[0], //residual output_0 1
        conv_residual7_output_0: conv_residual7_output_0[0], //residual output_0 1

        fc1_output_0: fc1_output_0[0],

        conv21_weights_0: conv21_weights_0[0],
        conv22_weights_0: conv22_weights_0[0],
        conv23_weights_0: conv23_weights_0[0],
        conv24_weights_0: conv24_weights_0[0],

        conv31_weights_0: conv31_weights_0[0],
        conv32_weights_0: conv32_weights_0[0],
        conv33_weights_0: conv33_weights_0[0],
        conv34_weights_0: conv34_weights_0[0],

        conv41_weights_0: conv41_weights_0[0],
        conv42_weights_0: conv42_weights_0[0],
        conv43_weights_0: conv43_weights_0[0],
        conv44_weights_0: conv44_weights_0[0],

        conv51_weights_0: conv51_weights_0[0],
        conv52_weights_0: conv52_weights_0[0],
        conv53_weights_0: conv53_weights_0[0],
        conv54_weights_0: conv54_weights_0[0],

        conv_residual1_weights_0: conv_residual1_weights_0[0], //residual weights_0 1
        conv_residual3_weights_0: conv_residual3_weights_0[0], //residual weights_0 1
        conv_residual5_weights_0: conv_residual5_weights_0[0], //residual weights_0 1
        conv_residual7_weights_0: conv_residual7_weights_0[0], //residual weights_0 1

        fc1_weights_0: fc1_weights_0[0],


        //multiplier for quantization

        conv21_multiplier: multiplier_conv21.clone(),
        conv22_multiplier: multiplier_conv22.clone(),
        conv23_multiplier: multiplier_conv23.clone(),
        conv24_multiplier: multiplier_conv24.clone(),

        conv31_multiplier: multiplier_conv31.clone(),
        conv32_multiplier: multiplier_conv32.clone(),
        conv33_multiplier: multiplier_conv33.clone(),
        conv34_multiplier: multiplier_conv34.clone(),

        conv41_multiplier: multiplier_conv41.clone(),
        conv42_multiplier: multiplier_conv42.clone(),
        conv43_multiplier: multiplier_conv43.clone(),
        conv44_multiplier: multiplier_conv44.clone(),

        conv51_multiplier: multiplier_conv51.clone(),
        conv52_multiplier: multiplier_conv52.clone(),
        conv53_multiplier: multiplier_conv53.clone(),
        conv54_multiplier: multiplier_conv54.clone(),

        conv_residual1_multiplier: conv_residual1_multiplier.clone(),  //residual multiplier_0 1
        conv_residual3_multiplier: conv_residual3_multiplier.clone(),  //residual multiplier_0 1
        conv_residual5_multiplier: conv_residual5_multiplier.clone(),  //residual multiplier_0 1
        conv_residual7_multiplier: conv_residual7_multiplier.clone(),  //residual multiplier_0 1

        add_residual1_output_0: add_residual1_output_0[0],
        add_residual2_output_0: add_residual2_output_0[0],
        add_residual3_output_0: add_residual3_output_0[0],
        add_residual4_output_0: add_residual4_output_0[0],
        add_residual5_output_0: add_residual5_output_0[0],
        add_residual6_output_0: add_residual6_output_0[0],
        add_residual7_output_0: add_residual7_output_0[0],
        add_residual8_output_0: add_residual8_output_0[0],

        add_residual1_first_multiplier: add_residual1_first_multiplier.clone(),
        add_residual2_first_multiplier: add_residual2_first_multiplier.clone(),
        add_residual3_first_multiplier: add_residual3_first_multiplier.clone(),
        add_residual4_first_multiplier: add_residual4_first_multiplier.clone(),
        add_residual5_first_multiplier: add_residual5_first_multiplier.clone(),
        add_residual6_first_multiplier: add_residual6_first_multiplier.clone(),
        add_residual7_first_multiplier: add_residual7_first_multiplier.clone(),
        add_residual8_first_multiplier: add_residual8_first_multiplier.clone(),

        add_residual1_second_multiplier: add_residual1_second_multiplier.clone(),
        add_residual2_second_multiplier: add_residual2_second_multiplier.clone(),
        add_residual3_second_multiplier: add_residual3_second_multiplier.clone(),
        add_residual4_second_multiplier: add_residual4_second_multiplier.clone(),
        add_residual5_second_multiplier: add_residual5_second_multiplier.clone(),
        add_residual6_second_multiplier: add_residual6_second_multiplier.clone(),
        add_residual7_second_multiplier: add_residual7_second_multiplier.clone(),
        add_residual8_second_multiplier: add_residual8_second_multiplier.clone(),

        multiplier_fc1: multiplier_fc1.clone(),


        z: z.clone(),
        z_open: z_open,
        z_com: z_com,
        knit_encoding: true,
    };

    // // sanity checks
    // {
    //     let sanity_cs = ConstraintSystem::<Fq>::new_ref();
    //     full_circuit
    //         .clone()
    //         .generate_constraints(sanity_cs.clone())
    //         .unwrap();

    //     let res = sanity_cs.is_satisfied().unwrap();
    //     println!("are the constraints satisfied?: {}\n", res);

    //     if !res {
    //         println!(
    //             "{:?} {} {:#?}",
    //             sanity_cs.constraint_names(),
    //             sanity_cs.num_constraints(),
    //             sanity_cs.which_is_unsatisfied().unwrap()
    //         );
    //     }
    // }
    println!("start generating random parameters");
    let begin = Instant::now();

    // pre-computed parameters
    let param =
        generate_random_parameters::<algebra::Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let end = Instant::now();

    println!("setup time {:?}", end.duration_since(begin));

    let mut buf = vec![];
    param.serialize(&mut buf).unwrap();
    println!("crs size: {}", buf.len());

    let pvk = prepare_verifying_key(&param.vk);
    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));

    let commitment = [x_com.x, x_com.y, z_com.x, z_com.y].to_vec();

    let inputs: Vec<Fq> = [
        commitment[..].as_ref(),
    ]
    .concat();

    let begin = Instant::now();
    assert!(verify_proof(&pvk, &proof, &inputs[..]).unwrap());
    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}
