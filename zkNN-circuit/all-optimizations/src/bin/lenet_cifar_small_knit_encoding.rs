use algebra::ed_on_bls12_381::*;
use algebra::CanonicalSerialize;
use algebra::UniformRand;
use crypto_primitives::commitment::pedersen::Randomness;
use groth16::*;
use r1cs_core::*;
use std::time::Instant;
use zk_ml_knit_encoding::lenet_circuit::*;
use zk_ml_knit_encoding::pedersen_commit::*;
use zk_ml_knit_encoding::read_inputs::*;
use zk_ml_knit_encoding::vanilla::*;

fn main() {
    let mut rng = rand::thread_rng();

    println!("LeNet optimized small on CIFAR dataset");
    let x: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/X_q.txt".to_string(),
        1,
        3,
        32,
        32,
    ); // only read one image
    let conv1_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_weight_q.txt".to_string(),
        6,
        3,
        5,
        5,
    );
    let conv2_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_weight_q.txt".to_string(),
        16,
        6,
        5,
        5,
    );
    let conv3_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_weight_q.txt".to_string(),
        120,
        16,
        4,
        4,
    );
    let fc1_w: Vec<Vec<u8>> = read_vector2d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_weight_q.txt".to_string(),
        84,
        480,
    );
    let fc2_w: Vec<Vec<u8>> = read_vector2d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_weight_q.txt".to_string(),
        10,
        84,
    );

    let x_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/X_z.txt".to_string(),
        1,
    );
    let conv1_output_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_output_z.txt".to_string(),
        1,
    );
    let conv2_output_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_output_z.txt".to_string(),
        1,
    );
    let conv3_output_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_output_z.txt".to_string(),
        1,
    );
    let fc1_output_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_output_z.txt".to_string(),
        1,
    );
    let fc2_output_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_output_z.txt".to_string(),
        1,
    );

    let conv1_weights_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_weight_z.txt".to_string(),
        1,
    );
    let conv2_weights_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_weight_z.txt".to_string(),
        1,
    );
    let conv3_weights_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_weight_z.txt".to_string(),
        1,
    );
    let fc1_weights_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_weight_z.txt".to_string(),
        1,
    );
    let fc2_weights_0: Vec<u8> = read_vector1d(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_weight_z.txt".to_string(),
        1,
    );

    let multiplier_conv1: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_weight_s.txt".to_string(),
        6,
    );
    let multiplier_conv2: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_weight_s.txt".to_string(),
        16,
    );
    let multiplier_conv3: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_weight_s.txt".to_string(),
        120,
    );

    let multiplier_fc1: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_weight_s.txt".to_string(),
        84,
    );
    let multiplier_fc2: Vec<f32> = read_vector1d_f32(
        "../pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_weight_s.txt".to_string(),
        10,
    );

    println!("finish reading parameters");

    let z: Vec<Vec<u8>> = lenet_circuit_forward_u8(
        x.clone(),
        conv1_w.clone(),
        conv2_w.clone(),
        conv3_w.clone(),
        fc1_w.clone(),
        fc2_w.clone(),
        x_0[0],
        conv1_output_0[0],
        conv2_output_0[0],
        conv3_output_0[0],
        fc1_output_0[0],
        fc2_output_0[0], // which is also lenet output(z) zero point
        conv1_weights_0[0],
        conv2_weights_0[0],
        conv3_weights_0[0],
        fc1_weights_0[0],
        fc2_weights_0[0],
        multiplier_conv1.clone(),
        multiplier_conv2.clone(),
        multiplier_conv3.clone(),
        multiplier_fc1.clone(),
        multiplier_fc2.clone(),
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
    let classification_res = argmax_u8(z[0].clone());

    let full_circuit = LeNetCircuitU8OptimizedLv2PedersenClassification {
        params: param.clone(),
        x: x.clone(),
        x_com: x_com.clone(),
        x_open: x_open,

        conv1_weights: conv1_w.clone(),
        conv2_weights: conv2_w.clone(),
        conv3_weights: conv3_w.clone(),
        fc1_weights: fc1_w.clone(),
        fc2_weights: fc2_w.clone(),

        //zero points for quantization.
        x_0: x_0[0],
        conv1_output_0: conv1_output_0[0],
        conv2_output_0: conv2_output_0[0],
        conv3_output_0: conv3_output_0[0],
        fc1_output_0: fc1_output_0[0],
        fc2_output_0: fc2_output_0[0], // which is also lenet output(z) zero point

        conv1_weights_0: conv1_weights_0[0],
        conv2_weights_0: conv2_weights_0[0],
        conv3_weights_0: conv3_weights_0[0],
        fc1_weights_0: fc1_weights_0[0],
        fc2_weights_0: fc2_weights_0[0],

        //multiplier for quantization
        multiplier_conv1: multiplier_conv1.clone(),
        multiplier_conv2: multiplier_conv2.clone(),
        multiplier_conv3: multiplier_conv3.clone(),
        multiplier_fc1: multiplier_fc1.clone(),
        multiplier_fc2: multiplier_fc2.clone(),

        z: z.clone(),
        z_open: z_open,
        z_com: z_com,
        argmax_res: classification_res,
        knit_encoding: true,

    };

    // sanity checks
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
