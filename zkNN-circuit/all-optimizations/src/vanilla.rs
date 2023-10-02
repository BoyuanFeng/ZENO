use crate::*;
use crypto_primitives::commitment::blake2s::Commitment;
use crypto_primitives::CommitmentScheme;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub const DEFAULT_ZERO_POINT: u8 = 10;

#[allow(non_snake_case)]
pub(crate) fn relu_u8(input: &mut [u8], zero_point: u8) -> Vec<bool> {
    let mut cmp_res: Vec<bool> = Vec::new();
    for e in input {
        if *e < zero_point {
            *e = zero_point;
            cmp_res.push(false);
        } else {
            cmp_res.push(true);
        }
    }
    cmp_res
}

#[allow(non_snake_case)]
pub(crate) fn relu2d_u8(input: &mut Vec<Vec<u8>>, zero_point: u8) {
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            if input[i][j] < zero_point {
                input[i][j] = zero_point;
            }
        }
    }
}

pub(crate) fn relu4d_u8(
    input: &mut Vec<Vec<Vec<Vec<u8>>>>,
    zero_point: u8,
) -> Vec<Vec<Vec<Vec<bool>>>> {
    let mut cmp_res: Vec<Vec<Vec<Vec<bool>>>> =
        vec![
            vec![vec![vec![false; input[0][0][0].len()]; input[0][0].len()]; input[0].len()];
            input.len()
        ];
    for i in 0..input.len() {
        for j in 0..input[i].len() {
            for k in 0..input[i][j].len() {
                for l in 0..input[i][j][k].len() {
                    if input[i][j][k][l] < zero_point {
                        input[i][j][k][l] = zero_point;
                        cmp_res[i][j][k][l] = false;
                    } else {
                        cmp_res[i][j][k][l] = true;
                    }
                }
            }
        }
    }

    cmp_res
}

pub(crate) fn scalar_with_remainder_u8(a: &[u8], b: &[u8], a_0: u8, b_0: u8, y_0: u64) -> u64 {
    if a.len() != b.len() {
        panic!("incorrect dim {} {}", a.len(), b.len());
    }
    let mut tmp1: u64 = 0;
    let mut tmp2: u64 = 0;
    let mut tmp3: u64 = 0;
    let mut tmp4: u64 = 0;
    for i in 0..a.len() {
        tmp1 += a[i] as u64 * b[i] as u64;
        tmp2 += a[i] as u64 * b_0 as u64;
        tmp3 += a_0 as u64 * b[i] as u64;
        tmp4 += a_0 as u64 * b_0 as u64;
    }

    let res = (tmp1 + tmp4 + y_0) - (tmp2 + tmp3);

    res
}

//return the remainder of divided by 2^24
pub(crate) fn vec_mat_mul_with_remainder_u8(
    vec: &[u8],
    mat: &[&[u8]],
    res: &mut [u8],
    vec_0: u8,
    mat_0: u8,
    res_0: u8,
    multiplier: &[f32],
) -> (Vec<u32>, Vec<u32>) {
    //record info loss during u64/u32 to u8 for later recovery
    //println!("q1 before shift {:?}", mat[0].clone());
    let mut remainder = vec![0u32; res.len()];
    let mut div_res = vec![0u32; res.len()];
    for i in 0..mat.len() {
        let m = (multiplier[i] * 2u64.pow(22) as f32) as u64;
        let res_converted = (res_0 as u64 * 2u64.pow(22)) / m;
        let scalar_tmp =
            m * scalar_with_remainder_u8(vec.clone(), mat[i], vec_0, mat_0, res_converted);
        remainder[i] = (scalar_tmp % 2u64.pow(22)) as u32;
        div_res[i] = (scalar_tmp / 2u64.pow(22 + 8)) as u32;
        res[i] = (scalar_tmp / 2u64.pow(22)) as u8;
    }

    //println!("res {:?}", res);
    (remainder, div_res)
}

fn conv_kernel_scala_with_remainder_u8(
    x: &Vec<Vec<Vec<u8>>>,
    kernel: &Vec<Vec<Vec<u8>>>,
    h_index: usize,
    w_index: usize,

    x_zeropoint: u8,
    kernel_zeropoint: u8,
    y_0: u64,
) -> u64 {
    let num_channels = kernel.len();
    let kernel_size = kernel[0].len();
    let mut tmp1: u64 = 0;
    let mut tmp2: u64 = 0;
    let mut tmp3: u64 = 0;
    let mut tmp4: u64 = 0;
    //println!("multiplier : {}\n y_converted : {}", m, y_converted);
    for i in 0..num_channels {
        //iterate through all channels

        for j in h_index..(h_index + kernel_size) {
            // println!("data {:?}", &x[i][j][w_index..w_index+kernel_size]);
            // println!("kernel {:?}", &kernel[i][j][0..kernel_size]);
            for k in w_index..(w_index + kernel_size) {
                //println!("i,j,k {} {} {}",i, j - h_index, k - w_index);
                tmp1 += x[i][j][k] as u64 * kernel[i][j - h_index][k - w_index] as u64;

                tmp2 += x[i][j][k] as u64 * kernel_zeropoint as u64;
                tmp3 += kernel[i][j - h_index][k - w_index] as u64 * x_zeropoint as u64;

                tmp4 += x_zeropoint as u64 * kernel_zeropoint as u64;
            }
        }
    }
    //println!("conv output {}  {} ", tmp1 *m +  tmp4*m, tmp2*m + tmp3*m,);
    //assert_eq!(tmp1, tmp2);

    //println!("tmp14 {}\ntmp23{}", tmp1+ tmp4, tmp2+ tmp3);
    let res = (tmp1 + tmp4 + y_0) - (tmp2 + tmp3);

    res
}

pub(crate) fn vec_conv_with_remainder_u8(
    vec: &Vec<Vec<Vec<Vec<u8>>>>,
    kernel: &Vec<Vec<Vec<Vec<u8>>>>,
    res: &mut Vec<Vec<Vec<Vec<u8>>>>,
    vec_0: u8,
    kernel_0: u8,
    res_0: u8,
    multiplier: &[f32],
) -> (Vec<Vec<Vec<Vec<u32>>>>, Vec<Vec<Vec<Vec<u32>>>>) {
    let num_kernels = kernel.len();
    let kernel_size = kernel[0][0].len();
    let batch_size = vec.len();
    let input_height = vec[0][0].len();
    let input_width = vec[0][0][0].len();
    //println!("kernel {:?}", kernel.clone());
    //record info loss during u64/u32 to u8 for later recovery
    let mut remainder =
        vec![vec![vec![vec![0u32; res[0][0][0].len()]; res[0][0].len()]; res[0].len()]; res.len()];
    let mut div =
        vec![vec![vec![vec![0u32; res[0][0][0].len()]; res[0][0].len()]; res[0].len()]; res.len()];
    for n in 0..batch_size {
        for h in 0..(input_height - kernel_size + 1) {
            for w in 0..(input_width - kernel_size + 1) {
                for k in 0..num_kernels {
                    // println!("{} {} {} {}",n, k, h, w);
                    let m = (multiplier[k] * 2u64.pow(22) as f32) as u64;
                    let res_converted = (res_0 as u64 * 2u64.pow(22)) / m;
                    let tmp = m * conv_kernel_scala_with_remainder_u8(
                        &vec[n],
                        &kernel[k],
                        h,
                        w,
                        vec_0,
                        kernel_0,
                        res_converted,
                    );

                    res[n][k][h][w] = (tmp / 2u64.pow(22)) as u8;

                    remainder[n][k][h][w] = (tmp % 2u64.pow(22)) as u32;
                    div[n][k][h][w] = (tmp / 2u64.pow(30)) as u32;
                }
            }
        }
    }
    //println!("kernel shape ({},{},{},{})", K,C,kernel_size,kernel_size);
    (remainder, div)
}

pub(crate) fn avg_pool_with_remainder_helper_u8(
    input: &Vec<Vec<u8>>,
    h_start: usize,
    w_start: usize,
    kernel_size: usize,
) -> (u8, u8) {
    let mut res: u32 = 0;

    for i in h_start..(h_start + kernel_size) {
        for j in w_start..(w_start + kernel_size) {
            res += input[i][j] as u32;
        }
    }

    (
        (res / (kernel_size as u32 * kernel_size as u32)) as u8,
        (res % (kernel_size as u32 * kernel_size as u32)) as u8,
    )
}

pub(crate) fn avg_pool_helper_u8(
    input: &Vec<Vec<u8>>,
    h_start: usize,
    w_start: usize,
    kernel_size: usize,
) -> u8 {
    let mut res: u32 = 0;

    for i in h_start..(h_start + kernel_size) {
        for j in w_start..(w_start + kernel_size) {
            res += input[i][j] as u32;
        }
    }

    (res / (kernel_size as u32 * kernel_size as u32)) as u8
}
pub(crate) fn avg_pool_scala_u8(
    vec: &Vec<Vec<Vec<Vec<u8>>>>,
    kernel_size: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    let batch_size = vec.len();
    let num_channels = vec[0].len(); //num of channels
    let input_height = vec[0][0].len(); // height of image
    let input_width = vec[0][0][0].len(); // width of image
    let mut output = vec![
        vec![
            vec![vec![0u8; input_width / kernel_size]; input_height / kernel_size];
            num_channels
        ];
        batch_size
    ];
    for n in 0..batch_size {
        for c in 0..num_channels {
            for h in 0..(input_height / kernel_size) {
                for w in 0..(input_width / kernel_size) {
                    output[n][c][h][w] = avg_pool_helper_u8(
                        &vec[n][c],
                        kernel_size * h,
                        kernel_size * w,
                        kernel_size,
                    );
                }
            }
        }
    }
    output
}

pub(crate) fn avg_pool_with_remainder_scala_u8(
    vec: &Vec<Vec<Vec<Vec<u8>>>>,
    kernel_size: usize,
) -> (Vec<Vec<Vec<Vec<u8>>>>, Vec<Vec<Vec<Vec<u8>>>>) {
    let batch_size = vec.len();
    let num_channels = vec[0].len(); //num of channels
    let input_height = vec[0][0].len(); // height of image
    let input_width = vec[0][0][0].len(); // width of image
    let mut output =
        vec![vec![vec![vec![0u8; input_width / kernel_size]; input_height / kernel_size]; num_channels]; batch_size];
    let mut remainder =
        vec![vec![vec![vec![0u8; input_width]; input_height]; num_channels]; batch_size];

    for n in 0..batch_size {
        for c in 0..num_channels {
            for h in 0..(input_height / kernel_size) {
                for w in 0..(input_width / kernel_size) {
                    let (res, remained) = avg_pool_with_remainder_helper_u8(
                        &vec[n][c],
                        kernel_size * h,
                        kernel_size * w,
                        kernel_size,
                    );
                    output[n][c][h][w] = res;
                    remainder[n][c][h][w] = remained;
                }
            }
        }
    }
    (output, remainder)
}

pub fn lenet_circuit_forward_u8(
    x: Vec<Vec<Vec<Vec<u8>>>>,
    conv1_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv2_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv3_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    fc1_weight: Vec<Vec<u8>>,
    fc2_weight: Vec<Vec<u8>>,
    x_0: u8,
    conv1_output_0: u8,
    conv2_output_0: u8,
    conv3_output_0: u8,
    fc1_output_0: u8,
    fc2_output_0: u8,
    conv1_weights_0: u8,
    conv2_weights_0: u8,
    conv3_weights_0: u8,
    fc1_weights_0: u8,
    fc2_weights_0: u8,
    multiplier_conv1: Vec<f32>,
    multiplier_conv2: Vec<f32>,
    multiplier_conv3: Vec<f32>,
    multiplier_fc1: Vec<f32>,
    multiplier_fc2: Vec<f32>,
) -> Vec<Vec<u8>> {
    println!("lenet vallina forward");
    //layer 1
    let mut conv1_output = vec![vec![vec![vec![0u8; x[0][0][0].len() - conv1_kernel[0][0][0].len() + 1];  // w - kernel_size + 1
                                        x[0][0].len() - conv1_kernel[0][0].len() + 1]; // h - kernel_size + 1
                                        conv1_kernel.len()]; //number of conv kernels
                                        x.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &x,
        &conv1_kernel,
        &mut conv1_output,
        x_0,
        conv1_weights_0,
        conv1_output_0,
        &multiplier_conv1,
    );

    //layer 1

    relu4d_u8(&mut conv1_output, conv1_output_0);

    let avg_pool1_output = avg_pool_scala_u8(&conv1_output, 2);
    //println!("{} {} {} ", avg_pool1_output[0].len() , avg_pool1_output[0][0].len() , avg_pool1_output[0][0][0].len());

    //layer 2

    let mut conv2_output = vec![vec![vec![vec![0u8; avg_pool1_output[0][0][0].len() - conv2_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                        avg_pool1_output[0][0].len() - conv2_kernel[0][0].len()+ 1]; // h - kernel_size + 1
                                                                        conv2_kernel.len()]; //number of conv kernels
                                                                        avg_pool1_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &avg_pool1_output,
        &conv2_kernel,
        &mut conv2_output,
        conv1_output_0,
        conv2_weights_0,
        conv2_output_0,
        &multiplier_conv2,
    );
    relu4d_u8(&mut conv2_output, conv2_output_0);

    let avg_pool2_output = avg_pool_scala_u8(&conv2_output, 2);
    //println!("{} {} {} ", avg_pool2_output[0].len() , avg_pool2_output[0][0].len() , avg_pool2_output[0][0][0].len());

    //layer 3
    let mut conv3_output = vec![vec![vec![vec![0u8; avg_pool2_output[0][0][0].len() - conv3_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                        avg_pool2_output[0][0].len() - conv3_kernel[0][0].len()+ 1]; // h - kernel_size + 1
                                                                        conv3_kernel.len()]; //number of conv kernels
                                                                        avg_pool2_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &avg_pool2_output,
        &conv3_kernel,
        &mut conv3_output,
        conv2_output_0,
        conv3_weights_0,
        conv3_output_0,
        &multiplier_conv3,
    );

    relu4d_u8(&mut conv3_output, conv3_output_0);
    //println!("{} {} {} ", conv3_output[0].len() , conv3_output[0][0].len() , conv3_output[0][0][0].len());

    //at the end of layer 3 we have to transform conv3_output to different shape to fit in FC layer.
    // previous shape is [batch size, xxx, 1, 1]. we  want to reshape it to [batch size, xxx]
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
    //println!("flattened conv3 output shape {} {}", transformed_conv3_output.len(), transformed_conv3_output[0].len());
    #[cfg(debug_assertion)]
    println!(
        " FC layer input len : {}, FC layer weight len {}",
        transformed_conv3_output[0].len(),
        fc1_weight[0].len()
    );
    //layer 4
    let mut fc1_output = vec![vec![0u8; fc1_weight.len()];  // channels
                                                transformed_conv3_output.len()]; //batch size
    let fc1_weight_ref: Vec<&[u8]> = fc1_weight.iter().map(|x| x.as_ref()).collect();

    for i in 0..transformed_conv3_output.len() {
        //iterate through each image in the batch
        vec_mat_mul_with_remainder_u8(
            &transformed_conv3_output[i],
            fc1_weight_ref[..].as_ref(),
            &mut fc1_output[i],
            conv3_output_0,
            fc1_weights_0,
            fc1_output_0,
            &multiplier_fc1,
        );
    }
    relu2d_u8(&mut fc1_output, fc1_output_0);

    //layer 5
    let mut fc2_output = vec![vec![0u8; fc2_weight.len()]; // channels
                                                    fc1_output.len()]; //batch size
    let fc2_weight_ref: Vec<&[u8]> = fc2_weight.iter().map(|x| x.as_ref()).collect();

    for i in 0..fc1_output.len() {
        //iterate through each image in the batch
        vec_mat_mul_with_remainder_u8(
            &fc1_output[i],
            fc2_weight_ref[..].as_ref(),
            &mut fc2_output[i],
            fc1_output_0,
            fc2_weights_0,
            fc2_output_0,
            &multiplier_fc2,
        );
    }

    fc2_output
}

pub fn full_circuit_forward_u8(
    x: Vec<u8>,
    l1_mat: Vec<Vec<u8>>,
    l2_mat: Vec<Vec<u8>>,
    x_0: u8,
    y_0: u8,
    z_0: u8,
    l1_mat_0: u8,
    l2_mat_0: u8,
    multiplier_l1: Vec<f32>,
    multiplier_l2: Vec<f32>,
) -> Vec<u8> {
    let mut y = vec![0u8; M];
    let l1_mat_ref: Vec<&[u8]> = l1_mat.iter().map(|x| x.as_ref()).collect();
    vec_mat_mul_with_remainder_u8(
        &x,
        l1_mat_ref[..].as_ref(),
        &mut y,
        x_0,
        l1_mat_0,
        y_0,
        &multiplier_l1,
    );
    //println!("x_0 {}, l1_mat_0 {}, l1_output_0 {}", x_0, l1_mat_0, y_0);
    //println!("l1 output {:?}\n", y);

    relu_u8(&mut y, y_0);
    // println!("relu output {:?}\n", y);
    let mut z = vec![0u8; N];
    let l2_mat_ref: Vec<&[u8]> = l2_mat.iter().map(|x| x.as_ref()).collect();
    vec_mat_mul_with_remainder_u8(
        &y,
        l2_mat_ref[..].as_ref(),
        &mut z,
        y_0,
        l2_mat_0,
        z_0,
        &multiplier_l2,
    );
    // println!("l2 output {:?}\n", z);

    z
}



fn padding_helper(input: Vec<Vec<Vec<Vec<u8>>>>, padding: usize) -> Vec<Vec<Vec<Vec<u8>>>> {
    let mut padded_output: Vec<Vec<Vec<Vec<u8>>>> =
        vec![
            vec![
                vec![vec![0; input[0][0][0].len() + padding * 2]; input[0][0].len() + padding * 2];
                input[0].len()
            ];
            input.len()
        ];
    for i in 0..input.len() {
        for j in 0..input[0].len(){
            for k in padding..(padded_output[0][0].len() - padding - 1) {
                for m in padding..(padded_output[0][0][0].len() - padding -1) {
                    padded_output[i][j][k][m] = input[i][j][k][m];
                }
            }
        }
    }
    padded_output
}

fn print_size(input: Vec<Vec<Vec<Vec<u8>>>>){
    let size1 = input.len();
    let size2 = input[0].len();
    let size3 = input[0][0].len();
    let size4 = input[0][0][0].len();
    println!("size:[{},{},{},{}]",size1,size2,size3,size4);
}

pub(crate) fn residual_add(input1: &Vec<Vec<Vec<Vec<u8>>>>, input2: &Vec<Vec<Vec<Vec<u8>>>>, output: &mut Vec<Vec<Vec<Vec<u8>>>>){
    let size1 = input1.len();
    let size2 = input1[0].len();
    let size3 = input1[0][0].len();
    let size4 = input1[0][0][0].len();
    for i in 0..size1 {
        for j in 0..size2 {
            for m in 0..size3 {
                for n in 0..size4 {
                    output[i][j][m][n] = input1[i][j][m][n] + input2[i][j][m][n];
                }
            }
        }
    }
}

pub(crate) fn residual_add_plaintext(
    input1: &Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    input2: &Vec<Vec<Vec<Vec<u8>>>>, 
    output: &mut Vec<Vec<Vec<Vec<u8>>>>,

    input1_0: u8,
    input2_0: u8,
    output_0: u8,
    multiplier_1: &[f32],
    multiplier_2: &[f32],
) -> (Vec<Vec<Vec<Vec<u32>>>>, Vec<Vec<Vec<Vec<u32>>>>){
    let batch_size = input1.len();
    let channel_num = input1[0].len();
    let input_height = input1[0][0].len();
    let input_width = input1[0][0][0].len();
    let mut remainder =
        vec![vec![vec![vec![0u32; input_width]; input_height]; channel_num]; batch_size];
    let mut div =
        vec![vec![vec![vec![0u32; input_width]; input_height]; channel_num]; batch_size];
    for n in 0..batch_size {
        for k in 0..channel_num {
            for h in 0..input_height {
                for w in 0..input_width {
                    let m1 = (multiplier_1[k] * 2u64.pow(22) as f32) as u64;
                    let m2 = (multiplier_2[k] * 2u64.pow(22) as f32) as u64;

                    let output_scaled_zero: u64 = output_0 as u64 *2u64.pow(22) ;
                    let tmp1: u64 = m1 as u64 * input2[n][k][h][w] as u64;
                    let tmp2: u64 = m2 as u64 * input1[n][k][h][w] as u64;
                    let tmp3: u64 = m1 as u64 * input2_0 as u64;
                    let tmp4: u64 = m2 as u64 *input1_0 as u64;

                    let tmp = (output_scaled_zero+tmp1+tmp2) - (tmp3+tmp4);
                    
                    output[n][k][h][w] = (tmp / 2u64.pow(22)) as u8;
                    
                    remainder[n][k][h][w] = (tmp % 2u64.pow(22)) as u32;
                    div[n][k][h][w] = (tmp / 2u64.pow(30)) as u32;
                }
            }
        }
    }
    (remainder, div)
}



fn conv_plaintext_wrapper(
    input: &Vec<Vec<Vec<Vec<u8>>>>,  // [Batch Size, Num Channel, Height, Width]
    conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    input_0: u8,
    weight_0: u8,
    output_0: u8,
    multiplier: Vec<f32>,
    padding: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    let padded_input = padding_helper(input.clone(), padding);

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

    output
}


fn residual_add_plaintext_wrapper(
    input1: &Vec<Vec<Vec<Vec<u8>>>>, // [Batch Size, Num Channel, Height, Width]
    input2: &Vec<Vec<Vec<Vec<u8>>>>,

    input1_0: u8,
    input2_0: u8,
    output_0: u8,
    multiplier_1: Vec<f32>,
    multiplier_2: Vec<f32>,
) -> Vec<Vec<Vec<Vec<u8>>>> {

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

    residual_output
}


fn resnet18_residual_block_plaintext_wrapper(
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
) -> Vec<Vec<Vec<Vec<u8>>>> {
    let first_conv_output = conv_plaintext_wrapper(
        &input,
        &first_conv_weights,
        input_0,
        first_conv_weight_0,
        first_conv_output_0,
        first_conv_weight_multiplier.clone(),
        padding,
    );

    let second_conv_output = conv_plaintext_wrapper(
        &first_conv_output,
        &second_conv_weights,
        first_conv_output_0,
        second_conv_weight_0,
        second_conv_output_0,
        second_conv_weight_multiplier.clone(),
        padding,
    );

    // residual add 1 =================================================================================================================
    let residual_output = residual_add_plaintext_wrapper(
        &first_residual_input,
        &second_conv_output,
        first_residual_input_0,
        second_conv_output_0,
        output_0,
        first_residual_input_multiplier,
        second_residual_input_multiplier,
    );

    residual_output
}


fn resnet50_residual_block_plaintext_wrapper(
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

    third_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    third_conv_weight_0: u8,
    third_conv_weight_multiplier: Vec<f32>,
    third_conv_output_0: u8,

    first_residual_input: &Vec<Vec<Vec<Vec<u8>>>>,
    first_residual_input_0: u8,

    first_residual_input_multiplier: Vec<f32>,
    second_residual_input_multiplier: Vec<f32>,

    output_0: u8,
    padding: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    // Three conv layers, and a residual layer.
    let first_conv_output = conv_plaintext_wrapper(
        &input,
        &first_conv_weights,
        input_0,
        first_conv_weight_0,
        first_conv_output_0,
        first_conv_weight_multiplier.clone(),
        padding,
    );

    let second_conv_output = conv_plaintext_wrapper(
        &first_conv_output,
        &second_conv_weights,
        first_conv_output_0,
        second_conv_weight_0,
        second_conv_output_0,
        second_conv_weight_multiplier.clone(),
        padding,
    );

    let third_conv_output = conv_plaintext_wrapper(
        &second_conv_output,
        &third_conv_weights,
        second_conv_output_0,
        third_conv_weight_0,
        third_conv_output_0,
        third_conv_weight_multiplier.clone(),
        padding,
    );

    // residual add 1 =================================================================================================================
    let residual_output = residual_add_plaintext_wrapper(
        &first_residual_input,
        &third_conv_output,
        first_residual_input_0,
        third_conv_output_0,
        output_0,
        first_residual_input_multiplier,
        second_residual_input_multiplier,
    );

    residual_output
}


fn resnet50_residual_meta_block_plaintext_wrapper(
    input: &Vec<Vec<Vec<Vec<u8>>>>,
    input_0: u8,

    conv_residual_weight: Vec<Vec<Vec<Vec<u8>>>>,
    conv_residual_weights_0: u8,
    conv_residual_weight_multiplier: Vec<f32>,
    conv_residual_output_0: u8,
    
    first_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    first_conv_weight_0: u8,
    first_conv_weight_multiplier: Vec<f32>,
    first_conv_output_0: u8,

    second_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    second_conv_weight_0: u8,
    second_conv_weight_multiplier: Vec<f32>,
    second_conv_output_0: u8,

    third_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    third_conv_weight_0: u8,
    third_conv_weight_multiplier: Vec<f32>,
    third_conv_output_0: u8,

    forth_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    forth_conv_weight_0: u8,
    forth_conv_weight_multiplier: Vec<f32>,
    forth_conv_output_0: u8,

    fifth_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    fifth_conv_weight_0: u8,
    fifth_conv_weight_multiplier: Vec<f32>,
    fifth_conv_output_0: u8,

    sixth_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    sixth_conv_weight_0: u8,
    sixth_conv_weight_multiplier: Vec<f32>,
    sixth_conv_output_0: u8,

    seventh_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    seventh_conv_weight_0: u8,
    seventh_conv_weight_multiplier: Vec<f32>,
    seventh_conv_output_0: u8,

    eighth_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    eighth_conv_weight_0: u8,
    eighth_conv_weight_multiplier: Vec<f32>,
    eighth_conv_output_0: u8,

    nineth_conv_weights: &Vec<Vec<Vec<Vec<u8>>>>,
    nineth_conv_weight_0: u8,
    nineth_conv_weight_multiplier: Vec<f32>,
    nineth_conv_output_0: u8,

    first_add_residual1_input_multiplier: Vec<f32>,
    second_add_residual1_input_multiplier: Vec<f32>,
    add_residual1_output_0: u8,

    first_add_residual2_input_multiplier: Vec<f32>,
    second_add_residual2_input_multiplier: Vec<f32>,
    add_residual2_output_0: u8,

    first_add_residual3_input_multiplier: Vec<f32>,
    second_add_residual3_input_multiplier: Vec<f32>,
    add_residual3_output_0: u8,

    padding: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    // 9 conv layers.

    // residual 1*1 conv 1 =================================================================================
    let residual_conv_output = conv_plaintext_wrapper(
        &input,
        &conv_residual_weight,
        input_0,
        conv_residual_weights_0,
        conv_residual_output_0,
        conv_residual_weight_multiplier,
        0,
    );

    //----------------------------------------------------------------------
    let add_residual1_output = resnet50_residual_block_plaintext_wrapper(
        &input,
        input_0,

        &first_conv_weights,
        first_conv_weight_0,
        first_conv_weight_multiplier,
        first_conv_output_0,

        &second_conv_weights,
        second_conv_weight_0,
        second_conv_weight_multiplier,
        second_conv_output_0,

        &third_conv_weights,
        third_conv_weight_0,
        third_conv_weight_multiplier,
        third_conv_output_0,

        &residual_conv_output,
        conv_residual_output_0,

        first_add_residual1_input_multiplier,
        second_add_residual1_input_multiplier,
        
        add_residual1_output_0,
        padding,
    );

    let add_residual2_output = resnet50_residual_block_plaintext_wrapper(
        &add_residual1_output,
        add_residual1_output_0,

        &forth_conv_weights,
        forth_conv_weight_0,
        forth_conv_weight_multiplier,
        forth_conv_output_0,

        &fifth_conv_weights,
        fifth_conv_weight_0,
        fifth_conv_weight_multiplier,
        fifth_conv_output_0,

        &sixth_conv_weights,
        sixth_conv_weight_0,
        sixth_conv_weight_multiplier,
        sixth_conv_output_0,

        &add_residual1_output,
        add_residual1_output_0,

        first_add_residual2_input_multiplier,
        second_add_residual2_input_multiplier,
        
        add_residual2_output_0,
        padding,
    );

    let add_residual3_output = resnet50_residual_block_plaintext_wrapper(
        &add_residual2_output,
        add_residual2_output_0,

        &seventh_conv_weights,
        seventh_conv_weight_0,
        seventh_conv_weight_multiplier,
        seventh_conv_output_0,

        &eighth_conv_weights,
        eighth_conv_weight_0,
        eighth_conv_weight_multiplier,
        eighth_conv_output_0,

        &nineth_conv_weights,
        nineth_conv_weight_0,
        nineth_conv_weight_multiplier,
        nineth_conv_output_0,

        &add_residual2_output,
        add_residual2_output_0,

        first_add_residual3_input_multiplier,
        second_add_residual3_input_multiplier,
        
        add_residual3_output_0,
        padding,
    );

    add_residual3_output
}


pub fn resnet50_circuit_forward_u8(
    padding:usize,
    x: Vec<Vec<Vec<Vec<u8>>>>,

    conv21_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv22_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv23_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv24_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv25_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv26_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv27_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv28_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv29_weights: Vec<Vec<Vec<Vec<u8>>>>,
    // ----------------------------- --
    conv31_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv32_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv33_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv34_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv35_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv36_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv37_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv38_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv39_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv3_10_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv3_11_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv3_12_weights: Vec<Vec<Vec<Vec<u8>>>>,

    //------------------------------ --
    conv41_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv42_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv43_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv44_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv45_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv46_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv47_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv48_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv49_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv4_10_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv4_11_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv4_12_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv4_13_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv4_14_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv4_15_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv4_16_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv4_17_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv4_18_weights: Vec<Vec<Vec<Vec<u8>>>>,

    //------------------------------ --
    conv51_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv52_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv53_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv54_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv55_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv56_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv57_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv58_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv59_weights: Vec<Vec<Vec<Vec<u8>>>>,

    conv_residual1_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    conv_residual4_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    conv_residual8_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    conv_residual11_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    conv_residual14_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1

    fc1_weights: Vec<Vec<u8>>,


    //zero points for quantization.
    x_0: u8,

    conv21_output_0: u8,
    conv22_output_0: u8,
    conv23_output_0: u8,
    conv24_output_0: u8,
    conv25_output_0: u8,
    conv26_output_0: u8,
    conv27_output_0: u8,
    conv28_output_0: u8,
    conv29_output_0: u8,

    conv31_output_0: u8,
    conv32_output_0: u8,
    conv33_output_0: u8,
    conv34_output_0: u8,
    conv35_output_0: u8,
    conv36_output_0: u8,
    conv37_output_0: u8,
    conv38_output_0: u8,
    conv39_output_0: u8,
    conv3_10_output_0: u8,
    conv3_11_output_0: u8,
    conv3_12_output_0: u8,

    conv41_output_0: u8,
    conv42_output_0: u8,
    conv43_output_0: u8,
    conv44_output_0: u8,
    conv45_output_0: u8,
    conv46_output_0: u8,
    conv47_output_0: u8,
    conv48_output_0: u8,
    conv49_output_0: u8,
    conv4_10_output_0: u8,
    conv4_11_output_0: u8,
    conv4_12_output_0: u8,
    conv4_13_output_0: u8,
    conv4_14_output_0: u8,
    conv4_15_output_0: u8,
    conv4_16_output_0: u8,
    conv4_17_output_0: u8,
    conv4_18_output_0: u8,

    conv51_output_0: u8,
    conv52_output_0: u8,
    conv53_output_0: u8,
    conv54_output_0: u8,
    conv55_output_0: u8,
    conv56_output_0: u8,
    conv57_output_0: u8,
    conv58_output_0: u8,
    conv59_output_0: u8,

    conv_residual1_output_0:u8,  // residual output_0 1
    conv_residual4_output_0:u8,  // residual output_0 1
    conv_residual8_output_0:u8,  // residual output_0 1
    conv_residual11_output_0:u8,  // residual output_0 1
    conv_residual14_output_0:u8,  // residual output_0 1

    fc1_output_0: u8,


    conv21_weights_0: u8,
    conv22_weights_0: u8,
    conv23_weights_0: u8,
    conv24_weights_0: u8,
    conv25_weights_0: u8,
    conv26_weights_0: u8,
    conv27_weights_0: u8,
    conv28_weights_0: u8,
    conv29_weights_0: u8,

    conv31_weights_0: u8,
    conv32_weights_0: u8,
    conv33_weights_0: u8,
    conv34_weights_0: u8,
    conv35_weights_0: u8,
    conv36_weights_0: u8,
    conv37_weights_0: u8,
    conv38_weights_0: u8,
    conv39_weights_0: u8,
    conv3_10_weights_0: u8,
    conv3_11_weights_0: u8,
    conv3_12_weights_0: u8,

    conv41_weights_0: u8,
    conv42_weights_0: u8,
    conv43_weights_0: u8,
    conv44_weights_0: u8,
    conv45_weights_0: u8,
    conv46_weights_0: u8,
    conv47_weights_0: u8,
    conv48_weights_0: u8,
    conv49_weights_0: u8,
    conv4_10_weights_0: u8,
    conv4_11_weights_0: u8,
    conv4_12_weights_0: u8,
    conv4_13_weights_0: u8,
    conv4_14_weights_0: u8,
    conv4_15_weights_0: u8,
    conv4_16_weights_0: u8,
    conv4_17_weights_0: u8,
    conv4_18_weights_0: u8,

    conv51_weights_0: u8,
    conv52_weights_0: u8,
    conv53_weights_0: u8,
    conv54_weights_0: u8,
    conv55_weights_0: u8,
    conv56_weights_0: u8,
    conv57_weights_0: u8,
    conv58_weights_0: u8,
    conv59_weights_0: u8,

    conv_residual1_weights_0:u8,  // residual weights_0 1
    conv_residual4_weights_0:u8,  // residual weights_0 1
    conv_residual8_weights_0:u8,  // residual weights_0 1
    conv_residual11_weights_0:u8,  // residual weights_0 1
    conv_residual14_weights_0:u8,  // residual weights_0 1

    fc1_weights_0: u8,

    conv21_multiplier: Vec<f32>,
    conv22_multiplier: Vec<f32>,
    conv23_multiplier: Vec<f32>,
    conv24_multiplier: Vec<f32>,
    conv25_multiplier: Vec<f32>,
    conv26_multiplier: Vec<f32>,
    conv27_multiplier: Vec<f32>,
    conv28_multiplier: Vec<f32>,
    conv29_multiplier: Vec<f32>,

    conv31_multiplier: Vec<f32>,
    conv32_multiplier: Vec<f32>,
    conv33_multiplier: Vec<f32>,
    conv34_multiplier: Vec<f32>,
    conv35_multiplier: Vec<f32>,
    conv36_multiplier: Vec<f32>,
    conv37_multiplier: Vec<f32>,
    conv38_multiplier: Vec<f32>,
    conv39_multiplier: Vec<f32>,
    conv3_10_multiplier: Vec<f32>,
    conv3_11_multiplier: Vec<f32>,
    conv3_12_multiplier: Vec<f32>,

    conv41_multiplier: Vec<f32>,
    conv42_multiplier: Vec<f32>,
    conv43_multiplier: Vec<f32>,
    conv44_multiplier: Vec<f32>,
    conv45_multiplier: Vec<f32>,
    conv46_multiplier: Vec<f32>,
    conv47_multiplier: Vec<f32>,
    conv48_multiplier: Vec<f32>,
    conv49_multiplier: Vec<f32>,
    conv4_10_multiplier: Vec<f32>,
    conv4_11_multiplier: Vec<f32>,
    conv4_12_multiplier: Vec<f32>,
    conv4_13_multiplier: Vec<f32>,
    conv4_14_multiplier: Vec<f32>,
    conv4_15_multiplier: Vec<f32>,
    conv4_16_multiplier: Vec<f32>,
    conv4_17_multiplier: Vec<f32>,
    conv4_18_multiplier: Vec<f32>,

    conv51_multiplier: Vec<f32>,
    conv52_multiplier: Vec<f32>,
    conv53_multiplier: Vec<f32>,
    conv54_multiplier: Vec<f32>,
    conv55_multiplier: Vec<f32>,
    conv56_multiplier: Vec<f32>,
    conv57_multiplier: Vec<f32>,
    conv58_multiplier: Vec<f32>,
    conv59_multiplier: Vec<f32>,

    conv_residual1_multiplier:Vec<f32>,  // residual multiplier 1
    conv_residual4_multiplier:Vec<f32>,  // residual multiplier 1
    conv_residual8_multiplier:Vec<f32>,  // residual multiplier 1
    conv_residual11_multiplier:Vec<f32>,  // residual multiplier 1
    conv_residual14_multiplier:Vec<f32>,  // residual multiplier 1

    add_residual1_output_0:u8,
    add_residual2_output_0:u8,
    add_residual3_output_0:u8,
    add_residual4_output_0:u8,
    add_residual5_output_0:u8,
    add_residual6_output_0:u8,
    add_residual7_output_0:u8,
    add_residual8_output_0:u8,
    add_residual9_output_0:u8,
    add_residual10_output_0:u8,
    add_residual11_output_0:u8,
    add_residual12_output_0:u8,
    add_residual13_output_0:u8,
    add_residual14_output_0:u8,
    add_residual15_output_0:u8,
    add_residual16_output_0:u8,

    add_residual1_first_multiplier:Vec<f32>,
    add_residual2_first_multiplier:Vec<f32>,
    add_residual3_first_multiplier:Vec<f32>,
    add_residual4_first_multiplier:Vec<f32>,
    add_residual5_first_multiplier:Vec<f32>,
    add_residual6_first_multiplier:Vec<f32>,
    add_residual7_first_multiplier:Vec<f32>,
    add_residual8_first_multiplier:Vec<f32>,
    add_residual9_first_multiplier:Vec<f32>,
    add_residual10_first_multiplier:Vec<f32>,
    add_residual11_first_multiplier:Vec<f32>,
    add_residual12_first_multiplier:Vec<f32>,
    add_residual13_first_multiplier:Vec<f32>,
    add_residual14_first_multiplier:Vec<f32>,
    add_residual15_first_multiplier:Vec<f32>,
    add_residual16_first_multiplier:Vec<f32>,

    add_residual1_second_multiplier:Vec<f32>,
    add_residual2_second_multiplier:Vec<f32>,
    add_residual3_second_multiplier:Vec<f32>,
    add_residual4_second_multiplier:Vec<f32>,
    add_residual5_second_multiplier:Vec<f32>,
    add_residual6_second_multiplier:Vec<f32>,
    add_residual7_second_multiplier:Vec<f32>,
    add_residual8_second_multiplier:Vec<f32>,
    add_residual9_second_multiplier:Vec<f32>,
    add_residual10_second_multiplier:Vec<f32>,
    add_residual11_second_multiplier:Vec<f32>,
    add_residual12_second_multiplier:Vec<f32>,
    add_residual13_second_multiplier:Vec<f32>,
    add_residual14_second_multiplier:Vec<f32>,
    add_residual15_second_multiplier:Vec<f32>,
    add_residual16_second_multiplier:Vec<f32>,

    multiplier_fc1: Vec<f32>,
) -> Vec<Vec<u8>> {

    let add_residual3_output = resnet50_residual_meta_block_plaintext_wrapper(
           &x,
           x_0,
       
           conv_residual1_weight,
           conv_residual1_weights_0,
           conv_residual1_multiplier,
           conv_residual1_output_0,

           &conv21_weights,
           conv21_weights_0,
           conv21_multiplier,
           conv21_output_0,
       
           &conv22_weights,
           conv22_weights_0,
           conv22_multiplier,
           conv22_output_0,

           &conv23_weights,
           conv23_weights_0,
           conv23_multiplier,
           conv23_output_0,

           &conv24_weights,
           conv24_weights_0,
           conv24_multiplier,
           conv24_output_0,

           &conv25_weights,
           conv25_weights_0,
           conv25_multiplier,
           conv25_output_0,

           &conv26_weights,
           conv26_weights_0,
           conv26_multiplier,
           conv26_output_0,

           &conv27_weights,
           conv27_weights_0,
           conv27_multiplier,
           conv27_output_0,

           &conv28_weights,
           conv28_weights_0,
           conv28_multiplier,
           conv28_output_0,

           &conv29_weights,
           conv29_weights_0,
           conv29_multiplier,
           conv29_output_0,

           add_residual1_first_multiplier,
           add_residual1_second_multiplier,
           add_residual1_output_0,
           add_residual2_first_multiplier,
           add_residual2_second_multiplier,
           add_residual2_output_0,
       
           add_residual3_first_multiplier,
           add_residual3_second_multiplier,
           add_residual3_output_0,
       
           padding,
    );

    // =================================================================================================================================
    let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&add_residual3_output.clone(), 2);

    // ====================================================================================================================================
    let add_residual6_output = resnet50_residual_meta_block_plaintext_wrapper(
        &avg_pool2_output,
        add_residual3_output_0,
    
        conv_residual4_weight,
        conv_residual4_weights_0,
        conv_residual4_multiplier,
        conv_residual4_output_0,

        &conv31_weights,
        conv31_weights_0,
        conv31_multiplier,
        conv31_output_0,
    
        &conv32_weights,
        conv32_weights_0,
        conv32_multiplier,
        conv32_output_0,

        &conv33_weights,
        conv33_weights_0,
        conv33_multiplier,
        conv33_output_0,

        &conv34_weights,
        conv34_weights_0,
        conv34_multiplier,
        conv34_output_0,

        &conv35_weights,
        conv35_weights_0,
        conv35_multiplier,
        conv35_output_0,

        &conv36_weights,
        conv36_weights_0,
        conv36_multiplier,
        conv36_output_0,

        &conv37_weights,
        conv37_weights_0,
        conv37_multiplier,
        conv37_output_0,

        &conv38_weights,
        conv38_weights_0,
        conv38_multiplier,
        conv38_output_0,

        &conv39_weights,
        conv39_weights_0,
        conv39_multiplier,
        conv39_output_0,

        add_residual4_first_multiplier,
        add_residual4_second_multiplier,
        add_residual4_output_0,
        add_residual5_first_multiplier,
        add_residual5_second_multiplier,
        add_residual5_output_0,
    
        add_residual6_first_multiplier,
        add_residual6_second_multiplier,
        add_residual6_output_0,
    
        padding,
    );

    //----------------------------------------------------------------------
    let add_residual7_output = resnet50_residual_block_plaintext_wrapper(
        &add_residual6_output,
        add_residual6_output_0,

        &conv3_10_weights,
        conv3_10_weights_0,
        conv3_10_multiplier,
        conv3_10_output_0,

        &conv3_11_weights,
        conv3_11_weights_0,
        conv3_11_multiplier,
        conv3_11_output_0,

        &conv3_12_weights,
        conv3_12_weights_0,
        conv3_12_multiplier,
        conv3_12_output_0,

        &add_residual6_output,
        add_residual6_output_0,

        add_residual7_first_multiplier,
        add_residual7_second_multiplier,
        
        add_residual7_output_0,
        padding,
    );

    // =================================================================================================================================
    let (avg_pool3_output, avg3_remainder) = avg_pool_with_remainder_scala_u8(&add_residual7_output.clone(), 2);

    // ====================================================================================================================================
    let add_residual10_output = resnet50_residual_meta_block_plaintext_wrapper(
        &avg_pool3_output,
        add_residual7_output_0,
    
        conv_residual8_weight,
        conv_residual8_weights_0,
        conv_residual8_multiplier,
        conv_residual8_output_0,

        &conv41_weights,
        conv41_weights_0,
        conv41_multiplier,
        conv41_output_0,
    
        &conv42_weights,
        conv42_weights_0,
        conv42_multiplier,
        conv42_output_0,

        &conv43_weights,
        conv43_weights_0,
        conv43_multiplier,
        conv43_output_0,

        &conv44_weights,
        conv44_weights_0,
        conv44_multiplier,
        conv44_output_0,

        &conv45_weights,
        conv45_weights_0,
        conv45_multiplier,
        conv45_output_0,

        &conv46_weights,
        conv46_weights_0,
        conv46_multiplier,
        conv46_output_0,

        &conv47_weights,
        conv47_weights_0,
        conv47_multiplier,
        conv47_output_0,

        &conv48_weights,
        conv48_weights_0,
        conv48_multiplier,
        conv48_output_0,

        &conv49_weights,
        conv49_weights_0,
        conv49_multiplier,
        conv49_output_0,

        add_residual8_first_multiplier,
        add_residual8_second_multiplier,
        add_residual8_output_0,
    
        add_residual9_first_multiplier,
        add_residual9_second_multiplier,
        add_residual9_output_0,
    
        add_residual10_first_multiplier,
        add_residual10_second_multiplier,
        add_residual10_output_0,
    
        padding,
    );


    let add_residual13_output = resnet50_residual_meta_block_plaintext_wrapper(
        &add_residual10_output,
        add_residual10_output_0,
    
        conv_residual11_weight,
        conv_residual11_weights_0,
        conv_residual11_multiplier,
        conv_residual11_output_0,

        &conv4_10_weights,
        conv4_10_weights_0,
        conv4_10_multiplier,
        conv4_10_output_0,
    
        &conv4_11_weights,
        conv4_11_weights_0,
        conv4_11_multiplier,
        conv4_11_output_0,

        &conv4_12_weights,
        conv4_12_weights_0,
        conv4_12_multiplier,
        conv4_12_output_0,

        &conv4_13_weights,
        conv4_13_weights_0,
        conv4_13_multiplier,
        conv4_13_output_0,

        &conv4_14_weights,
        conv4_14_weights_0,
        conv4_14_multiplier,
        conv4_14_output_0,

        &conv4_15_weights,
        conv4_15_weights_0,
        conv4_15_multiplier,
        conv4_15_output_0,

        &conv4_16_weights,
        conv4_16_weights_0,
        conv4_16_multiplier,
        conv4_16_output_0,

        &conv4_17_weights,
        conv4_17_weights_0,
        conv4_17_multiplier,
        conv4_17_output_0,

        &conv4_18_weights,
        conv4_18_weights_0,
        conv4_18_multiplier,
        conv4_18_output_0,

        add_residual11_first_multiplier,
        add_residual11_second_multiplier,
        add_residual11_output_0,
    
        add_residual12_first_multiplier,
        add_residual12_second_multiplier,
        add_residual12_output_0,
    
        add_residual13_first_multiplier,
        add_residual13_second_multiplier,
        add_residual13_output_0,
    
        padding,
    );

    // =================================================================================================================================
    let (avg_pool4_output, avg4_remainder) = avg_pool_with_remainder_scala_u8(&add_residual13_output.clone(), 2);

    // ==================================================================================================================================
    let add_residual16_output = resnet50_residual_meta_block_plaintext_wrapper(
        &avg_pool4_output,
        add_residual13_output_0,
    
        conv_residual14_weight,
        conv_residual14_weights_0,
        conv_residual14_multiplier,
        conv_residual14_output_0,

        &conv51_weights,
        conv51_weights_0,
        conv51_multiplier,
        conv51_output_0,
    
        &conv52_weights,
        conv52_weights_0,
        conv52_multiplier,
        conv52_output_0,

        &conv53_weights,
        conv53_weights_0,
        conv53_multiplier,
        conv53_output_0,

        &conv54_weights,
        conv54_weights_0,
        conv54_multiplier,
        conv54_output_0,

        &conv55_weights,
        conv55_weights_0,
        conv55_multiplier,
        conv55_output_0,

        &conv56_weights,
        conv56_weights_0,
        conv56_multiplier,
        conv56_output_0,

        &conv57_weights,
        conv57_weights_0,
        conv57_multiplier,
        conv57_output_0,

        &conv58_weights,
        conv58_weights_0,
        conv58_multiplier,
        conv58_output_0,

        &conv59_weights,
        conv59_weights_0,
        conv59_multiplier,
        conv59_output_0,

        add_residual14_first_multiplier,
        add_residual14_second_multiplier,
        add_residual14_output_0,
    
        add_residual15_first_multiplier,
        add_residual15_second_multiplier,
        add_residual15_output_0,
    
        add_residual16_first_multiplier,
        add_residual16_second_multiplier,
        add_residual16_output_0,
    
        padding,
    );

    // =================================================================================================================================
    let (avg_pool5_output, avg5_remainder) = avg_pool_with_remainder_scala_u8(&add_residual16_output.clone(), 4);

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
    let mut fc1_output = vec![vec![0u8; fc1_weights.len()];  // channels
    transformed_avg_pool5_output_output.len()]; //batch size
    let fc1_weight_ref: Vec<&[u8]> = fc1_weights.iter().map(|x| x.as_ref()).collect();

    for i in 0..transformed_avg_pool5_output_output.len() {
        //iterate through each image in the batch
        //in the zkp nn system, we feed one image in each batch to reduce the overhead.
        let (remainder_fc1, div_fc1) = vec_mat_mul_with_remainder_u8(
            &transformed_avg_pool5_output_output[i],
            fc1_weight_ref[..].as_ref(),
            &mut fc1_output[i],
            conv54_output_0,
            fc1_weights_0,
            fc1_output_0,
            &multiplier_fc1.clone(),
        );
    }
    fc1_output
}

pub fn resnet18_circuit_forward_u8(
    padding: usize,            //all the same padding for conv layers
    x: Vec<Vec<Vec<Vec<u8>>>>, //32*32

    conv21_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv22_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv23_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv24_weights: Vec<Vec<Vec<Vec<u8>>>>,
    //------------------------------ --
    conv31_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv32_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv33_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv34_weights: Vec<Vec<Vec<Vec<u8>>>>,

    //------------------------------ --
    conv41_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv42_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv43_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv44_weights: Vec<Vec<Vec<Vec<u8>>>>,

    //------------------------------ --
    conv51_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv52_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv53_weights: Vec<Vec<Vec<Vec<u8>>>>,
    conv54_weights: Vec<Vec<Vec<Vec<u8>>>>,

    conv_residual1_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    conv_residual3_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    conv_residual5_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1
    conv_residual7_weight: Vec<Vec<Vec<Vec<u8>>>>,  // residual kernel 1

    fc1_weights: Vec<Vec<u8>>,

    //zero points for quantization.
    x_0: u8,

    conv21_output_0: u8,
    conv22_output_0: u8,
    conv23_output_0: u8,
    conv24_output_0: u8,

    conv31_output_0: u8,
    conv32_output_0: u8,
    conv33_output_0: u8,
    conv34_output_0: u8,

    conv41_output_0: u8,
    conv42_output_0: u8,
    conv43_output_0: u8,
    conv44_output_0: u8,

    conv51_output_0: u8,
    conv52_output_0: u8,
    conv53_output_0: u8,
    conv54_output_0: u8,

    conv_residual1_output_0:u8,  // residual output_0 1
    conv_residual3_output_0:u8,  // residual output_0 1
    conv_residual5_output_0:u8,  // residual output_0 1
    conv_residual7_output_0:u8,  // residual output_0 1

    fc1_output_0: u8,

    conv21_weights_0: u8,
    conv22_weights_0: u8,
    conv23_weights_0: u8,
    conv24_weights_0: u8,

    conv31_weights_0: u8,
    conv32_weights_0: u8,
    conv33_weights_0: u8,
    conv34_weights_0: u8,

    conv41_weights_0: u8,
    conv42_weights_0: u8,
    conv43_weights_0: u8,
    conv44_weights_0: u8,

    conv51_weights_0: u8,
    conv52_weights_0: u8,
    conv53_weights_0: u8,
    conv54_weights_0: u8,

    conv_residual1_weights_0:u8,  // residual weights_0 1
    conv_residual3_weights_0:u8,  // residual weights_0 1
    conv_residual5_weights_0:u8,  // residual weights_0 1
    conv_residual7_weights_0:u8,  // residual weights_0 1

    fc1_weights_0: u8,

    //multiplier for quantization
    conv21_multiplier: Vec<f32>,
    conv22_multiplier: Vec<f32>,
    conv23_multiplier: Vec<f32>,
    conv24_multiplier: Vec<f32>,

    conv31_multiplier: Vec<f32>,
    conv32_multiplier: Vec<f32>,
    conv33_multiplier: Vec<f32>,
    conv34_multiplier: Vec<f32>,

    conv41_multiplier: Vec<f32>,
    conv42_multiplier: Vec<f32>,
    conv43_multiplier: Vec<f32>,
    conv44_multiplier: Vec<f32>,

    conv51_multiplier: Vec<f32>,
    conv52_multiplier: Vec<f32>,
    conv53_multiplier: Vec<f32>,
    conv54_multiplier: Vec<f32>,

    conv_residual1_multiplier:Vec<f32>,  // residual multiplier 1
    conv_residual3_multiplier:Vec<f32>,  // residual multiplier 1
    conv_residual5_multiplier:Vec<f32>,  // residual multiplier 1
    conv_residual7_multiplier:Vec<f32>,  // residual multiplier 1

    add_residual1_output_0:u8,
    add_residual2_output_0:u8,
    add_residual3_output_0:u8,
    add_residual4_output_0:u8,
    add_residual5_output_0:u8,
    add_residual6_output_0:u8,
    add_residual7_output_0:u8,
    add_residual8_output_0:u8,

    add_residual1_first_multiplier:Vec<f32>,
    add_residual2_first_multiplier:Vec<f32>,
    add_residual3_first_multiplier:Vec<f32>,
    add_residual4_first_multiplier:Vec<f32>,
    add_residual5_first_multiplier:Vec<f32>,
    add_residual6_first_multiplier:Vec<f32>,
    add_residual7_first_multiplier:Vec<f32>,
    add_residual8_first_multiplier:Vec<f32>,

    add_residual1_second_multiplier:Vec<f32>,
    add_residual2_second_multiplier:Vec<f32>,
    add_residual3_second_multiplier:Vec<f32>,
    add_residual4_second_multiplier:Vec<f32>,
    add_residual5_second_multiplier:Vec<f32>,
    add_residual6_second_multiplier:Vec<f32>,
    add_residual7_second_multiplier:Vec<f32>,
    add_residual8_second_multiplier:Vec<f32>,

    multiplier_fc1: Vec<f32>,
) -> Vec<Vec<u8>> {
    println!("resnet18 vallina forward");

    // residual 1*1 conv 1 =================================================================================
    let residual_conv1_output = conv_plaintext_wrapper(
        &x,
        &conv_residual1_weight,
        x_0,
        conv_residual1_weights_0,
        conv_residual1_output_0,
        conv_residual1_multiplier,
        0,
    );

    //----------------------------------------------------------------------
    let add_residual1_output = resnet18_residual_block_plaintext_wrapper(
        &x,
        x_0,
        &conv21_weights,
        conv21_weights_0,
        conv21_multiplier,
        conv21_output_0,
        &conv22_weights,
        conv22_weights_0,
        conv22_multiplier,
        conv22_output_0,
        &residual_conv1_output,
        conv_residual1_output_0,
        add_residual1_first_multiplier,
        add_residual1_second_multiplier,
        add_residual1_output_0,
        padding,
    );


    // =================================================================================================================================
    let add_residual2_output = resnet18_residual_block_plaintext_wrapper(
        &add_residual1_output,
        add_residual1_output_0,
        &conv23_weights,
        conv23_weights_0,
        conv23_multiplier,
        conv23_output_0,
        &conv24_weights,
        conv24_weights_0,
        conv24_multiplier,
        conv24_output_0,
        &residual_conv1_output,
        add_residual1_output_0,
        add_residual2_first_multiplier,
        add_residual2_second_multiplier,
        add_residual2_output_0,
        padding,
    );
    
    // =================================================================================================================================
    let (avg_pool2_output, avg2_remainder) = avg_pool_with_remainder_scala_u8(&add_residual2_output.clone(), 2);

    // residual 1*1 conv 2 =================================================================================
    let residual_conv3_output = conv_plaintext_wrapper(
        &avg_pool2_output,
        &conv_residual3_weight,
        add_residual2_output_0,
        conv_residual3_weights_0,
        conv_residual3_output_0,
        conv_residual3_multiplier,
        0,
    );

    //----------------------------------------------------------------------
    let add_residual3_output = resnet18_residual_block_plaintext_wrapper(
        &avg_pool2_output,
        add_residual2_output_0,
        &conv31_weights,
        conv31_weights_0,
        conv31_multiplier,
        conv31_output_0,
        &conv32_weights,
        conv32_weights_0,
        conv32_multiplier,
        conv32_output_0,
        &residual_conv3_output,
        conv_residual3_output_0,
        add_residual3_first_multiplier,
        add_residual3_second_multiplier,
        add_residual3_output_0,
        padding,
    );

    // =================================================================================================================================
    let add_residual4_output = resnet18_residual_block_plaintext_wrapper(
        &add_residual3_output,
        add_residual3_output_0,
        &conv33_weights,
        conv33_weights_0,
        conv33_multiplier,
        conv33_output_0,
        &conv34_weights,
        conv34_weights_0,
        conv34_multiplier,
        conv34_output_0,
        &add_residual3_output,
        add_residual3_output_0,
        add_residual4_first_multiplier,
        add_residual4_second_multiplier,
        add_residual4_output_0,
        padding,
    );

    // =================================================================================================================================
    let (avg_pool3_output, avg3_remainder) = avg_pool_with_remainder_scala_u8(&add_residual4_output.clone(), 2);

    // residual 1*1 conv 2 =================================================================================
    let residual_conv5_output = conv_plaintext_wrapper(
        &avg_pool3_output,
        &conv_residual5_weight,
        add_residual4_output_0,
        conv_residual5_weights_0,
        conv_residual5_output_0,
        conv_residual5_multiplier,
        0,
    );

    //----------------------------------------------------------------------
    let add_residual5_output = resnet18_residual_block_plaintext_wrapper(
        &avg_pool3_output,
        add_residual4_output_0,
        &conv41_weights,
        conv41_weights_0,
        conv41_multiplier,
        conv41_output_0,
        &conv42_weights,
        conv42_weights_0,
        conv42_multiplier,
        conv42_output_0,
        &residual_conv5_output,
        conv_residual5_output_0,
        add_residual5_first_multiplier,
        add_residual5_second_multiplier,
        add_residual5_output_0,
        padding,
    );


    // =================================================================================================================================
    let add_residual6_output = resnet18_residual_block_plaintext_wrapper(
        &add_residual5_output,
        add_residual5_output_0,
        &conv43_weights,
        conv43_weights_0,
        conv43_multiplier,
        conv43_output_0,
        &conv44_weights,
        conv44_weights_0,
        conv44_multiplier,
        conv44_output_0,
        &add_residual5_output,
        add_residual5_output_0,
        add_residual6_first_multiplier,
        add_residual6_second_multiplier,
        add_residual6_output_0,
        padding,
    );

    // =================================================================================================================================
    let (avg_pool4_output, avg4_remainder) = avg_pool_with_remainder_scala_u8(&add_residual6_output.clone(), 2);

    // residual 1*1 conv 2 =================================================================================
    let residual_conv7_output = conv_plaintext_wrapper(
        &avg_pool4_output,
        &conv_residual7_weight,
        add_residual6_output_0,
        conv_residual7_weights_0,
        conv_residual7_output_0,
        conv_residual7_multiplier,
        0,
    );

    //----------------------------------------------------------------------
    let add_residual7_output = resnet18_residual_block_plaintext_wrapper(
        &avg_pool4_output,
        add_residual6_output_0,
        &conv51_weights,
        conv51_weights_0,
        conv51_multiplier,
        conv51_output_0,
        &conv52_weights,
        conv52_weights_0,
        conv52_multiplier,
        conv52_output_0,
        &residual_conv7_output,
        conv_residual7_output_0,
        add_residual7_first_multiplier,
        add_residual7_second_multiplier,
        add_residual7_output_0,
        padding,
    );

    // =================================================================================================================================
    let add_residual8_output = resnet18_residual_block_plaintext_wrapper(
        &add_residual7_output,
        add_residual7_output_0,
        &conv53_weights,
        conv53_weights_0,
        conv53_multiplier,
        conv53_output_0,
        &conv54_weights,
        conv54_weights_0,
        conv54_multiplier,
        conv54_output_0,
        &add_residual7_output,
        add_residual7_output_0,
        add_residual8_first_multiplier,
        add_residual8_second_multiplier,
        add_residual8_output_0,
        padding,
    );

    // =================================================================================================================================
    let (avg_pool5_output, avg5_remainder) = avg_pool_with_remainder_scala_u8(&add_residual8_output.clone(), 4);

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
    let mut fc1_output = vec![vec![0u8; fc1_weights.len()];  // channels
    transformed_avg_pool5_output_output.len()]; //batch size
    let fc1_weight_ref: Vec<&[u8]> = fc1_weights.iter().map(|x| x.as_ref()).collect();

    for i in 0..transformed_avg_pool5_output_output.len() {
        //iterate through each image in the batch
        //in the zkp nn system, we feed one image in each batch to reduce the overhead.
        let (remainder_fc1, div_fc1) = vec_mat_mul_with_remainder_u8(
            &transformed_avg_pool5_output_output[i],
            fc1_weight_ref[..].as_ref(),
            &mut fc1_output[i],
            conv54_output_0,
            fc1_weights_0,
            fc1_output_0,
            &multiplier_fc1.clone(),
        );
    }

    fc1_output
}

pub fn vgg_circuit_forward_u8(
    x: Vec<Vec<Vec<Vec<u8>>>>, //32*32
    conv11_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv12_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    //--------------------------------- ------ avg pool --------------------
    conv21_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv22_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    //---------------------------------------avg pool --------------- -----
    conv31_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv32_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv33_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    //--------------------------------------- avg pool --------------- -----
    conv41_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv42_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv43_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    //------------------------------ --- ------ avg pool --------------- ----
    conv51_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv52_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    conv53_kernel: Vec<Vec<Vec<Vec<u8>>>>,
    //------------------------------ --- ------ avg pool --------------- ----
    fc1_weight: Vec<Vec<u8>>,
    fc2_weight: Vec<Vec<u8>>,
    fc3_weight: Vec<Vec<u8>>,
    //------------------------------ --- ------ output --------------------
    x_0: u8,
    conv11_output_0: u8,
    conv12_output_0: u8,

    conv21_output_0: u8,
    conv22_output_0: u8,

    conv31_output_0: u8,
    conv32_output_0: u8,
    conv33_output_0: u8,

    conv41_output_0: u8,
    conv42_output_0: u8,
    conv43_output_0: u8,

    conv51_output_0: u8,
    conv52_output_0: u8,
    conv53_output_0: u8,
    
    fc1_output_0: u8,
    fc2_output_0: u8,
    fc3_output_0: u8,

    conv11_weights_0: u8,
    conv12_weights_0: u8,

    conv21_weights_0: u8,
    conv22_weights_0: u8,

    conv31_weights_0: u8,
    conv32_weights_0: u8,
    conv33_weights_0: u8,

    conv41_weights_0: u8,
    conv42_weights_0: u8,
    conv43_weights_0: u8,

    conv51_weights_0: u8,
    conv52_weights_0: u8,
    conv53_weights_0: u8,


    fc1_weights_0: u8,
    fc2_weights_0: u8,
    fc3_weights_0: u8,

    multiplier_conv11: Vec<f32>,
    multiplier_conv12: Vec<f32>,

    multiplier_conv21: Vec<f32>,
    multiplier_conv22: Vec<f32>,

    multiplier_conv31: Vec<f32>,
    multiplier_conv32: Vec<f32>,
    multiplier_conv33: Vec<f32>,

    multiplier_conv41: Vec<f32>,
    multiplier_conv42: Vec<f32>,
    multiplier_conv43: Vec<f32>,

    multiplier_conv51: Vec<f32>,
    multiplier_conv52: Vec<f32>,
    multiplier_conv53: Vec<f32>,

    multiplier_fc1: Vec<f32>,
    multiplier_fc2: Vec<f32>,
    multiplier_fc3: Vec<f32>,

) -> Vec<Vec<u8>> {
    println!("vgg vallina forward1");
    //layer 1
    let padded_x = padding_helper(x.clone(), 1);
    let mut conv11_output = vec![vec![vec![vec![0u8; padded_x[0][0][0].len() - conv11_kernel[0][0][0].len() + 1];  // w - kernel_size  + 1
                                        padded_x[0][0].len() - conv11_kernel[0][0].len() + 1]; // h - kernel_size + 1
                                        conv11_kernel.len()]; //number of conv kernels
                                        padded_x.len()]; //input (image) batch size

    // print_size(x.clone());
    // print_size(conv11_kernel.clone());
    // print_size(conv11_output.clone());
    // print_size(padded_x.clone());

    vec_conv_with_remainder_u8(
        &padded_x,
        &conv11_kernel,
        &mut conv11_output,
        x_0,
        conv11_weights_0,
        conv11_output_0,
        &multiplier_conv11,
    );

    let padded_conv11_output = padding_helper(conv11_output.clone(), 1);
        let mut conv12_output = vec![vec![vec![vec![0u8; padded_conv11_output[0][0][0].len() - conv12_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv11_output[0][0].len() - conv12_kernel[0][0].len()+ 1]; // h - kernel_size+ 1
        conv12_kernel.len()]; //number of conv kernels
        padded_conv11_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_conv11_output,
        &conv12_kernel,
        &mut conv12_output,
        conv11_output_0,
        conv12_weights_0,
        conv12_output_0,
        &multiplier_conv12,
    );

    println!("vgg vallina forward conv11 and 12 finished");

    let avg_pool1_output = avg_pool_scala_u8(&conv12_output, 2);
    print_size(avg_pool1_output.clone());


    //layer 2
    let padded_avg_pool1_output = padding_helper(avg_pool1_output.clone(), 1);
        let mut conv21_output = vec![vec![vec![vec![0u8; padded_avg_pool1_output[0][0][0].len() - conv21_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_avg_pool1_output[0][0].len() - conv21_kernel[0][0].len()+ 1]; // h - kernel_size+ 1
        conv21_kernel.len()]; //number of conv kernels
        padded_avg_pool1_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_avg_pool1_output,
        &conv21_kernel,
        &mut conv21_output,
        conv12_output_0,
        conv21_weights_0,
        conv21_output_0,
        &multiplier_conv21,
    );

    let padded_conv21_output = padding_helper(conv21_output.clone(), 1);
    let mut conv22_output = vec![vec![vec![vec![0u8; padded_conv21_output[0][0][0].len() - conv22_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv21_output[0][0].len() - conv22_kernel[0][0].len()+ 1]; // h - kernel_size+ 1
    conv22_kernel.len()]; //number of conv kernels
    padded_conv21_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_conv21_output,
        &conv22_kernel,
        &mut conv22_output,
        conv21_output_0,
        conv22_weights_0,
        conv22_output_0,
        &multiplier_conv22,
    );


    let avg_pool2_output = avg_pool_scala_u8(&conv22_output, 2);
    print_size(avg_pool2_output.clone());

    //println!("{} {} {} ", conv3_output[0].len() , conv3_output[0][0].len() , conv3_output[0][0][0].len());
    println!("vgg vallina forward conv21 and 22 finished");


  
    //layer 3
    let padded_avg_pool2_output = padding_helper(avg_pool2_output.clone(), 1);
        let mut conv31_output = vec![vec![vec![vec![0u8; padded_avg_pool2_output[0][0][0].len() - conv21_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_avg_pool2_output[0][0].len() - conv31_kernel[0][0].len()+ 1]; // h - kernel_size+ 1
        conv31_kernel.len()]; //number of conv kernels
        padded_avg_pool2_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_avg_pool2_output,
        &conv31_kernel,
        &mut conv31_output,
        conv22_output_0,
        conv31_weights_0,
        conv31_output_0,
        &multiplier_conv31,
    );

    let padded_conv31_output = padding_helper(conv31_output.clone(), 1);
        let mut conv32_output = vec![vec![vec![vec![0u8; padded_conv31_output[0][0][0].len() - conv32_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv31_output[0][0].len() - conv32_kernel[0][0].len()+ 1]; // h - kernel_size+ 1
        conv32_kernel.len()]; //number of conv kernels
        padded_conv31_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_conv31_output,
        &conv32_kernel,
        &mut conv32_output,
        conv31_output_0,
        conv32_weights_0,
        conv32_output_0,
        &multiplier_conv32,
    );

    let padded_conv32_output = padding_helper(conv32_output.clone(), 0);

    let mut conv33_output = vec![vec![vec![vec![0u8; padded_conv32_output[0][0][0].len()];  // w - kernel_size + 1
    padded_conv32_output[0][0].len()]; // h - kernel_size+ 1
    conv33_kernel.len()]; //number of conv kernels
    padded_conv32_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
    &padded_conv32_output,
    &conv33_kernel,
    &mut conv33_output,
    conv32_output_0,
    conv33_weights_0,
    conv33_output_0,
    &multiplier_conv33,
    );
    println!("vgg vallina forward conv31 and 32 33 finished");

    let avg_pool3_output = avg_pool_scala_u8(&conv33_output, 2);
    print_size(avg_pool3_output.clone());

    let padded_conv33_output = padding_helper(avg_pool3_output.clone(), 1);

    let mut conv41_output = vec![vec![vec![vec![0u8; padded_conv33_output[0][0][0].len() - conv41_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv33_output[0][0].len() - conv41_kernel[0][0].len()+ 1]; // h - kernel_size+ 1
    conv41_kernel.len()]; //number of conv kernels
    padded_conv33_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
    &padded_conv33_output,
    &conv41_kernel,
    &mut conv41_output,
    conv33_output_0,
    conv41_weights_0,
    conv41_output_0,
    &multiplier_conv41,
    );
    println!("vgg vallina forward conv41  finished");

    let padded_conv41_output = padding_helper(conv41_output.clone(), 1);

    let mut conv42_output = vec![vec![vec![vec![0u8; padded_conv41_output[0][0][0].len() - conv42_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv41_output[0][0].len() - conv42_kernel[0][0].len()+ 1]; // h - kernel_size+ 1
    conv42_kernel.len()]; //number of conv kernels
    padded_conv41_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_conv41_output,
        &conv42_kernel,
        &mut conv42_output,
        conv41_output_0,
        conv42_weights_0,
        conv42_output_0,
        &multiplier_conv42,
    );
    println!("vgg vallina forward conv42  finished");

    let padded_conv42_output = padding_helper(conv42_output.clone(), 0);

    let mut conv43_output = vec![vec![vec![vec![0u8; padded_conv42_output[0][0][0].len()];  // w - kernel_size + 1
    padded_conv42_output[0][0].len()]; // h - kernel_size+ 1
    conv43_kernel.len()]; //number of conv kernels
    padded_conv42_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_conv42_output,
        &conv43_kernel,
        &mut conv43_output,
        conv42_output_0,
        conv43_weights_0,
        conv43_output_0,
        &multiplier_conv43,
    );

    println!("vgg vallina forward conv41 and 42 43 finished");

    let avg_pool4_output = avg_pool_scala_u8(&conv43_output, 2);
    print_size(avg_pool4_output.clone());

    let padded_conv43_output = padding_helper(avg_pool4_output.clone(), 1);

    let mut conv51_output = vec![vec![vec![vec![0u8; padded_conv43_output[0][0][0].len() - conv51_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv43_output[0][0].len() - conv51_kernel[0][0].len()+ 1]; // h - kernel_size+ 1
    conv51_kernel.len()]; //number of conv kernels
    padded_conv43_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_conv43_output,
        &conv51_kernel,
        &mut conv51_output,
        conv43_output_0,
        conv51_weights_0,
        conv51_output_0,
        &multiplier_conv51,
    );

    let padded_conv51_output = padding_helper(conv51_output.clone(), 1);

    let mut conv52_output = vec![vec![vec![vec![0u8; padded_conv51_output[0][0][0].len() - conv52_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv51_output[0][0].len() - conv52_kernel[0][0].len()+ 1]; // h - kernel_size+ 1
    conv52_kernel.len()]; //number of conv kernels
    padded_conv51_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_conv51_output,
        &conv52_kernel,
        &mut conv52_output,
        conv51_output_0,
        conv52_weights_0,
        conv52_output_0,
        &multiplier_conv52,
    );

    let padded_conv52_output = padding_helper(conv52_output.clone(), 0);

    let mut conv53_output = vec![vec![vec![vec![0u8; padded_conv52_output[0][0][0].len()];  // w - kernel_size + 1
    padded_conv52_output[0][0].len()]; // h - kernel_size+ 1
    conv53_kernel.len()]; //number of conv kernels
    padded_conv52_output.len()]; //input (image) batch size

    vec_conv_with_remainder_u8(
        &padded_conv52_output,
        &conv53_kernel,
        &mut conv53_output,
        conv52_output_0,
        conv53_weights_0,
        conv53_output_0,
        &multiplier_conv53,
    );

    let avg_pool5_output = avg_pool_scala_u8(&conv53_output, 2);

    print_size(avg_pool5_output.clone());
    println!("vgg vallina forward conv51 and 52 53 finished");

    //at the end of layer 5 we have to transform conv53_output to different shape to fit in FC layer.
    // previous shape is [batch size, xxx, 1, 1]. we  want to reshape it to [batch size, xxx]
    let mut transformed_conv53_output =
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
                    transformed_conv53_output[i][counter] = avg_pool5_output[i][j][p][q];
                    counter += 1;
                }
            }
        }
    }
    //println!("flattened conv3 output shape {} {}", transformed_conv53_output.len(), transformed_conv53_output[0].len());
    // println!(
    //     " FC layer input len : {}, FC layer weight len {}",
    //     transformed_conv53_output[0].len(),
    //     fc1_weight[0].len()
    // );





    //FC layer 
    let mut fc1_output = vec![vec![0u8; fc1_weight.len()];  // channels
                                                transformed_conv53_output.len()]; //batch size
    let fc1_weight_ref: Vec<&[u8]> = fc1_weight.iter().map(|x| x.as_ref()).collect();

    for i in 0..transformed_conv53_output.len() {
        //iterate through each image in the batch
        vec_mat_mul_with_remainder_u8(
            &transformed_conv53_output[i],
            fc1_weight_ref[..].as_ref(),
            &mut fc1_output[i],
            conv53_output_0,
            fc1_weights_0,
            fc1_output_0,
            &multiplier_fc1,
        );
    }
    println!("vgg vallina forward fc1 finished");

    let mut fc2_output = vec![vec![0u8; fc2_weight.len()]; // channels
                                                    fc1_output.len()]; //batch size
    let fc2_weight_ref: Vec<&[u8]> = fc2_weight.iter().map(|x| x.as_ref()).collect();

    for i in 0..fc1_output.len() {
        //iterate through each image in the batch
        vec_mat_mul_with_remainder_u8(
            &fc1_output[i],
            fc2_weight_ref[..].as_ref(),
            &mut fc2_output[i],
            fc1_output_0,
            fc2_weights_0,
            fc2_output_0,
            &multiplier_fc2,
        );
    }
    println!("vgg vallina forward fc2 finished");

    let mut fc3_output = vec![vec![0u8; fc3_weight.len()]; // channels
                                                fc2_output.len()]; //batch size
    let fc3_weight_ref: Vec<&[u8]> = fc3_weight.iter().map(|x| x.as_ref()).collect();

    for i in 0..fc2_output.len() {
    //iterate through each image in the batch
        vec_mat_mul_with_remainder_u8(
            &fc2_output[i],
            fc3_weight_ref[..].as_ref(),
            &mut fc3_output[i],
            fc2_output_0,
            fc3_weights_0,
            fc3_output_0,
            &multiplier_fc3,
        );
    }
    println!("vgg vallina forward fc3 finished");

    fc3_output
    // vec![vec![0;1];1]
}


pub fn vec_mat_mul_cos_helper(vec: &[u8], mat: &[u8]) -> u64 {
    let mut res = 0u64;
    for i in 0..mat.len() {
        res += vec[i] as u64 * mat[i] as u64;
    }
    res
}

pub fn cosine_similarity(vec1: Vec<u8>, vec2: Vec<u8>, threshold: u32) -> bool {
    let norm_1 = vec_mat_mul_cos_helper(&vec1, &vec1);
    let norm_2 = vec_mat_mul_cos_helper(&vec2, &vec2);
    let numerator = vec_mat_mul_cos_helper(&vec1, &vec2);

    let res: bool =
        (10000 * numerator * numerator) > (threshold as u64) * (threshold as u64) * norm_1 * norm_2;

    res
}

pub fn argmax_u8(input: Vec<u8>) -> usize {
    let mut res = 0usize;
    let mut tmp_max = 0u8;
    for i in 0..input.len() {
        if input[i] > tmp_max {
            tmp_max = input[i];
            res = i;
        }
    }
    res
}

/// commit the account, output the commitment, and a openning (randomness)
/// currently uses blake2s as the underlying hash function
pub fn commit_x(data: &[i8], seed: &[u8; 32]) -> (Commit, Open) {
    // input
    let input = compress_x(data);

    commit_u8(&input, seed)
}

/// commit the account, output the commitment, and a openning (randomness)
/// currently uses blake2s as the underlying hash function
pub fn commit_z(data: &[i8], seed: &[u8; 32]) -> (Commit, Open) {
    let input = data.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>();

    commit_u8(&input, seed)
}

/// commit the account, output the commitment, and a openning (randomness)
/// currently uses blake2s as the underlying hash function
pub fn commit_u8(data: &[u8], seed: &[u8; 32]) -> (Commit, Open) {
    // blake2s do not take parameters
    let parameters = ();

    // openning
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mut open = [0u8; 32];
    rng.fill(&mut open);

    // commit
    (Commitment::commit(&parameters, &data, &open).unwrap(), open)
}

/// compress x
/// requirement: -128 <= x[i] < 128
pub fn compress_x(data: &[i8]) -> Vec<u8> {
    data.iter().map(|x| (*x as i8) as u8).collect::<Vec<u8>>()
}

/// compress x
/// requirement: 0 <= x[i] < 256
pub fn compress_x_u8(data: &[u8]) -> Vec<u8> {
    data.iter().map(|x| *x as u8).collect::<Vec<u8>>()
}
