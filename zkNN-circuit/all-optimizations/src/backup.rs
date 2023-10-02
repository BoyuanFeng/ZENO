println!("resnet18 vallina forward");

    let mut residual1 = vec![vec![vec![vec![0u8; x[0][0][0].len()];  // w - kernel_size + 1
                                       x[0][0].len()]; // h - kernel_size + 1
                                        conv22_kernel.len()]; //number of conv kernels
                                        x.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &x,
        &conv_residuel1_kernel,
        &mut residual1,
        x_0,
        conv_residuel1_weights_0,
        conv_residuel1_output_0,
        &multiplier_residuel1,
    );
    println!("residual_1:");
    print_size(residual1.clone());


    println!("x:");
    print_size(x.clone());
    
    //layer 1
    let padded_x = padding_helper(x, padding);


    let mut conv21_output = vec![vec![vec![vec![0u8; padded_x[0][0][0].len() - conv21_kernel[0][0][0].len() + 1];  // w - kernel_size + 1
                                        padded_x[0][0].len() - conv21_kernel[0][0].len() + 1]; // h - kernel_size + 1
                                        conv21_kernel.len()]; //number of conv kernels
                                        padded_x.len()]; //input (image) batch size

    
    vec_conv_with_remainder_u8(
        &padded_x,
        &conv21_kernel,
        &mut conv21_output,
        x_0,
        conv21_weights_0,
        conv21_output_0,
        &multiplier_conv21,
    );


    let padded_conv21_output = padding_helper(conv21_output, padding);
    let mut conv22_output = vec![vec![vec![vec![0u8; padded_conv21_output[0][0][0].len() - conv22_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv21_output[0][0].len() - conv22_kernel[0][0].len()+ 1]; // h - kernel_size + 1
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

    let mut residual1_output = vec![vec![vec![vec![0u8; padded_conv21_output[0][0][0].len() - conv22_kernel[0][0][0].len()+ 1];

    println!("conv22_output:");
    print_size(conv22_output.clone());
    //TODO add residual layer, currently not implemented due to its low overhead.

    let padded_conv22_output = padding_helper(conv22_output, padding);
    let mut conv23_output = vec![vec![vec![vec![0u8; padded_conv22_output[0][0][0].len() - conv23_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv22_output[0][0].len() - conv23_kernel[0][0].len()+ 1]; // h - kernel_size + 1
    conv23_kernel.len()]; //number of conv kernels
    padded_conv22_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &padded_conv22_output,
        &conv23_kernel,
        &mut conv23_output,
        conv22_output_0,
        conv23_weights_0,
        conv23_output_0,
        &multiplier_conv23,
    );


    let padded_conv23_output = padding_helper(conv23_output, padding);
    let mut conv24_output = vec![vec![vec![vec![0u8; padded_conv23_output[0][0][0].len() - conv24_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv23_output[0][0].len() - conv24_kernel[0][0].len()+ 1]; // h - kernel_size + 1
    conv24_kernel.len()]; //number of conv kernels
    padded_conv23_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &padded_conv23_output,
        &conv24_kernel,
        &mut conv24_output,
        conv23_output_0,
        conv24_weights_0,
        conv24_output_0,
        &multiplier_conv24,
    );

    println!("conv24_output:");
    print_size(conv24_output.clone());

    //TODO add residual layer, currently not implemented due to its low overhead.

    let avg_pool2_output = avg_pool_scala_u8(&conv24_output, 2);
    println!("avg_pool2_output:");
    print_size(avg_pool2_output.clone());

    // let mut residual2 = vec![vec![vec![vec![0u8; avg_pool2_output[0][0][0].len()];  // w - kernel_size + 1
    //                                     avg_pool2_output[0][0].len()]; // h - kernel_size + 1
    //                                     conv32_kernel.len()]; //number of conv kernels
    //                                     avg_pool2_output.len()]; //input (image) batch size
    // vec_conv_with_remainder_u8(
    //     &avg_pool2_output,
    //     &conv_residuel2_kernel,
    //     &mut residual2,
    //     conv24_output_0,
    //     conv_residuel2_weights_0,
    //     conv_residuel2_output_0,
    //     &multiplier_residuel2,
    // );
    // println!("residual_2:");
    // print_size(residual2.clone());


    let padded_conv24_output = padding_helper(avg_pool2_output, padding);

    //layer 3
    let mut conv31_output = vec![vec![vec![vec![0u8; padded_conv24_output[0][0][0].len() - conv31_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
        padded_conv24_output[0][0].len() - conv31_kernel[0][0].len()+ 1]; // h - kernel_size + 1
        conv31_kernel.len()]; //number of conv kernels
        padded_conv24_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &padded_conv24_output,
        &conv31_kernel,
        &mut conv31_output,
        conv24_output_0,
        conv31_weights_0,
        conv31_output_0,
        &multiplier_conv31,
    );

    let padded_conv31_output = padding_helper(conv31_output, padding);

    let mut conv32_output = vec![vec![vec![vec![0u8; padded_conv31_output[0][0][0].len() - conv32_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv31_output[0][0].len() - conv32_kernel[0][0].len()+ 1]; // h - kernel_size + 1
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

    println!("conv32_output:");
    print_size(conv32_output.clone());
    //TODO add residual layer, currently not implemented due to its low overhead.

    let padded_conv32_output = padding_helper(conv32_output, padding);

    let mut conv33_output = vec![vec![vec![vec![0u8; padded_conv32_output[0][0][0].len() - conv33_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv32_output[0][0].len() - conv32_kernel[0][0].len()+ 1]; // h - kernel_size + 1
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

    let padded_conv33_output = padding_helper(conv33_output, padding);

    let mut conv34_output = vec![vec![vec![vec![0u8; padded_conv33_output[0][0][0].len() - conv34_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv33_output[0][0].len() - conv32_kernel[0][0].len()+ 1]; // h - kernel_size + 1
    conv34_kernel.len()]; //number of conv kernels
    padded_conv33_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &padded_conv33_output,
        &conv34_kernel,
        &mut conv34_output,
        conv33_output_0,
        conv34_weights_0,
        conv34_output_0,
        &multiplier_conv34,
    );

    println!("conv34_output:");
    print_size(conv34_output.clone());

    //TODO add residual layer, currently not implemented due to its low overhead.

    let avg_pool3_output = avg_pool_scala_u8(&conv34_output, 2);

    println!("avg_pool3_output:");
    print_size(avg_pool3_output.clone());

    // let mut residual3 = vec![vec![vec![vec![0u8; avg_pool3_output[0][0][0].len()];  // w - kernel_size + 1
    //                                     avg_pool3_output[0][0].len()]; // h - kernel_size + 1
    //                                     conv42_kernel.len()]; //number of conv kernels
    //                                     avg_pool3_output.len()]; //input (image) batch size
    // vec_conv_with_remainder_u8(
    //     &avg_pool3_output,
    //     &conv_residuel3_kernel,
    //     &mut residual3,
    //     conv34_output_0,
    //     conv_residuel3_weights_0,
    //     conv_residuel3_output_0,
    //     &multiplier_residuel3,
    // );
    // println!("residual_3:");
    // print_size(residual3.clone());



    let padded_conv34_output = padding_helper(avg_pool3_output, padding);

    let mut conv41_output = vec![vec![vec![vec![0u8;
    padded_conv34_output[0][0][0].len() - conv41_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv34_output[0][0].len() - conv41_kernel[0][0].len()+ 1]; // h - kernel_size + 1
    conv41_kernel.len()]; //number of conv kernels
    padded_conv34_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &padded_conv34_output,
        &conv41_kernel,
        &mut conv41_output,
        conv34_output_0,
        conv41_weights_0,
        conv41_output_0,
        &multiplier_conv41,
    );

    let padded_conv41_output = padding_helper(conv41_output, padding);

    let mut conv42_output = vec![vec![vec![vec![0u8;
    padded_conv41_output[0][0][0].len() - conv42_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv41_output[0][0].len() - conv42_kernel[0][0].len()+ 1]; // h - kernel_size + 1
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

    println!("conv42_output:");
    print_size(conv42_output.clone());
    //TODO add residual layer, currently not implemented due to its low overhead.

    let padded_conv42_output = padding_helper(conv42_output, padding);

    let mut conv43_output = vec![vec![vec![vec![0u8; padded_conv42_output[0][0][0].len() - conv43_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
    padded_conv42_output[0][0].len() - conv43_kernel[0][0].len()+ 1]; // h - kernel_size + 1
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

    let padded_conv43_output = padding_helper(conv43_output, padding);

    let mut conv44_output = vec![vec![vec![vec![0u8; padded_conv43_output[0][0][0].len() - conv44_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                padded_conv43_output[0][0].len() - conv43_kernel[0][0].len()+ 1]; // h - kernel_size + 1
                                                conv44_kernel.len()]; //number of conv kernels
                                                padded_conv43_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &padded_conv43_output,
        &conv44_kernel,
        &mut conv44_output,
        conv43_output_0,
        conv44_weights_0,
        conv44_output_0,
        &multiplier_conv44,
    );

    println!("conv44_output:");
    print_size(conv44_output.clone());

    let avg_pool4_output = avg_pool_scala_u8(&conv44_output, 2);

    println!("avg_pool4_output:");
    print_size(avg_pool4_output.clone());

    // let mut residual4 = vec![vec![vec![vec![0u8; avg_pool4_output[0][0][0].len()];  // w - kernel_size + 1
    //                                     avg_pool4_output[0][0].len()]; // h - kernel_size + 1
    //                                     conv52_kernel.len()]; //number of conv kernels
    //                                     avg_pool4_output.len()]; //input (image) batch size
    // vec_conv_with_remainder_u8(
    //     &avg_pool4_output,
    //     &conv_residuel4_kernel,
    //     &mut residual4,
    //     conv44_output_0,
    //     conv_residuel4_weights_0,
    //     conv_residuel4_output_0,
    //     &multiplier_residuel4,
    // );
    // println!("residual_4:");
    // print_size(residual4.clone());


    let padded_conv44_output = padding_helper(avg_pool4_output, padding);

    let mut conv51_output = vec![vec![vec![vec![0u8; padded_conv44_output[0][0][0].len() - conv51_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                                padded_conv44_output[0][0].len() - conv51_kernel[0][0].len()+ 1]; // h - kernel_size + 1
                                                                conv51_kernel.len()]; //number of conv kernels
                                                                padded_conv44_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &padded_conv44_output,
        &conv51_kernel,
        &mut conv51_output,
        conv44_output_0,
        conv51_weights_0,
        conv51_output_0,
        &multiplier_conv51,
    );

    let padded_conv51_output = padding_helper(conv51_output, padding);

    let mut conv52_output = vec![vec![vec![vec![0u8; padded_conv51_output[0][0][0].len() - conv52_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                padded_conv51_output[0][0].len() - conv52_kernel[0][0].len()+ 1]; // h - kernel_size + 1
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

    println!("conv52_output:");
    print_size(conv52_output.clone());
    
    let padded_conv52_output = padding_helper(conv52_output, padding);
    //TODO add residual layer, currently not implemented due to its low overhead.

    let mut conv53_output = vec![vec![vec![vec![0u8; padded_conv52_output[0][0][0].len() - conv53_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                    padded_conv52_output[0][0].len() - conv53_kernel[0][0].len()+ 1]; // h - kernel_size + 1
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

    let padded_conv53_output = padding_helper(conv53_output, padding);

    let mut conv54_output = vec![vec![vec![vec![0u8; padded_conv53_output[0][0][0].len() - conv54_kernel[0][0][0].len()+ 1];  // w - kernel_size + 1
                                                padded_conv53_output[0][0].len() - conv53_kernel[0][0].len()+ 1]; // h - kernel_size + 1
                                                conv54_kernel.len()]; //number of conv kernels
                                                padded_conv53_output.len()]; //input (image) batch size
    vec_conv_with_remainder_u8(
        &padded_conv53_output,
        &conv54_kernel,
        &mut conv54_output,
        conv53_output_0,
        conv54_weights_0,
        conv54_output_0,
        &multiplier_conv54,
    );

    println!("conv54_output:");
    print_size(conv54_output.clone());

    let avg_pool5_output = avg_pool_scala_u8(&conv54_output, 2);

    //at the end of layer 5 we have to transform conv54_output to different shape to fit in FC layer.
    // previous shape is [batch size, xxx, 1, 1]. we  want to reshape it to [batch size, xxx]
    let mut transformed_conv54_output =
        vec![
            vec![
                0u8;
                conv54_output[0].len() * conv54_output[0][0].len() * conv54_output[0][0][0].len()
            ];
            conv54_output.len()
        ];
    for i in 0..conv54_output.len() {
        let mut counter = 0;
        for j in 0..conv54_output[0].len() {
            for p in 0..conv54_output[0][0].len() {
                for q in 0..conv54_output[0][0][0].len() {
                    transformed_conv54_output[i][counter] = conv54_output[i][j][p][q];
                    counter += 1;
                }
            }
        }
    }

    //FC layer
    let mut fc1_output = vec![vec![0u8; fc1_weight.len()];  // channels
                                                transformed_conv54_output.len()]; //batch size
    let fc1_weight_ref: Vec<&[u8]> = fc1_weight.iter().map(|x| x.as_ref()).collect();

    for i in 0..transformed_conv54_output.len() {
        //iterate through each image in the batch
        vec_mat_mul_with_remainder_u8(
            &transformed_conv54_output[i],
            fc1_weight_ref[..].as_ref(),
            &mut fc1_output[i],
            conv53_output_0,
            fc1_weights_0,
            fc1_output_0,
            &multiplier_fc1,
        );
    }

    fc1_output
}



