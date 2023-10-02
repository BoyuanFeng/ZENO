rustup override set 1.43.0
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin shallownet_knit_encoding --release > result/shallownet_knit.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin lenet_small_cifar_knit_encoding --release > result/small_cifar_knit.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin lenet_medium_cifar_knit_encoding --release > result/medium_cifar_knit.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin vgg_one_private --release > result/vgg_one_private.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin resnet18_one_private --release > result/resnet18_one_private.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin vgg_both_private --release > result/vgg_both_private.log
