rustup override set 1.43.0
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin shallownet_non_knit_encoding --release > result/shallownet_non_knit.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin lenet_small_cifar_non_knit_encoding --release > result/small_cifar_non_knit.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin lenet_large_cifar_non_knit_encoding --release > result/large_cifar_non_knit.log
