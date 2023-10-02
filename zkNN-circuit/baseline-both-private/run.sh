rustup override set 1.43.0
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin shallownet_private --release > result/shallownet_private.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin lenet_small_private_cifar --release > result/lenet_small_private_cifar.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo --frozen run --bin lenet_large_private_cifar --release > result/lenet_large_private_cifar.log
