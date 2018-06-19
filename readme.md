# sbr

[![Crates.io badge](https://img.shields.io/crates/v/sbr.svg)](https://crates.io/crates/sbr)
[![Docs.rs badge](https://docs.rs/sbr/badge.svg)](https://docs.rs/sbr/)
[![Build Status](https://travis-ci.org/maciejkula/sbr-rs.svg?branch=master)](https://travis-ci.org/maciejkula/sbr-rs)

An implementation of sequence recommenders based on the [wyrm](https://github.com/maciejkula/wyrm) autdifferentiaton library.

## sbr-rs

`sbr` implements efficient recommender algorithms which operate on
sequences of items: given previous items a user has interacted with,
the model will recommend the items the user is likely to interact with
in the future.

### Example
You can fit a model on the Movielens 100K dataset in about 10 seconds:

```rust
let mut data = sbr::datasets::download_movielens_100k().unwrap();

let mut rng = rand::XorShiftRng::from_seed([42; 16]);

let (train, test) = sbr::data::user_based_split(&mut data, &mut rng, 0.2);
let train_mat = train.to_compressed();
let test_mat = test.to_compressed();

println!("Train: {}, test: {}", train.len(), test.len());

let mut model = sbr::models::lstm::Hyperparameters::new(data.num_items(), 32)
    .embedding_dim(32)
    .learning_rate(0.16)
    .l2_penalty(0.0004)
    .lstm_variant(sbr::models::lstm::LSTMVariant::Normal)
    .loss(sbr::models::lstm::Loss::WARP)
    .optimizer(sbr::models::lstm::Optimizer::Adagrad)
    .num_epochs(10)
    .rng(rng)
    .build();

let start = Instant::now();
let loss = model.fit(&train_mat).unwrap();
let elapsed = start.elapsed();
let train_mrr = sbr::evaluation::mrr_score(&model, &train_mat).unwrap();
let test_mrr = sbr::evaluation::mrr_score(&model, &test_mat).unwrap();

println!(
    "Train MRR {} at loss {} and test MRR {} (in {:?})",
    train_mrr, loss, test_mrr, elapsed
);
```

License: MIT
