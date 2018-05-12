#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate csv;
extern crate rand;
extern crate serde;
extern crate serde_json;
extern crate wheedle;
extern crate wyrm;
#[macro_use]
extern crate serde_derive;

use std::fs::File;
use std::io::{BufReader, Read};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use wheedle::data::{train_test_split, user_based_split, Interaction, Interactions,
                    TripletInteractions};
use wheedle::evaluation::*;
use wheedle::models::factorization;

#[derive(Deserialize, Serialize)]
struct GoodbooksInteraction {
    user_id: usize,
    book_id: usize,
    rating: usize,
}

fn load_goodbooks(path: &str) -> Interactions {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let interactions: Vec<Interaction> = reader
        .deserialize::<GoodbooksInteraction>()
        .map(|x| x.unwrap())
        .enumerate()
        .map(|(i, x)| Interaction::new(x.user_id, x.book_id, i))
        .take(100_000)
        .collect();

    Interactions::from(interactions)
}

#[derive(Debug, Serialize, Deserialize)]
struct Result {
    test_mrr: f32,
    train_mrr: f32,
    elapsed: Duration,
    hyperparameters: factorization::Hyperparameters,
}

fn load_movielens(path: &str) -> Interactions {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let interactions: Vec<Interaction> = reader
        .deserialize()
        .map(|x| x.unwrap())
        .map(|x: Interaction| Interaction::new(x.user_id(), x.item_id(), x.timestamp()))
        .take(100_000)
        .collect();

    Interactions::from(interactions)
}

fn fit(
    train: &TripletInteractions,
    hyper: factorization::Hyperparameters,
) -> factorization::ImplicitFactorizationModel {
    let mut model = factorization::ImplicitFactorizationModel::new(hyper);
    println!("loss {}", model.fit(train).unwrap());

    model
}

fn main() {
    let mut data = load_movielens("data.csv");
    // let mut data = load_goodbooks("ratings.csv");
    let mut rng = rand::thread_rng();

    let (mut train, test) = user_based_split(&mut data, &mut rng, 0.2);
    // let (mut train, test) = train_test_split(&mut data, &mut rng, 0.2);

    train.shuffle(&mut rng);

    for _ in 0..1000 {
        let mut results: Vec<Result> = File::open("factorization_results.json")
            .map(|file| serde_json::from_reader(&file).unwrap())
            .unwrap_or(Vec::new());

        let hyper = factorization::Hyperparameters::random(&mut rng);
        println!("Running {:#?}", &hyper);
        // let hyper = factorization::HyperparametersBuilder::default()
        //     .learning_rate(0.5)
        //     .fold_in_epochs(50)
        //     .latent_dim(32)
        //     .num_epochs(50)
        //     .l2_penalty(0.0)
        //     .build()
        //     .unwrap();

        println!("Users {} items {}", train.num_users(), train.num_items());

        let start = Instant::now();
        let model = fit(&train.to_triplet(), hyper.clone());
        let result = Result {
            train_mrr: mrr_score_train(&model, &train.to_compressed()).unwrap(),
            test_mrr: mrr_score(&model, &test.to_compressed()).unwrap(),
            elapsed: start.elapsed(),
            hyperparameters: hyper,
        };

        println!("{:#?}", result);

        if !result.test_mrr.is_nan() {
            results.push(result);
            //results.sort_by(|a, b| a.train_mrr.partial_cmp(&b.train_mrr).unwrap());
            results.sort_by(|a, b| a.test_mrr.partial_cmp(&b.test_mrr).unwrap());
        }

        println!("Best result: {:#?}", results.last());

        File::create("factorization_results.json")
            .map(|file| serde_json::to_writer_pretty(&file, &results).unwrap())
            .unwrap();
    }
}
