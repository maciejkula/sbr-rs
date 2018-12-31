#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate csv;
extern crate rand;
extern crate sbr;
extern crate serde;
extern crate serde_json;
extern crate wyrm;

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read};

use std::collections::HashSet;
use std::time::{Duration, Instant};

use sbr::data::{user_based_split, CompressedInteractions, Interaction, Interactions};
use sbr::evaluation::mrr_score;
use sbr::models::lstm;

#[derive(Deserialize, Serialize)]
struct GoodbooksInteraction {
    user_id: usize,
    book_id: usize,
    rating: usize,
}

fn load_goodbooks(path: &str) -> Interactions {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let mut interactions: Vec<Interaction> = reader
        .deserialize::<GoodbooksInteraction>()
        .map(|x| x.unwrap())
        .enumerate()
        .map(|(i, x)| Interaction::new(x.user_id, x.book_id, i))
        .collect();
    interactions.sort_by_key(|x| x.user_id());

    Interactions::from(interactions[..1_000_000].to_owned())
}

fn load_dummy() -> Interactions {
    let num_users = 100;
    let num_items = 50;

    let mut interactions = Vec::new();

    for user in 0..num_users {
        for item in 0..num_items {
            interactions.push(Interaction::new(user, 1000 + item, item));
        }
    }

    Interactions::from(interactions)
}

#[derive(Debug, Serialize, Deserialize)]
struct Result {
    test_mrr: f32,
    train_mrr: f32,
    elapsed: Duration,
    hyperparameters: lstm::Hyperparameters,
}

fn load_movielens(path: &str) -> Interactions {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let interactions: Vec<Interaction> = reader.deserialize().map(|x| x.unwrap()).collect();

    let interactions = rand::seq::sample_slice(&mut rand::thread_rng(), &interactions, 100000);

    Interactions::from(interactions)
}

fn fit(train: &CompressedInteractions, hyper: lstm::Hyperparameters) -> lstm::ImplicitLSTMModel {
    let mut model = hyper.build();
    model.fit(train).unwrap();

    model
}

fn main() {
    // let data = load_goodbooks("ratings.csv");
    let data = load_movielens("data.csv");
    // let mut data = load_dummy();
    let mut rng = rand::thread_rng();

    let (train, test) = user_based_split(&data, &mut rng, 0.2);

    let train = train.to_compressed();
    let test = test.to_compressed();

    println!(
        "Train {} {} {}",
        train.num_users(),
        train.num_items(),
        data.len()
    );

    for _ in 0..1000 {
        let mut results: Vec<Result> = File::open("lstm_results.json")
            .map(|file| serde_json::from_reader(&file).unwrap())
            .unwrap_or(Vec::new());

        let hyper = lstm::Hyperparameters::random(data.num_items(), &mut rng);

        println!("Running {:#?}", &hyper);
        let start = Instant::now();
        let model = fit(&train, hyper.clone());
        let result = Result {
            train_mrr: mrr_score(&model, &train).unwrap(),
            test_mrr: mrr_score(&model, &test).unwrap(),
            elapsed: start.elapsed(),
            hyperparameters: hyper,
        };

        println!("{:#?}", result);

        if !result.test_mrr.is_nan() {
            results.push(result);
            results.sort_by(|a, b| a.test_mrr.partial_cmp(&b.test_mrr).unwrap());
        }

        println!("Best result: {:#?}", results.last());

        File::create("lstm_results.json")
            .map(|file| serde_json::to_writer_pretty(&file, &results).unwrap())
            .unwrap();
    }
}
