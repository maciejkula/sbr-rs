use std;
use std::rc::Rc;
use std::sync::Arc;

use rand;
use rand::distributions::{Exp, IndependentSample, Normal, Range, Sample};
use rand::{Rng, SeedableRng, XorShiftRng};
use rayon;
use rayon::prelude::*;

use ndarray::Axis;

use wyrm;
use wyrm::{Arr, DataInput};

use super::super::{ItemId, OnlineRankingModel};
use data::TripletInteractions;

fn embedding_init<T: Rng>(rows: usize, cols: usize, rng: &mut T) -> wyrm::Arr {
    let normal = Normal::new(0.0, 1.0 / cols as f64);
    Arr::zeros((rows, cols)).map(|_| normal.ind_sample(rng) as f32)
    // let mut range = Range::new(-0.5, 0.5);
    // Arr::zeros((rows, cols)).map(|_| range.sample(rng) / (cols as f32).sqrt())
}

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
pub struct Hyperparameters {
    #[builder(default = "16")]
    latent_dim: usize,
    #[builder(default = "10")]
    minibatch_size: usize,
    #[builder(default = "0.01")]
    learning_rate: f32,
    #[builder(default = "0.0")]
    l2_penalty: f32,
    #[builder(default = "50")]
    fold_in_epochs: usize,
    #[builder(default = "50")]
    num_epochs: usize,
    #[builder(default = "XorShiftRng::from_seed(rand::thread_rng().gen())")]
    rng: XorShiftRng,
    #[builder(default = "rayon::current_num_threads()")]
    num_threads: usize,
}

impl Hyperparameters {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        Hyperparameters {
            latent_dim: Range::new(4, 64).ind_sample(rng),
            minibatch_size: Range::new(4, 64).ind_sample(rng),
            learning_rate: Exp::new(0.5e1).sample(rng) as f32,
            l2_penalty: Exp::new(1e8).sample(rng) as f32,
            fold_in_epochs: Range::new(16, 64).ind_sample(rng),
            num_epochs: Range::new(16, 64).ind_sample(rng),
            rng: XorShiftRng::from_seed(rand::thread_rng().gen()),
            num_threads: rayon::current_num_threads(),
        }
    }

    pub fn build(self) -> ImplicitFactorizationModel {
        ImplicitFactorizationModel {
            hyper: self,
            model: None,
        }
    }
}

#[derive(Debug)]
struct Parameters {
    num_users: usize,
    num_items: usize,
    user_embedding: Arc<wyrm::HogwildParameter>,
    item_embedding: Arc<wyrm::HogwildParameter>,
    item_biases: Arc<wyrm::HogwildParameter>,
}

impl Parameters {
    fn build(&self, user_embedding: Arc<wyrm::HogwildParameter>, minibatch_size: usize) -> Model {
        let user_embeddings = wyrm::ParameterNode::shared(user_embedding);
        let item_embeddings = wyrm::ParameterNode::shared(self.item_embedding.clone());
        let item_biases = wyrm::ParameterNode::shared(self.item_biases.clone());

        let user_idx = wyrm::IndexInputNode::new(&vec![0; minibatch_size]);
        let positive_item_idx = wyrm::IndexInputNode::new(&vec![0; minibatch_size]);
        let negative_item_idx = wyrm::IndexInputNode::new(&vec![0; minibatch_size]);

        let user_vector = user_embeddings.index(&user_idx);
        let positive_item_vector = item_embeddings.index(&positive_item_idx);
        let negative_item_vector = item_embeddings.index(&negative_item_idx);
        let positive_item_bias = item_biases.index(&positive_item_idx);
        let negative_item_bias = item_biases.index(&negative_item_idx);

        let positive_prediction =
            user_vector.vector_dot(&positive_item_vector) + positive_item_bias;
        let negative_prediciton =
            user_vector.vector_dot(&negative_item_vector) + negative_item_bias;

        let score_diff = positive_prediction - negative_prediciton;
        let loss = 1.0 - score_diff.sigmoid();

        Model {
            user_idx: user_idx,
            user_embeddings: user_embeddings,
            positive_item_idx: positive_item_idx,
            negative_item_idx: negative_item_idx,
            loss: loss.boxed(),
        }
    }
}

struct Model {
    user_idx: wyrm::Variable<wyrm::IndexInputNode>,
    user_embeddings: wyrm::Variable<wyrm::ParameterNode>,
    positive_item_idx: wyrm::Variable<wyrm::IndexInputNode>,
    negative_item_idx: wyrm::Variable<wyrm::IndexInputNode>,
    loss: wyrm::Variable<Rc<wyrm::Node<Value = Arr, InputGradient = Arr>>>,
}

#[derive(Debug)]
pub struct ImplicitFactorizationModel {
    hyper: Hyperparameters,
    model: Option<Parameters>,
}

impl std::default::Default for ImplicitFactorizationModel {
    fn default() -> Self {
        ImplicitFactorizationModel {
            hyper: HyperparametersBuilder::default().build().unwrap(),
            model: None,
        }
    }
}

#[derive(Debug)]
pub struct ImplicitFactorizationUser {
    user_embedding: Vec<f32>,
}

impl OnlineRankingModel for ImplicitFactorizationModel {
    type UserRepresentation = ImplicitFactorizationUser;
    fn user_representation(
        &self,
        item_ids: &[ItemId],
    ) -> Result<Self::UserRepresentation, &'static str> {
        let embedding = self.fold_in_user(item_ids)?;
        Ok(ImplicitFactorizationUser {
            user_embedding: embedding,
        })
    }
    fn predict(
        &self,
        user: &Self::UserRepresentation,
        item_ids: &[ItemId],
    ) -> Result<Vec<f32>, &'static str> {
        self.predict_user(&user.user_embedding, item_ids)
    }
}

impl ImplicitFactorizationModel {
    pub fn new(hyper: Hyperparameters) -> Self {
        ImplicitFactorizationModel {
            hyper: hyper,
            model: None,
        }
    }

    pub fn num_users(&self) -> Option<usize> {
        match self.model {
            Some(ref model) => Some(model.num_users),
            _ => None,
        }
    }

    pub fn num_items(&self) -> Option<usize> {
        match self.model {
            Some(ref model) => Some(model.num_items),
            _ => None,
        }
    }

    fn predict_user(
        &self,
        user_embedding: &[f32],
        item_ids: &[usize],
    ) -> Result<Vec<f32>, &'static str> {
        if let Some(ref model) = self.model {
            let item_embeddings = &model.item_embedding;
            let item_biases = &model.item_biases;

            let embeddings = item_embeddings.value();
            let biases = item_biases.value();

            let predictions: Vec<f32> = item_ids
                .iter()
                .map(|&item_idx| {
                    let embedding = embeddings.subview(Axis(0), item_idx);
                    let bias = biases[(item_idx, 0)];

                    bias + wyrm::simd_dot(user_embedding, embedding.as_slice().unwrap())
                })
                .collect();

            Ok(predictions)
        } else {
            Err("Model must be fitted first.")
        }
    }

    fn build_model(&mut self, num_users: usize, num_items: usize) -> Parameters {
        let user_embeddings = Arc::new(wyrm::HogwildParameter::new(embedding_init(
            num_users,
            self.hyper.latent_dim,
            &mut self.hyper.rng,
        )));
        let item_embeddings = Arc::new(wyrm::HogwildParameter::new(embedding_init(
            num_items,
            self.hyper.latent_dim,
            &mut self.hyper.rng,
        )));

        let item_biases = Arc::new(wyrm::HogwildParameter::new(Arr::zeros((num_items, 1))));

        Parameters {
            num_users: num_users,
            num_items: num_items,
            user_embedding: user_embeddings,
            item_embedding: item_embeddings,
            item_biases: item_biases,
        }
    }

    fn fold_in_user(&self, interactions: &[ItemId]) -> Result<Vec<f32>, &'static str> {
        if self.model.is_none() {
            return Err("Model must be fitted before trying to fold-in users.");
        }

        let negative_item_range = Range::new(0, self.model.as_ref().unwrap().num_items);
        let minibatch_size = 1;
        let mut rng = rand::XorShiftRng::from_seed(rand::thread_rng().gen());

        let user_vector = Arc::new(wyrm::HogwildParameter::new(embedding_init(
            1,
            self.hyper.latent_dim,
            &mut rng,
        )));

        let mut model = self.model
            .as_ref()
            .unwrap()
            .build(user_vector.clone(), minibatch_size);
        let mut optimizer =
            wyrm::Adagrad::new(self.hyper.learning_rate, vec![model.user_embeddings])
                .l2_penalty(self.hyper.l2_penalty);

        model.user_idx.set_value(0);

        // shuffle
        // println!("New user ---------------------------");
        let mut interactions = interactions.to_owned();

        for _ in 0..self.hyper.fold_in_epochs {
            let mut loss = 0.0;
            rng.shuffle(&mut interactions);
            for &item_id in &interactions {
                model.positive_item_idx.set_value(item_id);
                model
                    .negative_item_idx
                    .set_value(negative_item_range.ind_sample(&mut rng));

                model.loss.forward();
                model.loss.backward(1.0);

                loss += model.loss.value().scalar_sum();

                optimizer.step();
                model.loss.zero_gradient();
            }

            // println!("fold in loss {}", loss / interactions.len() as f32);
        }

        let user_vec = user_vector.value();

        Ok(user_vec.as_slice().unwrap().to_owned())
    }

    pub fn fit(&mut self, interactions: &TripletInteractions) -> Result<f32, &'static str> {
        let minibatch_size = self.hyper.minibatch_size;

        if self.model.is_none() {
            self.model = Some(self.build_model(interactions.num_users(), interactions.num_items()));
        }

        let negative_item_range = Range::new(0, interactions.num_items());
        let num_partitions = self.hyper.num_threads;
        let seeds: Vec<[u8; 16]> = (0..num_partitions).map(|_| self.hyper.rng.gen()).collect();

        let mut partitions: Vec<_> = interactions
            .iter_minibatch_partitioned(minibatch_size, num_partitions)
            .into_iter()
            .zip(seeds.into_iter())
            .collect();

        let losses: Vec<f32> = partitions
            .into_par_iter()
            .map(|(data, seed)| {
                let mut model = self.model.as_ref().unwrap().build(
                    self.model.as_ref().unwrap().user_embedding.clone(),
                    minibatch_size,
                );

                let mut optimizer =
                    wyrm::Adagrad::new(self.hyper.learning_rate, model.loss.parameters())
                        .l2_penalty(self.hyper.l2_penalty);

                let mut batch_negatives = vec![0; minibatch_size];

                let mut rng = rand::XorShiftRng::from_seed(seed);
                let mut loss_value = 0.0;

                let mut num_observations = 0;

                for _ in 0..self.hyper.num_epochs {
                    for batch in data.clone() {
                        if batch.len() < minibatch_size {
                            break;
                        }

                        num_observations += batch.len();

                        for negative in &mut batch_negatives {
                            *negative = negative_item_range.ind_sample(&mut rng);
                        }

                        model.user_idx.set_value(batch.user_ids);
                        model.positive_item_idx.set_value(batch.item_ids);
                        model
                            .negative_item_idx
                            .set_value(batch_negatives.as_slice());

                        model.loss.forward();
                        model.loss.backward(1.0 / (batch.len() as f32));

                        loss_value += model.loss.value().scalar_sum();

                        optimizer.step();
                        model.loss.zero_gradient();
                    }

                    println!("loss {}", loss_value / num_observations as f32);
                    loss_value = 0.0;
                }

                loss_value / num_observations as f32
            })
            .collect();

        Ok(losses.into_iter().sum())
    }
}

#[cfg(test)]
mod tests {

    // use csv;

    // use super::*;
    // use data::{user_based_split, Interaction, Interactions};
    // use evaluation::mrr_score;
    // use test::Bencher;

    // fn load_movielens(path: &str) -> Interactions {
    //     let mut reader = csv::Reader::from_path(path).unwrap();
    //     let interactions: Vec<Interaction> = reader.deserialize().map(|x| x.unwrap()).collect();

    //     Interactions::from(interactions)
    // }

    // #[test]
    // fn fold_in() {
    //     let mut data = load_movielens("data.csv");

    //     let mut rng = rand::XorShiftRng::from_seed([42; 16]);

    //     let (train, test) = user_based_split(&mut data, &mut rng, 0.2);
    //     let train_mat = train.to_compressed();
    //     let test_mat = test.to_compressed();

    //     let hyper = HyperparametersBuilder::default()
    //         .learning_rate(0.5)
    //         .fold_in_epochs(50)
    //         .latent_dim(32)
    //         .num_threads(1)
    //         .rng(rng)
    //         .build()
    //         .unwrap();

    //     let num_epochs = 50;

    //     let mut model = ImplicitFactorizationModel::new(hyper);
    //     let train_triplet = train.to_triplet();

    //     for _ in 0..num_epochs {
    //         println!("Loss: {}", model.fit(&train_triplet, 1).unwrap());
    //         let mrr = mrr_score(&model, &test_mat).unwrap();
    //         println!("Test MRR {}", mrr);
    //         let mrr = mrr_score(&model, &train_mat).unwrap();
    //         println!("Train MRR {}", mrr);
    //     }

    //     let mrr = mrr_score(&model, &test_mat).unwrap();

    //     println!("MRR {}", mrr);

    //     assert!(mrr > 0.065);
    // }

    // #[bench]
    // fn bench_movielens(b: &mut Bencher) {
    //     let data = load_movielens("data.csv");
    //     let num_epochs = 2;

    //     let mut model = ImplicitFactorizationModel::default();

    //     let data = data.to_triplet();

    //     model.fit(&data).unwrap();

    //     b.iter(|| {
    //         model.fit(&data, num_epochs).unwrap();
    //     });
    // }

    // // #[bench]
    // // fn bench_movielens_10m(b: &mut Bencher) {
    // //     let data = load_movielens("/home/maciej/Downloads/data.csv");
    // //     //let data = load_movielens("data.csv");
    // //     let num_epochs = 1;

    // //     let mut model = ImplicitFactorizationModel::default();
    // //     println!("Num obs {}", data.len());

    // //     model.fit(&data, 1).unwrap();

    // //     let mut runs = 0;
    // //     let mut elapsed = std::time::Duration::default();

    // //     b.iter(|| {
    // //         let start = std::time::Instant::now();
    // //         println!("Loss: {}", model.fit(&data, num_epochs).unwrap());
    // //         elapsed += start.elapsed();
    // //         runs += 1;
    // //         println!("Avg duration: {:#?}", elapsed / runs);
    // //     });
    // // }
}
