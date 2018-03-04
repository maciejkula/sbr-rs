use std;
use std::rc::Rc;
use std::sync::Arc;

use rayon;
use rayon::prelude::*;
use rand;
use rand::distributions::{IndependentSample, Range};
use rand::{Rng, SeedableRng, XorShiftRng};

use ndarray::Axis;

use wyrm;
use wyrm::nn;
use wyrm::{Arr, BoxedNode, DataInput, Variable};

use super::super::{ItemId, OnlineRankingModel};
use data::{CompressedInteractions, TripletInteractions};

fn embedding_init<T: Rng>(rows: usize, cols: usize, rng: &mut T) -> wyrm::Arr {
    Arr::zeros((rows, cols)).map(|_| rng.gen::<f32>() / (cols as f32).sqrt())
}

#[derive(Builder, Clone, Debug)]
pub struct Hyperparameters {
    num_items: usize,
    max_sequence_length: usize,
    item_embedding_dim: usize,
    hidden_dim: usize,
    learning_rate: f32,
    rng: XorShiftRng,
    num_threads: usize,
    num_epochs: usize,
}

impl Hyperparameters {
    pub fn new(num_items: usize, max_sequence_length: usize) -> Self {
        Hyperparameters {
            num_items: num_items,
            max_sequence_length: max_sequence_length,
            item_embedding_dim: 16,
            hidden_dim: 16,
            learning_rate: 0.05,
            rng: XorShiftRng::from_seed(rand::thread_rng().gen()),
            num_threads: rayon::current_num_threads(),
            num_epochs: 1,
        }
    }

    fn build_params(&mut self) -> Parameters {
        let item_embeddings = Arc::new(wyrm::HogwildParameter::new(embedding_init(
            self.num_items,
            self.item_embedding_dim,
            &mut self.rng,
        )));

        let item_biases = Arc::new(wyrm::HogwildParameter::new(embedding_init(
            self.num_items,
            1,
            &mut self.rng,
        )));

        Parameters {
            item_embedding: item_embeddings,
            item_biases: item_biases,
            lstm: nn::lstm::Parameters::new(self.item_embedding_dim, self.hidden_dim),
        }
    }

    pub fn build(mut self) -> ImplicitLSTMModel {
        let params = self.build_params();

        ImplicitLSTMModel {
            hyper: self,
            params: params,
        }
    }
}

#[derive(Debug)]
struct Parameters {
    item_embedding: Arc<wyrm::HogwildParameter>,
    item_biases: Arc<wyrm::HogwildParameter>,
    lstm: nn::lstm::Parameters,
}

impl Clone for Parameters {
    fn clone(&self) -> Self {
        Parameters {
            item_embedding: Arc::new(self.item_embedding.as_ref().clone()),
            item_biases: Arc::new(self.item_biases.as_ref().clone()),
            lstm: self.lstm.clone(),
        }
    }
}

impl Parameters {
    fn build(&self, max_sequence_length: usize) -> Model {
        let item_embeddings = wyrm::ParameterNode::shared(self.item_embedding.clone());
        let item_biases = wyrm::ParameterNode::shared(self.item_biases.clone());

        let inputs = vec![wyrm::IndexInputNode::new(&vec![0; 1]); max_sequence_length];
        let outputs = vec![wyrm::IndexInputNode::new(&vec![0; 1]); max_sequence_length];
        let negatives = vec![wyrm::IndexInputNode::new(&vec![0; 1]); max_sequence_length];

        let input_embeddings: Vec<_> = inputs
            .iter()
            .map(|input| item_embeddings.index(input))
            .collect();
        let negative_embeddings: Vec<_> = negatives
            .iter()
            .map(|negative| item_embeddings.index(negative))
            .collect();
        let output_embeddings: Vec<_> = outputs
            .iter()
            .map(|output| item_embeddings.index(output))
            .collect();
        let output_biases: Vec<_> = outputs
            .iter()
            .map(|output| item_biases.index(output))
            .collect();
        let negative_biases: Vec<_> = negatives
            .iter()
            .map(|negative| item_biases.index(negative))
            .collect();

        let layer = self.lstm.build();
        let hidden = layer.forward(&input_embeddings);

        let positive_predictions: Vec<_> =
            izip!(hidden.iter(), output_embeddings.iter(), output_biases)
                .map(|(hidden_state, output_embedding, output_bias)| {
                    hidden_state.vector_dot(output_embedding) + output_bias
                })
                .collect();
        let negative_predictions: Vec<_> =
            izip!(hidden.iter(), negative_embeddings.iter(), negative_biases)
                .map(|(hidden_state, negative_embedding, negative_bias)| {
                    hidden_state.vector_dot(negative_embedding) + negative_bias
                })
                .collect();

        let losses: Vec<_> = positive_predictions
            .into_iter()
            .zip(negative_predictions.into_iter())
            .map(|(pos, neg)| (neg - pos).sigmoid().boxed())
            .collect();

        Model {
            inputs: inputs,
            outputs: outputs,
            negatives: negatives,
            losses: losses,
        }
    }
}

struct Model {
    inputs: Vec<Variable<wyrm::IndexInputNode>>,
    outputs: Vec<Variable<wyrm::IndexInputNode>>,
    negatives: Vec<Variable<wyrm::IndexInputNode>>,
    losses: Vec<Variable<BoxedNode>>,
}

#[derive(Debug, Clone)]
pub struct ImplicitLSTMModel {
    hyper: Hyperparameters,
    params: Parameters,
}

#[derive(Debug, Clone)]
pub struct ImplicitLSTMUser {
    user_embedding: Vec<f32>,
}

impl ImplicitLSTMModel {
    pub fn fit(&mut self, interactions: &CompressedInteractions) -> Result<f32, &'static str> {
        let negative_item_range = Range::new(0, interactions.num_items());
        let num_partitions = self.hyper.num_threads;
        //let seeds: Vec<[u32; 4]> = (0..num_partitions).map(|_| self.hyper.rng.gen()).collect();

        //let interactions: Vec<_> = interactions.iter_users().collect();

        // TODO: parallelism

        let mut model = self.params.build(self.hyper.max_sequence_length);
        let mut negatives = vec![0; self.hyper.max_sequence_length];
        let mut optimizer = wyrm::Adagrad::new(
            self.hyper.learning_rate,
            model.losses.last().unwrap().parameters(),
        );

        let mut loss_value = 0.0;

        for user in interactions.iter_users().filter(|user| !user.is_empty()) {
            // Cap item_ids to be at most `max_sequence_length` elements,
            // cutting off early parts of the sequence if necessary.
            let item_ids = &user.item_ids[user.item_ids
                                              .len()
                                              .saturating_sub(self.hyper.max_sequence_length)..];

            // Sample negatives.
            for neg_idx in &mut negatives[..user.len()] {
                *neg_idx = negative_item_range.ind_sample(&mut self.hyper.rng);
            }

            // Set all the inputs.
            for (&input_idx, &output_idx, &negative_idx, input, output, negative) in izip!(
                item_ids,
                &item_ids[1..],
                &negatives,
                &mut model.inputs,
                &mut model.outputs,
                &mut model.negatives
            ) {
                input.set_value(input_idx);
                output.set_value(output_idx);
                negative.set_value(negative_idx);
            }

            // Get the loss at the end of the sequence.
            let loss = &mut model.losses[item_ids.len() - 1];

            loss.forward();
            loss.backward(1.0);
            optimizer.step();

            loss_value += loss.value().scalar_sum();
        }

        Ok(loss_value)
    }
}

// impl OnlineRankingModel for ImplicitFactorizationModel {
//     type UserRepresentation = ImplicitFactorizationUser;
//     fn user_representation(
//         &self,
//         item_ids: &[ItemId],
//     ) -> Result<Self::UserRepresentation, &'static str> {
//         let embedding = self.fold_in_user(item_ids)?;
//         Ok(ImplicitFactorizationUser {
//             user_embedding: embedding,
//         })
//     }
//     fn predict(
//         &self,
//         user: &Self::UserRepresentation,
//         item_ids: &[ItemId],
//     ) -> Result<Vec<f32>, &'static str> {
//         self.predict_user(&user.user_embedding, item_ids)
//     }
// }

// impl ImplicitFactorizationModel {
//     pub fn new(hyper: Hyperparameters) -> Self {
//         ImplicitFactorizationModel {
//             hyper: hyper,
//             model: None,
//         }
//     }

//     pub fn num_users(&self) -> Option<usize> {
//         match self.model {
//             Some(ref model) => Some(model.num_users),
//             _ => None,
//         }
//     }

//     pub fn num_items(&self) -> Option<usize> {
//         match self.model {
//             Some(ref model) => Some(model.num_items),
//             _ => None,
//         }
//     }

//     fn predict_user(
//         &self,
//         user_embedding: &[f32],
//         item_ids: &[usize],
//     ) -> Result<Vec<f32>, &'static str> {
//         if let Some(ref model) = self.model {
//             let item_embeddings = &model.item_embedding;
//             let item_biases = &model.item_biases;

//             let embeddings = item_embeddings.value();
//             let biases = item_biases.value();

//             let predictions: Vec<f32> = item_ids
//                 .iter()
//                 .map(|&item_idx| {
//                     let embedding = embeddings.subview(Axis(0), item_idx);
//                     let bias = biases[(item_idx, 0)];

//                     bias + wyrm::simd_dot(user_embedding, embedding.as_slice().unwrap())
//                 })
//                 .collect();

//             Ok(predictions)
//         } else {
//             Err("Model must be fitted first.")
//         }
//     }

//     fn build_model(&mut self, num_users: usize, num_items: usize) -> Parameters {
//         let user_embeddings = Arc::new(wyrm::HogwildParameter::new(embedding_init(
//             num_users,
//             self.hyper.latent_dim,
//             &mut self.hyper.rng,
//         )));
//         let item_embeddings = Arc::new(wyrm::HogwildParameter::new(embedding_init(
//             num_items,
//             self.hyper.latent_dim,
//             &mut self.hyper.rng,
//         )));

//         let item_biases = Arc::new(wyrm::HogwildParameter::new(embedding_init(
//             num_items,
//             1,
//             &mut self.hyper.rng,
//         )));

//         Parameters {
//             num_users: num_users,
//             num_items: num_items,
//             user_embedding: user_embeddings,
//             item_embedding: item_embeddings,
//             item_biases: item_biases,
//         }
//     }

//     fn fold_in_user(&self, interactions: &[ItemId]) -> Result<Vec<f32>, &'static str> {
//         if self.model.is_none() {
//             return Err("Model must be fitted before trying to fold-in users.");
//         }

//         let negative_item_range = Range::new(0, self.model.as_ref().unwrap().num_items);
//         let minibatch_size = 1;
//         let mut rng = rand::XorShiftRng::from_seed([42; 4]);

//         let user_vector = Arc::new(wyrm::HogwildParameter::new(embedding_init(
//             1,
//             self.hyper.latent_dim,
//             &mut rng,
//         )));

//         let mut model = self.model
//             .as_ref()
//             .unwrap()
//             .build(user_vector.clone(), minibatch_size);
//         let mut optimizer =
//             wyrm::Adagrad::new(self.hyper.learning_rate, vec![model.user_embeddings]);

//         model.user_idx.set_value(0);

//         for _ in 0..self.hyper.fold_in_epochs {
//             for &item_id in interactions {
//                 model.positive_item_idx.set_value(item_id);
//                 model
//                     .negative_item_idx
//                     .set_value(negative_item_range.ind_sample(&mut rng));

//                 model.loss.forward();
//                 model.loss.backward(1.0);

//                 optimizer.step();
//                 model.loss.zero_gradient();
//             }
//         }

//         let user_vec = user_vector.value();

//         Ok(user_vec.as_slice().unwrap().to_owned())
//     }

//     pub fn fit(
//         &mut self,
//         interactions: &TripletInteractions,
//         num_epochs: usize,
//     ) -> Result<f32, &'static str> {
//         let minibatch_size = self.hyper.minibatch_size;

//         if self.model.is_none() {
//             self.model = Some(self.build_model(interactions.num_users(), interactions.num_items()));
//         }

//         let negative_item_range = Range::new(0, interactions.num_items());
//         let num_partitions = self.hyper.num_threads;
//         let seeds: Vec<[u32; 4]> = (0..num_partitions).map(|_| self.hyper.rng.gen()).collect();

//         let partitions: Vec<_> = interactions
//             .iter_minibatch_partitioned(minibatch_size, num_partitions)
//             .into_iter()
//             .zip(seeds.into_iter())
//             .collect();

//         let losses: Vec<f32> = partitions
//             .into_par_iter()
//             .map(|(data, seed)| {
//                 let mut model = self.model.as_ref().unwrap().build(
//                     self.model.as_ref().unwrap().user_embedding.clone(),
//                     minibatch_size,
//                 );

//                 let mut optimizer =
//                     wyrm::Adagrad::new(self.hyper.learning_rate, model.loss.parameters());

//                 let mut batch_negatives = vec![0; minibatch_size];

//                 let mut rng = rand::XorShiftRng::from_seed(seed);
//                 let mut loss_value = 0.0;

//                 let mut num_observations = 0;

//                 for _ in 0..num_epochs {
//                     for batch in data.clone() {
//                         if batch.len() < minibatch_size {
//                             break;
//                         }

//                         num_observations += batch.len();

//                         for negative in &mut batch_negatives {
//                             *negative = negative_item_range.ind_sample(&mut rng);
//                         }

//                         model.user_idx.set_value(batch.user_ids);
//                         model.positive_item_idx.set_value(batch.item_ids);
//                         model
//                             .negative_item_idx
//                             .set_value(batch_negatives.as_slice());

//                         model.loss.forward();
//                         model.loss.backward(1.0);

//                         loss_value += model.loss.value().scalar_sum();

//                         optimizer.step();
//                         model.loss.zero_gradient();
//                     }
//                 }

//                 loss_value / num_observations as f32
//             })
//             .collect();

//         Ok(losses.into_iter().sum())
//     }
// }

// #[cfg(test)]
// mod tests {

//     use csv;

//     use super::*;
//     use data::{user_based_split, Interaction, Interactions};
//     use evaluation::mrr_score;
//     use test::Bencher;

//     fn load_movielens(path: &str) -> Interactions {
//         let mut reader = csv::Reader::from_path(path).unwrap();
//         let interactions: Vec<Interaction> = reader.deserialize().map(|x| x.unwrap()).collect();

//         Interactions::from(interactions)
//     }

//     #[test]
//     fn fold_in() {
//         let mut data = load_movielens("data.csv");

//         let mut rng = rand::XorShiftRng::from_seed([42; 4]);

//         let (train, test) = user_based_split(&mut data, &mut rng, 0.2);

//         println!("Train: {}, test: {}", train.len(), test.len());

//         let hyper = HyperparametersBuilder::default()
//             .learning_rate(0.5)
//             .fold_in_epochs(50)
//             .latent_dim(32)
//             .num_threads(1)
//             .rng(rng)
//             .build()
//             .unwrap();

//         let num_epochs = 50;

//         let mut model = ImplicitFactorizationModel::new(hyper);

//         println!(
//             "Loss: {}",
//             model.fit(&train.to_triplet(), num_epochs).unwrap()
//         );

//         let test_mat = test.to_compressed();

//         let mrr = mrr_score(&model, &test_mat).unwrap();

//         println!("MRR {}", mrr);

//         assert!(mrr > 0.065);
//     }

//     #[bench]
//     fn bench_movielens(b: &mut Bencher) {
//         let data = load_movielens("data.csv");
//         let num_epochs = 2;

//         let mut model = ImplicitFactorizationModel::default();

//         let data = data.to_triplet();

//         model.fit(&data, num_epochs).unwrap();

//         b.iter(|| {
//             model.fit(&data, num_epochs).unwrap();
//         });
//     }

//     // #[bench]
//     // fn bench_movielens_10m(b: &mut Bencher) {
//     //     let data = load_movielens("/home/maciej/Downloads/data.csv");
//     //     //let data = load_movielens("data.csv");
//     //     let num_epochs = 1;

//     //     let mut model = ImplicitFactorizationModel::default();
//     //     println!("Num obs {}", data.len());

//     //     model.fit(&data, 1).unwrap();

//     //     let mut runs = 0;
//     //     let mut elapsed = std::time::Duration::default();

//     //     b.iter(|| {
//     //         let start = std::time::Instant::now();
//     //         println!("Loss: {}", model.fit(&data, num_epochs).unwrap());
//     //         elapsed += start.elapsed();
//     //         runs += 1;
//     //         println!("Avg duration: {:#?}", elapsed / runs);
//     //     });
//     // }
// }
