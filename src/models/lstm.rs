//! Module for LSTM-based models.

use std::sync::Arc;

use rand;
use rand::distributions::{Distribution, Normal, Range, Uniform};
use rand::{Rng, SeedableRng, XorShiftRng};
use rayon;
use rayon::prelude::*;

use ndarray::Axis;

use wyrm;
use wyrm::nn;
use wyrm::optim::Optimizer as Optim;
use wyrm::{Arr, BoxedNode, DataInput, Variable};

use data::CompressedInteractions;
use {ItemId, OnlineRankingModel, PredictionError};

fn embedding_init<T: Rng>(rows: usize, cols: usize, rng: &mut T) -> wyrm::Arr {
    let normal = Normal::new(0.0, 1.0 / cols as f64);
    Arr::zeros((rows, cols)).map(|_| normal.sample(rng) as f32)
}

/// The loss used for training the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Loss {
    /// Bayesian Personalised Ranking.
    BPR,
    /// Pairwise hinge loss.
    Hinge,
}

/// Optimizer user to train the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Optimizer {
    /// Adagrad.
    Adagrad,
    /// Adam.
    Adam,
}

/// Type of parallelism used to train the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Parallelism {
    /// Multiple threads operate in parallel without any locking.
    Asynchronous,
    /// Multiple threads synchronise parameters between minibatches.
    Synchronous,
}

/// Hyperparameters for the [ImplicitLSTMModel].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hyperparameters {
    num_items: usize,
    max_sequence_length: usize,
    item_embedding_dim: usize,
    learning_rate: f32,
    l2_penalty: f32,
    loss: Loss,
    optimizer: Optimizer,
    parallelism: Parallelism,
    rng: XorShiftRng,
    num_threads: usize,
    num_epochs: usize,
}

impl Hyperparameters {
    /// Build new hyperparameters.
    pub fn new(num_items: usize, max_sequence_length: usize) -> Self {
        Hyperparameters {
            num_items: num_items,
            max_sequence_length: max_sequence_length,
            item_embedding_dim: 16,
            learning_rate: 0.01,
            l2_penalty: 0.0,
            loss: Loss::BPR,
            optimizer: Optimizer::Adam,
            parallelism: Parallelism::Asynchronous,
            rng: XorShiftRng::from_seed(rand::thread_rng().gen()),
            num_threads: rayon::current_num_threads(),
            num_epochs: 10,
        }
    }

    /// Set the learning rate.
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the l2 penalty.
    pub fn l2_penalty(mut self, l2_penalty: f32) -> Self {
        self.l2_penalty = l2_penalty;
        self
    }

    /// Set the embedding dimensionality.
    pub fn embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.item_embedding_dim = embedding_dim;
        self
    }

    /// Set the number of epochs to run per each `fit` call.
    pub fn num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    /// Set the loss function.
    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    /// Set number of threads to be used.
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Set the type of paralellism.
    pub fn parallelism(mut self, parallelism: Parallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    /// Set the random number generator.
    pub fn rng(mut self, rng: XorShiftRng) -> Self {
        self.rng = rng;
        self
    }

    /// Set the random number generator from seed.
    pub fn from_seed(mut self, seed: [u8; 16]) -> Self {
        self.rng = XorShiftRng::from_seed(seed);
        self
    }

    /// Set the optimizer type.
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set hyperparameters randomly: useful for hyperparameter search.
    pub fn random<R: Rng>(num_items: usize, rng: &mut R) -> Self {
        Hyperparameters {
            num_items: num_items,
            max_sequence_length: 2_usize.pow(Uniform::new(4, 8).sample(rng)),
            item_embedding_dim: 2_usize.pow(Uniform::new(4, 8).sample(rng)),
            learning_rate: (10.0_f32).powf(Uniform::new(-3.0, 0.5).sample(rng)),
            l2_penalty: (10.0_f32).powf(Uniform::new(-7.0, -3.0).sample(rng)),
            loss: if Uniform::new(0.0, 1.0).sample(rng) < 0.5 {
                Loss::BPR
            } else {
                Loss::Hinge
            },
            optimizer: if Uniform::new(0.0, 1.0).sample(rng) < 0.5 {
                Optimizer::Adam
            } else {
                Optimizer::Adagrad
            },
            parallelism: if Uniform::new(0.0, 1.0).sample(rng) < 0.5 {
                Parallelism::Asynchronous
            } else {
                Parallelism::Synchronous
            },
            rng: XorShiftRng::from_seed(rand::thread_rng().gen()),
            num_threads: Uniform::new(1, rayon::current_num_threads() + 1).sample(rng),
            num_epochs: 2_usize.pow(Uniform::new(3, 7).sample(rng)),
        }
    }

    fn build_params(&mut self) -> Parameters {
        let item_embeddings = Arc::new(wyrm::HogwildParameter::new(embedding_init(
            self.num_items,
            self.item_embedding_dim,
            &mut self.rng,
        )));

        let item_biases = Arc::new(wyrm::HogwildParameter::new(Arr::zeros((self.num_items, 1))));

        Parameters {
            item_embedding: item_embeddings,
            item_biases: item_biases,
            lstm: nn::lstm::Parameters::new(
                self.item_embedding_dim,
                self.item_embedding_dim,
                &mut self.rng,
            ),
        }
    }

    /// Build a model out of the chosen hyperparameters.
    pub fn build(mut self) -> ImplicitLSTMModel {
        let params = self.build_params();

        ImplicitLSTMModel {
            hyper: self,
            params: params,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
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
    fn build(&self, max_sequence_length: usize, loss: &Loss) -> Model {
        let item_embeddings = wyrm::ParameterNode::shared(self.item_embedding.clone());
        let item_biases = wyrm::ParameterNode::shared(self.item_biases.clone());

        let inputs: Vec<_> = (0..max_sequence_length)
            .map(|_| wyrm::IndexInputNode::new(&vec![0; 1]))
            .collect();
        let outputs: Vec<_> = (0..max_sequence_length)
            .map(|_| wyrm::IndexInputNode::new(&vec![0; 1]))
            .collect();
        let negatives: Vec<_> = (0..max_sequence_length)
            .map(|_| wyrm::IndexInputNode::new(&vec![0; 1]))
            .collect();

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
            .map(|(pos, neg)| match loss {
                Loss::BPR => (neg - pos).sigmoid().boxed(),
                Loss::Hinge => (1.0 + neg - pos).relu().boxed(),
            })
            .collect();

        let mut summed_losses = Vec::with_capacity(losses.len());
        summed_losses.push(losses[0].clone());

        for loss in &losses[1..] {
            let loss = (summed_losses.last().unwrap().clone() + loss.clone()).boxed();
            summed_losses.push(loss);
        }

        Model {
            inputs: inputs,
            outputs: outputs,
            negatives: negatives,
            hidden_states: hidden,
            losses: losses,
            summed_losses: summed_losses,
        }
    }
}

struct Model {
    inputs: Vec<Variable<wyrm::IndexInputNode>>,
    outputs: Vec<Variable<wyrm::IndexInputNode>>,
    negatives: Vec<Variable<wyrm::IndexInputNode>>,
    hidden_states: Vec<Variable<BoxedNode>>,
    losses: Vec<Variable<BoxedNode>>,
    summed_losses: Vec<Variable<BoxedNode>>,
}

/// An LSTM-based sequence model for implicit feedback.
#[derive(Debug, Clone)]
pub struct ImplicitLSTMModel {
    hyper: Hyperparameters,
    params: Parameters,
}

impl ImplicitLSTMModel {
    fn optimizer(
        &self,
        parameters: Vec<wyrm::Variable<wyrm::ParameterNode>>,
        barrier: &wyrm::optim::SynchronizationBarrier,
    ) -> Box<Optim> {
        match self.hyper.optimizer {
            Optimizer::Adagrad => Box::new({
                let opt = wyrm::optim::Adagrad::new(parameters)
                    .learning_rate(self.hyper.learning_rate)
                    .l2_penalty(self.hyper.l2_penalty);

                if self.hyper.parallelism == Parallelism::Synchronous {
                    opt.synchronized(barrier)
                } else {
                    opt
                }
            }) as Box<Optim>,
            Optimizer::Adam => Box::new({
                let opt = wyrm::optim::Adam::new(parameters)
                    .learning_rate(self.hyper.learning_rate)
                    .l2_penalty(self.hyper.l2_penalty);

                if self.hyper.parallelism == Parallelism::Synchronous {
                    opt.synchronized(barrier)
                } else {
                    opt
                }
            }) as Box<Optim>,
        }
    }
    /// Fit the model.
    ///
    /// Returns the loss value.
    pub fn fit(&mut self, interactions: &CompressedInteractions) -> Result<f32, &'static str> {
        let negative_item_range = Range::new(0, interactions.num_items());

        let mut subsequences: Vec<_> = interactions
            .iter_users()
            .flat_map(|user| {
                user.chunks(self.hyper.max_sequence_length)
                    .map(|(item_ids, _)| item_ids)
                    .filter(|item_ids| item_ids.len() > 2)
            })
            .collect();
        self.hyper.rng.shuffle(&mut subsequences);

        let num_chunks = subsequences.len() / self.hyper.num_threads;
        let mut partitions: Vec<_> = subsequences
            .chunks_mut(num_chunks)
            .map(|chunk| (chunk, XorShiftRng::from_seed(self.hyper.rng.gen())))
            .collect();

        let sync_barrier = wyrm::optim::SynchronizationBarrier::new();

        let loss = partitions
            .par_iter_mut()
            .map(|(partition, ref mut thread_rng)| {
                let mut model = self
                    .params
                    .build(self.hyper.max_sequence_length, &self.hyper.loss);
                let optimizer =
                    self.optimizer(model.losses.last().unwrap().parameters(), &sync_barrier);

                let mut loss_value = 0.0;
                let mut examples = 0;

                for _ in 0..self.hyper.num_epochs {
                    thread_rng.shuffle(partition);

                    for &item_ids in partition.iter() {
                        for (&input_idx, &output_idx, input, output, negative) in izip!(
                            item_ids,
                            item_ids.iter().skip(1),
                            &mut model.inputs,
                            &mut model.outputs,
                            &mut model.negatives
                        ) {
                            let negative_idx = negative_item_range.sample(thread_rng);

                            input.set_value(input_idx);
                            output.set_value(output_idx);
                            negative.set_value(negative_idx);
                        }

                        // Get the loss at the end of the sequence.
                        let loss_idx = item_ids.len().saturating_sub(2);
                        let loss = &mut model.summed_losses[loss_idx];

                        loss_value += loss.value().scalar_sum();
                        examples += loss_idx + 1;

                        loss.forward();
                        loss.backward(1.0);

                        optimizer.step();
                        loss.zero_gradient();
                    }
                }

                loss_value / (1.0 + examples as f32)
            })
            .sum();

        Ok(loss)
    }
}

/// The user representation used by the `ImplicitLSTM` model.
#[derive(Clone, Debug)]
pub struct ImplicitLSTMUser {
    user_embedding: Vec<f32>,
}

impl OnlineRankingModel for ImplicitLSTMModel {
    type UserRepresentation = ImplicitLSTMUser;
    fn user_representation(
        &self,
        item_ids: &[ItemId],
    ) -> Result<Self::UserRepresentation, PredictionError> {
        let model = self
            .params
            .build(self.hyper.max_sequence_length, &self.hyper.loss);

        let item_ids = &item_ids[item_ids
                                     .len()
                                     .saturating_sub(self.hyper.max_sequence_length)..];

        for (&input_idx, input) in izip!(item_ids, &model.inputs) {
            input.set_value(input_idx);
        }

        // Get the loss at the end of the sequence.
        let loss_idx = item_ids.len().saturating_sub(1);

        // Select the hidden state after ingesting all the inputs.
        let hidden_state = &model.hidden_states[loss_idx];

        // Run the network forward up to that point.
        hidden_state.forward();

        // Get the value.
        let representation = hidden_state.value();

        Ok(ImplicitLSTMUser {
            user_embedding: representation.as_slice().unwrap().to_owned(),
        })
    }
    fn predict(
        &self,
        user: &Self::UserRepresentation,
        item_ids: &[ItemId],
    ) -> Result<Vec<f32>, PredictionError> {
        let item_embeddings = &self.params.item_embedding;
        let item_biases = &self.params.item_biases;

        let embeddings = item_embeddings.value();
        let biases = item_biases.value();

        item_ids
            .iter()
            .map(|&item_idx| {
                let embedding = embeddings.subview(Axis(0), item_idx);
                let bias = biases[(item_idx, 0)];
                let dot = wyrm::simd_dot(&user.user_embedding, embedding.as_slice().unwrap());

                let prediction = bias + dot;

                if prediction.is_finite() {
                    Ok(prediction)
                } else {
                    Err(PredictionError::InvalidPredictionValue)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use csv;

    use super::*;
    use data::{user_based_split, Interaction, Interactions};
    use evaluation::mrr_score;

    fn load_movielens(path: &str) -> Interactions {
        let mut reader = csv::Reader::from_path(path).unwrap();
        let interactions: Vec<Interaction> = reader.deserialize().map(|x| x.unwrap()).collect();

        Interactions::from(interactions)
    }

    #[test]
    fn fold_in() {
        let mut data = load_movielens("data.csv");

        let mut rng = rand::XorShiftRng::from_seed([42; 16]);

        let (train, test) = user_based_split(&mut data, &mut rng, 0.2);
        let train_mat = train.to_compressed();
        let test_mat = test.to_compressed();

        println!("Train: {}, test: {}", train.len(), test.len());

        let mut model = Hyperparameters::new(data.num_items(), 128)
            .embedding_dim(32)
            .learning_rate(0.16)
            .l2_penalty(0.0004)
            .loss(Loss::Hinge)
            .optimizer(Optimizer::Adagrad)
            .num_epochs(10)
            .num_threads(1)
            .rng(rng)
            .build();

        let start = Instant::now();
        let loss = model.fit(&train_mat).unwrap();
        let elapsed = start.elapsed();
        let train_mrr = mrr_score(&model, &train_mat).unwrap();
        let test_mrr = mrr_score(&model, &test_mat).unwrap();

        // Results differ between different vector widths in MKL.
        let expected_mrr = if ::std::env::var("MKL_CBWR") == Ok("AVX".to_owned()) {
            0.091
        } else {
            0.102
        };

        println!(
            "Train MRR {} at loss {} and test MRR {} (in {:?})",
            train_mrr, loss, test_mrr, elapsed
        );

        assert!(test_mrr > expected_mrr)
    }
}
