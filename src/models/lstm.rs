// use std;
// use std::rc::Rc;
use std::sync::Arc;

use rand;
use rand::distributions::{Exp, IndependentSample, Normal, Range, Sample};
use rand::{Rng, SeedableRng, XorShiftRng};
use rayon;
use rayon::prelude::*;

use ndarray::Axis;

use wyrm;
use wyrm::nn;
use wyrm::{Arr, BoxedNode, DataInput, Variable};

use super::super::{ItemId, OnlineRankingModel};
use data::CompressedInteractions;

fn embedding_init<T: Rng>(rows: usize, cols: usize, rng: &mut T) -> wyrm::Arr {
    let normal = Normal::new(0.0, 1.0 / cols as f64);
    Arr::zeros((rows, cols)).map(|_| normal.ind_sample(rng) as f32)
}

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
pub struct Hyperparameters {
    num_items: usize,
    max_sequence_length: usize,
    item_embedding_dim: usize,
    learning_rate: f32,
    l2_penalty: f32,
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
            learning_rate: 0.01,
            l2_penalty: 0.0,
            rng: XorShiftRng::from_seed(rand::thread_rng().gen()),
            num_threads: rayon::current_num_threads(),
            num_epochs: 10,
        }
    }

    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn random<R: Rng>(num_items: usize, rng: &mut R) -> Self {
        Hyperparameters {
            num_items: num_items,
            max_sequence_length: Range::new(2, 40).ind_sample(rng),
            item_embedding_dim: Range::new(4, 64).ind_sample(rng),
            learning_rate: Exp::new(0.5e1).sample(rng) as f32,
            l2_penalty: Exp::new(1e8).sample(rng) as f32,
            rng: XorShiftRng::from_seed(rand::thread_rng().gen()),
            num_threads: rayon::current_num_threads(),
            num_epochs: Range::new(1, 64).ind_sample(rng),
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
            lstm: nn::lstm::Parameters::new(self.item_embedding_dim, self.item_embedding_dim),
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

#[derive(Debug, Clone)]
pub struct ImplicitLSTMModel {
    hyper: Hyperparameters,
    params: Parameters,
}

impl ImplicitLSTMModel {
    pub fn fit(&mut self, interactions: &CompressedInteractions) -> Result<f32, &'static str> {
        let negative_item_range = Range::new(0, interactions.num_items());
        // TODO: parallelism

        let mut model = self.params.build(self.hyper.max_sequence_length);
        let mut negatives = vec![0; self.hyper.max_sequence_length];
        let mut optimizer = wyrm::Adagrad::new(
            self.hyper.learning_rate,
            model.losses.last().unwrap().parameters(),
        ).l2_penalty(self.hyper.l2_penalty)
            .clamp(-5.0, 5.0);

        let mut loss_value = 0.0;

        let mut data: Vec<_> = interactions
            .iter_users()
            .filter(|user| !user.is_empty())
            .collect();

        let data: Vec<_> = data.chunks(data.len() / self.hyper.num_threads).collect();

        data.par_iter().for_each(|data| {
            let mut model = self.params.build(self.hyper.max_sequence_length);
            let mut optimizer = wyrm::Adagrad::new(
                self.hyper.learning_rate,
                model.losses.last().unwrap().parameters(),
            ).l2_penalty(self.hyper.l2_penalty)
                .clamp(-5.0, 5.0);

            for _ in 0..self.hyper.num_epochs {
                for user in data.iter() {
                    // Cap item_ids to be at most `max_sequence_length` elements,
                    // cutting off early parts of the sequence if necessary.
                    let item_ids = &user.item_ids[user.item_ids.len().saturating_sub(
                        self.hyper.max_sequence_length,
                    )..];

                    if item_ids.len() < 2 {
                        continue;
                    }

                    for (i, &input_idx, &output_idx, input, output, negative) in izip!(
                        0..item_ids.len(),
                        item_ids,
                        &item_ids[1..],
                        &mut model.inputs,
                        &mut model.outputs,
                        &mut model.negatives
                    ) {
                        let negative_idx = negative_item_range.ind_sample(&mut rand::thread_rng());
                        input.set_value(input_idx);
                        output.set_value(output_idx);
                        negative.set_value(negative_idx);
                    }

                    // Get the loss at the end of the sequence.
                    let loss_idx = item_ids.len().saturating_sub(2);
                    let loss = &mut model.summed_losses[loss_idx];

                    loss.forward();
                    loss.backward(1.0 / (loss_idx + 1) as f32);
                    optimizer.step();
                    loss.zero_gradient();
                }
            }
        });

        Ok(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct ImplicitLSTMUser {
    user_embedding: Vec<f32>,
}

impl OnlineRankingModel for ImplicitLSTMModel {
    type UserRepresentation = ImplicitLSTMUser;
    fn user_representation(
        &self,
        item_ids: &[ItemId],
    ) -> Result<Self::UserRepresentation, &'static str> {
        let model = self.params.build(self.hyper.max_sequence_length);
        let item_ids = &item_ids[item_ids
                                     .len()
                                     .saturating_sub(self.hyper.max_sequence_length)..];

        for (&input_idx, input) in izip!(item_ids, &model.inputs) {
            input.set_value(input_idx);
        }

        // Select the hidden state after ingesting all the inputs.
        let hidden_state = &model.hidden_states[item_ids.len() - 1];

        // Run the network forward up to that point. Remember to reset
        // the cached state.
        hidden_state.zero_gradient();
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
    ) -> Result<Vec<f32>, &'static str> {
        let item_embeddings = &self.params.item_embedding;
        let item_biases = &self.params.item_biases;

        let embeddings = item_embeddings.value();
        let biases = item_biases.value();

        let predictions: Vec<f32> = item_ids
            .iter()
            .map(|&item_idx| {
                let embedding = embeddings.subview(Axis(0), item_idx);
                let bias = biases[(item_idx, 0)];

                bias + wyrm::simd_dot(&user.user_embedding, embedding.as_slice().unwrap())
            })
            .collect();

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {

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

        let (train, test) = user_based_split(&mut data, &mut rand::thread_rng(), 0.2);

        println!("Train: {}, test: {}", train.len(), test.len());

        let mut model = Hyperparameters::new(train.num_items(), 10)
            .learning_rate(0.1)
            .build();

        let num_epochs = 300;
        let all_mat = data.to_compressed();
        let train_mat = train.to_compressed();
        let test_mat = test.to_compressed();

        for _ in 0..num_epochs {
            println!("Loss: {}", model.fit(&train_mat).unwrap());
            let mrr = mrr_score(&model, &test_mat).unwrap();
            println!("Test MRR {}", mrr);
            let mrr = mrr_score(&model, &train_mat).unwrap();
            println!("Train MRR {}", mrr);
        }
        let mrr = mrr_score(&model, &train.to_compressed()).unwrap();
        println!("MRR {}", mrr);
        let mrr = mrr_score(&model, &test.to_compressed()).unwrap();
        println!("MRR {}", mrr);

        //assert!(mrr > 0.065);
    }

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
}
