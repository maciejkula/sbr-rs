#[macro_use]
extern crate serde_derive;
extern crate serde_json;

#[macro_use]
extern crate itertools;

extern crate csv;
#[macro_use]
extern crate failure;
extern crate ndarray;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate siphasher;

extern crate wyrm;

pub mod data;
pub mod evaluation;
pub mod models;

pub type UserId = usize;
pub type ItemId = usize;
pub type Timestamp = usize;

#[derive(Debug, Fail)]
pub enum PredictionError {
    #[fail(display = "Invalid prediction value: non-finite or not a number.")]
    InvalidPredictionValue,
}

pub trait OnlineRankingModel {
    type UserRepresentation: std::fmt::Debug;
    fn user_representation(
        &self,
        item_ids: &[ItemId],
    ) -> Result<Self::UserRepresentation, PredictionError>;
    fn predict(
        &self,
        user: &Self::UserRepresentation,
        item_ids: &[ItemId],
    ) -> Result<Vec<f32>, PredictionError>;
}
