//! Model containing evaluation functions.
use std;

use rayon::prelude::*;

use crate::data::CompressedInteractions;
use crate::{OnlineRankingModel, PredictionError};

/// Compute the MRR (mean reciprocal rank) of predictions for the last
/// item in `test` sequences, treating all but the last one item as inputs
/// in computing the user representation.
pub fn mrr_score<T: OnlineRankingModel + Sync>(
    model: &T,
    test: &CompressedInteractions,
) -> Result<f32, PredictionError> {
    let item_ids: Vec<usize> = (0..test.num_items()).collect();

    let mrrs = test
        .iter_users()
        .filter(|user| user.item_ids.len() >= 2)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|test_user| {
            let train_items = &test_user.item_ids[..test_user.item_ids.len().saturating_sub(1)];
            let test_item = *test_user.item_ids.last().unwrap();

            let user_embedding = model.user_representation(train_items).unwrap();
            let mut predictions = model.predict(&user_embedding, &item_ids)?;

            for &train_item_id in train_items {
                predictions[train_item_id] = std::f32::MIN;
            }

            let test_score = predictions[test_item];
            let mut rank = 0;

            for &prediction in &predictions {
                if prediction >= test_score {
                    rank += 1;
                }
            }

            Ok(1.0 / rank as f32)
        })
        .collect::<Result<Vec<f32>, PredictionError>>()?;

    Ok(mrrs.iter().sum::<f32>() / mrrs.len() as f32)
}
