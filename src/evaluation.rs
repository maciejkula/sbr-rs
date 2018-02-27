use std;

use super::OnlineRankingModel;
use data::CompressedInteractions;

pub fn mrr_score<T: OnlineRankingModel>(
    model: &T,
    test: &CompressedInteractions,
) -> Result<f32, &'static str> {
    let item_ids: Vec<usize> = (0..test.num_items()).collect();

    let mrrs: Vec<f32> = test.iter_users()
        .filter_map(|test_user| {
            if test_user.item_ids.is_empty() {
                return None;
            }

            let train_items = &test_user.item_ids[..test_user.item_ids.len() - 1];
            let test_item = *test_user.item_ids.last().unwrap();

            let user_embedding = model.user_representation(train_items).unwrap();

            let mut predictions = model.predict(&user_embedding, &item_ids).unwrap();

            for &train_item_id in train_items {
                predictions[train_item_id] = std::f32::MIN;
            }

            let test_scores = vec![predictions[test_item]];
            let mut ranks: Vec<usize> = vec![0];

            for &prediction in &predictions {
                for (rank, &score) in ranks.iter_mut().zip(&test_scores) {
                    if prediction >= score {
                        *rank += 1;
                    }
                }
            }

            Some(ranks.iter().map(|&x| 1.0 / x as f32).sum::<f32>() / ranks.len() as f32)
        })
        .collect();

    Ok(mrrs.iter().sum::<f32>() / mrrs.len() as f32)
}
