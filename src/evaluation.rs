use std;

use rayon::prelude::*;

use super::OnlineRankingModel;
use data::CompressedInteractions;

pub fn mrr_score<T: OnlineRankingModel + Sync>(
    model: &T,
    test: &CompressedInteractions,
) -> Result<f32, &'static str> {
    let item_ids: Vec<usize> = (0..test.num_items()).collect();

    let mrrs: Vec<f32> = test.iter_users()
        .collect::<Vec<_>>()
        .par_iter()
        .filter_map(|test_user| {
            if test_user.item_ids.len() < 2 {
                return None;
            }

            let train_items = &test_user.item_ids[..test_user.item_ids.len().saturating_sub(1)];
            let test_item = *test_user.item_ids.last().unwrap();

            let user_embedding = model.user_representation(train_items).unwrap();
            let mut predictions = model.predict(&user_embedding, &item_ids).unwrap();

            //let mean_prediction: f32 =
            //predictions.iter().map(|&x| x).sum::<f32>() / (predictions.len() as f32);

            for &train_item_id in train_items {
                predictions[train_item_id] = std::f32::MIN;
            }

            let test_score = predictions[test_item];
            let mut rank = 0;

            for &prediction in &predictions {
                assert!(prediction.is_finite());

                if prediction >= test_score {
                    rank += 1;
                }
            }

            // println!(
            //     "item {} rank {} at prediction {}",
            //     test_item, rank, test_score
            // );
            // println!("mean predict score {}", mean_prediction);
            // println!("user embeddding {:?}", user_embedding);

            Some(1.0 / rank as f32)
        })
        .collect();

    Ok(mrrs.iter().sum::<f32>() / mrrs.len() as f32)
}

pub fn mrr_score_train<T: OnlineRankingModel + Sync>(
    model: &T,
    test: &CompressedInteractions,
) -> Result<f32, &'static str> {
    let item_ids: Vec<usize> = (0..test.num_items()).collect();

    let mrrs: Vec<f32> = test.iter_users()
        .collect::<Vec<_>>()
        .par_iter()
        .filter_map(|test_user| {
            if test_user.item_ids.len() < 2 {
                return None;
            }

            let train_items = &test_user.item_ids;
            let test_item = *test_user.item_ids.last().unwrap();

            let user_embedding = model
                .user_representation(&train_items[test_user.item_ids.len().saturating_sub(1000)..])
                .unwrap();
            let mut predictions = model.predict(&user_embedding, &item_ids).unwrap();

            // let mean_prediction: f32 =
            //     predictions.iter().map(|&x| x).sum::<f32>() / (predictions.len() as f32);

            for &train_item_id in &train_items[..test_user.item_ids.len().saturating_sub(1)] {
                predictions[train_item_id] = std::f32::MIN;
            }

            let test_score = predictions[test_item];
            let mut rank = 0;

            for &prediction in &predictions {
                assert!(prediction.is_finite());

                if prediction >= test_score {
                    rank += 1;
                }
            }

            // println!(
            //     "item {} rank {} at prediction {}",
            //     test_item, rank, test_score
            // );
            // println!("mean predict score {}", mean_prediction);
            // println!("user embeddding {:?}", user_embedding);

            Some(1.0 / rank as f32)
        })
        .collect();

    Ok(mrrs.iter().sum::<f32>() / mrrs.len() as f32)
}
