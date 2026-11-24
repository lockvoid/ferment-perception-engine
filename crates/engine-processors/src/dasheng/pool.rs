//! Element-wise mean pooling over `[N, D]` embeddings.

/// Element-wise mean of N equal-length embeddings. Empty input → empty
/// output. Single-embedding input is returned unchanged (avoids a copy).
pub fn mean_pool(embeddings: &[Vec<f32>]) -> Vec<f32> {
    match embeddings.len() {
        0 => Vec::new(),
        1 => embeddings[0].clone(),
        _ => {
            let dim = embeddings[0].len();
            let mut sum = vec![0.0f32; dim];
            for emb in embeddings {
                debug_assert_eq!(emb.len(), dim);
                for (i, &v) in emb.iter().enumerate() {
                    sum[i] += v;
                }
            }
            let inv = 1.0 / embeddings.len() as f32;
            for v in &mut sum {
                *v *= inv;
            }
            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mean_pool_empty_returns_empty() {
        assert!(mean_pool(&[]).is_empty());
    }

    #[test]
    fn test_mean_pool_single_embedding_returns_itself() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let out = mean_pool(std::slice::from_ref(&v));
        assert_eq!(out, v);
    }

    #[test]
    fn test_mean_pool_element_wise_average() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let c = vec![5.0, 6.0, 7.0, 8.0];
        let pooled = mean_pool(&[a, b, c]);
        assert_eq!(pooled.len(), 4);
        for (actual, expected) in pooled.iter().zip([3.0, 4.0, 5.0, 6.0]) {
            assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mean_pool_with_negative_values() {
        let pooled = mean_pool(&[vec![-1.0, -2.0], vec![1.0, 2.0]]);
        assert_abs_diff_eq!(pooled[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pooled[1], 0.0, epsilon = 1e-6);
    }
}
