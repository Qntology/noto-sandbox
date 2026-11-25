//! Fitting one set of 3-D points onto another: delegated to `lv_spatial::fit3`.

pub use lv_spatial::fit3::{align_rigid, align_similarity, mpjpe, pa_mpjpe, Similarity, M3, V3};

#[cfg(test)]
mod tests {
    use super::*;

    fn sub(a: V3, b: V3) -> V3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn norm(a: V3) -> f64 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
    }

    #[test]
    fn the_best_similarity_undoes_a_scaled_rotation_and_pa_mpjpe_ignores_it() {
        let src: Vec<V3> = vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.08, 0.0], [0.0, 0.0, 0.05], [0.07, 0.03, 0.02]];
        let (c, s) = (0.6f64.cos(), 0.6f64.sin());
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let moved = Similarity { s: 1.3, r, t: [0.4, -0.2, 1.0] };
        let dst: Vec<V3> = src.iter().map(|p| moved.apply(*p)).collect();
        let fit = align_similarity(&src, &dst).unwrap();
        assert!((fit.s - 1.3).abs() < 1e-9, "{}", fit.s);
        for (p, q) in src.iter().zip(&dst) {
            assert!(norm(sub(fit.apply(*p), *q)) < 1e-9);
        }
        assert!(pa_mpjpe(&src, &dst).unwrap() < 1e-9);
        assert!(mpjpe(&src, &dst) > 0.3);
        let mut bent = dst.clone();
        bent[4][2] += 0.02;
        assert!(pa_mpjpe(&src, &bent).unwrap() > 1e-3);
    }

    #[test]
    fn the_rigid_fit_keeps_the_scale() {
        let src: Vec<V3> = vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.08, 0.0], [0.0, 0.0, 0.05]];
        let dst: Vec<V3> = src.iter().map(|p| [p[0] + 1.0, p[1], p[2] - 2.0]).collect();
        let fit = align_rigid(&src, &dst).unwrap();
        assert_eq!(fit.s, 1.0);
        for (p, q) in src.iter().zip(&dst) {
            assert!(norm(sub(fit.apply(*p), *q)) < 1e-9);
        }
    }
}
