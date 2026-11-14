//! Fitting one set of 3-D points onto another: the least-squares similarity (rotation, scale, translation) by Horn's
//! quaternion method with Umeyama's scale, and the joint errors with and without that alignment (MPJPE, PA-MPJPE).

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// The largest eigenvector of a symmetric 4 × 4 matrix (Jacobi rotations).
fn top_eigenvector4(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let off: f64 = (0..4).flat_map(|i| (0..4).filter(move |j| *j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let k = (0..4).max_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap_or(0);
    [v[0][k], v[1][k], v[2][k], v[3][k]]
}

/// A similarity transform: x ↦ s·R·x + t.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    pub s: f64,
    pub r: M3,
    pub t: V3,
}

impl Similarity {
    pub fn apply(&self, x: V3) -> V3 {
        [0, 1, 2].map(|i| self.s * (0..3).map(|j| self.r[i][j] * x[j]).sum::<f64>() + self.t[i])
    }
}

/// The similarity that best maps `src` onto `dst` in least squares (Horn's quaternion for R, then Umeyama's scale).
pub fn align_similarity(src: &[V3], dst: &[V3]) -> Option<Similarity> {
    let n = src.len();
    if n < 3 || dst.len() != n {
        return None;
    }
    let mean = |v: &[V3]| [0, 1, 2].map(|i| v.iter().map(|p| p[i]).sum::<f64>() / n as f64);
    let (mp, mq) = (mean(src), mean(dst));
    let mut s = [[0.0; 3]; 3];
    let mut var = 0.0;
    for (p, q) in src.iter().zip(dst) {
        let (a, b) = (sub(*p, mp), sub(*q, mq));
        var += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += a[i] * b[j];
            }
        }
    }
    if var < 1e-18 {
        return None;
    }
    let (sxx, sxy, sxz, syx, syy, syz, szx, szy, szz) = (s[0][0], s[0][1], s[0][2], s[1][0], s[1][1], s[1][2], s[2][0], s[2][1], s[2][2]);
    let nm = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let [w, x, y, z] = top_eigenvector4(nm);
    let qn = (w * w + x * x + y * y + z * z).sqrt().max(1e-300);
    let (w, x, y, z) = (w / qn, x / qn, y / qn, z / qn);
    let r = [
        [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z],
    ];
    let mut num = 0.0;
    for (p, q) in src.iter().zip(dst) {
        let (a, b) = (sub(*p, mp), sub(*q, mq));
        let ra = [0, 1, 2].map(|i| (0..3).map(|j| r[i][j] * a[j]).sum::<f64>());
        num += ra[0] * b[0] + ra[1] * b[1] + ra[2] * b[2];
    }
    let scale = num / var;
    let rm = [0, 1, 2].map(|i| (0..3).map(|j| r[i][j] * mp[j]).sum::<f64>());
    Some(Similarity { s: scale, r, t: [0, 1, 2].map(|i| mq[i] - scale * rm[i]) })
}

/// Mean per-joint distance after aligning `pred` to `reference` with the best similarity (PA-MPJPE), m.
pub fn pa_mpjpe(pred: &[V3], reference: &[V3]) -> Option<f64> {
    let sim = align_similarity(pred, reference)?;
    Some(pred.iter().zip(reference).map(|(p, q)| norm(sub(sim.apply(*p), *q))).sum::<f64>() / pred.len() as f64)
}

/// Mean per-joint distance without alignment (MPJPE), m.
pub fn mpjpe(pred: &[V3], reference: &[V3]) -> f64 {
    pred.iter().zip(reference).map(|(p, q)| norm(sub(*p, *q))).sum::<f64>() / pred.len().max(1) as f64
}


/// The rigid transform (scale 1) that best maps `src` onto `dst`.
pub fn align_rigid(src: &[V3], dst: &[V3]) -> Option<Similarity> {
    let sim = align_similarity(src, dst)?;
    let n = src.len() as f64;
    let mean = |v: &[V3]| [0, 1, 2].map(|i| v.iter().map(|p| p[i]).sum::<f64>() / n);
    let (mp, mq) = (mean(src), mean(dst));
    let rm = [0, 1, 2].map(|i| (0..3).map(|j| sim.r[i][j] * mp[j]).sum::<f64>());
    Some(Similarity { s: 1.0, r: sim.r, t: [0, 1, 2].map(|i| mq[i] - rm[i]) })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // A bent finger is a shape error the alignment cannot remove.
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
