use std::ops::{Deref, DerefMut};
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::ml_bridge::{device_label, trim_idle_pool, QuantPlan, QuantStats};
use lv_spatial::depth::{DepthModel, DepthScene, SceneHeads};
use lv_spatial::frame::{DepthKind, DepthMap, Rgb};
use lv_spatial::pyramid::PyramidCfg;
use lv_spatial::stereo::{StereoRig, StereoStats};

pub use lv_spatial::tips::{
    ade_colour, ade_name, bin_centers, decode_bins, is_ade_ground, maybe_load_monocular, InputGrid,
    TipsConfig, ADE_BOX, ADE_CLASSES, ADE_FAR, ADE_GROUND, ADE_NAMES, ADE_ROCK, ADE_SKY, ADE_TREE,
    ADE_VEGETATION, CONFIG_FILE, DEFAULT_DIR, DEFAULT_SHORT_SIDE_PATCHES, DIR_ENV, MODE,
    NORMAL_RIGHT, WEIGHTS_FILE,
};

pub const HEADS_ENV: &str = "NOTO_TIPS_HEADS";

pub fn heads_from_env() -> SceneHeads {
    match std::env::var(HEADS_ENV).ok().as_deref().map(str::trim) {
        Some("depth") | Some("none") => SceneHeads::NONE,
        Some("normals") => SceneHeads { normals: true, segmentation: false },
        Some("segmentation") | Some("seg") => SceneHeads { normals: false, segmentation: true },
        Some(_) => SceneHeads::ALL,
        None => {
            if std::env::var_os(lv_spatial::tips::HEADS_ENV).is_some() {
                lv_spatial::tips::heads_from_env()
            } else {
                SceneHeads::ALL
            }
        }
    }
}

/// Thin runtime wrapper over [`lv_spatial::tips::TipsDepth`] adding disk loading and compatibility
/// metadata methods expected by `noto-runtime`.
pub struct TipsDepth {
    inner: lv_spatial::tips::TipsDepth,
    device: Device,
    quant_plan: QuantPlan,
    quant_stats: QuantStats,
    pub max_rel_spread: Option<f32>,
}

impl Deref for TipsDepth {
    type Target = lv_spatial::tips::TipsDepth;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TipsDepth {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl TipsDepth {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        Self::load_with(dir, device, heads_from_env())
    }

    pub fn load_with(dir: &Path, device: &Device, heads: SceneHeads) -> Result<Self> {
        let weights = dir.join(WEIGHTS_FILE);
        if !weights.exists() {
            bail!(
                "TIPSv2 DPT weights not found at {}. Download google/tipsv2-b14-dpt ({WEIGHTS_FILE}, optionally \
                 {CONFIG_FILE}) into that folder, or point ${DIR_ENV} at the folder that has them (default {DEFAULT_DIR}).",
                weights.display()
            );
        }
        let config_path = dir.join(CONFIG_FILE);
        let cfg = if config_path.exists() {
            TipsConfig::from_file(&config_path)?
        } else {
            TipsConfig::default()
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, device)? };
        Self::from_var_builder_with(cfg, vb, device, heads)
            .with_context(|| format!("loading {}", weights.display()))
    }

    pub fn load_quant(
        dir: &Path,
        device: &Device,
        heads: SceneHeads,
        plan: &QuantPlan,
    ) -> Result<Self> {
        if !plan.is_f32() {
            bail!("TIPSv2 weight quantization was retired in lv-spatial; only F32 plan is supported");
        }
        Self::load_with(dir, device, heads)
    }

    pub fn from_var_builder(cfg: TipsConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        Self::from_var_builder_with(cfg, vb, device, SceneHeads::NONE)
    }

    pub fn from_var_builder_with(
        cfg: TipsConfig,
        vb: VarBuilder,
        device: &Device,
        heads: SceneHeads,
    ) -> Result<Self> {
        let inner = lv_spatial::tips::TipsDepth::from_var_builder_with(cfg, vb, device, heads)?;
        Ok(Self {
            inner,
            device: device.clone(),
            quant_plan: QuantPlan::default(),
            quant_stats: QuantStats::default(),
            max_rel_spread: None,
        })
    }

    pub fn quant_plan(&self) -> &QuantPlan {
        &self.quant_plan
    }

    pub fn quant_stats(&self) -> &QuantStats {
        &self.quant_stats
    }

    pub fn set_readout_gelu_tanh(&mut self, _on: bool) {}

    pub fn predict_batch_spread(&self, frames: &[&Rgb]) -> Result<Vec<(DepthMap, DepthMap)>> {
        let depths = self.inner.predict_batch(frames)?;
        trim_idle_pool(&self.device);
        Ok(depths
            .into_iter()
            .map(|d| {
                let spread = DepthMap {
                    width: d.width,
                    height: d.height,
                    kind: DepthKind::Metric,
                    data: vec![0.0; d.data.len()],
                };
                (d, spread)
            })
            .collect())
    }
}

impl DepthModel for TipsDepth {
    fn mode(&self) -> &str {
        self.inner.mode()
    }

    fn device(&self) -> String {
        device_label(&self.device)
    }

    fn max_depth_m(&self) -> Option<f32> {
        self.inner.max_depth_m()
    }

    fn infer(&mut self, rgb: &Rgb) -> Result<DepthMap> {
        let out = self.inner.infer(rgb);
        trim_idle_pool(&self.device);
        out
    }

    fn infer_batch(&mut self, frames: &[&Rgb]) -> Result<Vec<DepthMap>> {
        let out = self.inner.infer_batch(frames);
        trim_idle_pool(&self.device);
        out
    }

    fn scene_heads(&self) -> SceneHeads {
        self.inner.scene_heads()
    }

    fn infer_scene_batch(&mut self, frames: &[&Rgb], want: SceneHeads) -> Result<Vec<DepthScene>> {
        let out = self.inner.infer_scene_batch(frames, want);
        trim_idle_pool(&self.device);
        out
    }

    fn set_pyramid(&mut self, cfg: Option<&PyramidCfg>) {
        self.inner.set_pyramid(cfg);
    }

    fn infer_scene(&mut self, rgb: &Rgb, want: SceneHeads) -> Result<DepthScene> {
        let out = self.inner.infer_scene(rgb, want);
        trim_idle_pool(&self.device);
        out
    }

    fn stereo_rig(&self) -> Option<StereoRig> {
        self.inner.stereo_rig()
    }

    fn infer_pair(&mut self, left: &Rgb, right: &Rgb) -> Result<DepthMap> {
        self.inner.infer_pair(left, right)
    }

    fn stereo_stats(&self) -> Option<StereoStats> {
        self.inner.stereo_stats()
    }
}
