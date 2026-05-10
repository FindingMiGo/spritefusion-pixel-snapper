use image::{GenericImageView, ImageBuffer, Rgba, RgbaImage};
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, WeightedIndex};
use std::cmp::Ordering;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::env;
use std::error::Error;
use std::fmt;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Config {
    pub k_colors: usize,
    pub pixel_size_override: Option<f64>,
    pub target_grid_width: Option<usize>,
    pub target_grid_height: Option<usize>,
    resample_mode: ResampleMode,
    edge_weight: f64,
    palette: Option<Vec<[u8; 3]>>,
    k_seed: u64,
    /// Input image path only used for CLI use
    #[allow(dead_code)]
    input_path: String,
    /// Output image path only used for CLI use
    #[allow(dead_code)]
    output_path: String,
    max_kmeans_iterations: usize,
    peak_threshold_multiplier: f64,
    peak_distance_filter: usize,
    walker_search_window_ratio: f64,
    walker_min_search_window: f64,
    walker_strength_threshold: f64,
    min_cuts_per_axis: usize,
    fallback_target_segments: usize,
    max_step_ratio: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            k_colors: 16,
            k_seed: 42,
            input_path: "samples/2/skeleton.png".to_string(),
            output_path: "samples/2/skeleton_fixed_clean2.png".to_string(),
            max_kmeans_iterations: 15,
            peak_threshold_multiplier: 0.2,
            peak_distance_filter: 4,
            walker_search_window_ratio: 0.35,
            walker_min_search_window: 2.0,
            walker_strength_threshold: 0.5,
            min_cuts_per_axis: 4,
            fallback_target_segments: 64,
            max_step_ratio: 1.8, // Lowered from 3.0 to catch more skew cases
            pixel_size_override: None,
            target_grid_width: None,
            target_grid_height: None,
            resample_mode: ResampleMode::Majority,
            edge_weight: 1.0,
            palette: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResampleMode {
    Majority,
    Center,
    Mean,
    EdgeAware,
    PaletteAware,
}

impl ResampleMode {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "majority" | "vote" | "mode" => Some(Self::Majority),
            "center" | "centre" => Some(Self::Center),
            "mean" | "average" | "avg" => Some(Self::Mean),
            "edge" | "edge-aware" | "edgeaware" => Some(Self::EdgeAware),
            "palette" | "palette-aware" | "paletteaware" => Some(Self::PaletteAware),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Majority => "majority",
            Self::Center => "center",
            Self::Mean => "mean",
            Self::EdgeAware => "edge-aware",
            Self::PaletteAware => "palette-aware",
        }
    }
}

#[derive(Debug)]
pub enum PixelSnapperError {
    ImageError(image::ImageError),
    InvalidInput(String),
    ProcessingError(String),
}

impl fmt::Display for PixelSnapperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PixelSnapperError::ImageError(e) => write!(f, "Image error: {}", e),
            PixelSnapperError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            PixelSnapperError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl Error for PixelSnapperError {}

impl From<image::ImageError> for PixelSnapperError {
    fn from(error: image::ImageError) -> Self {
        PixelSnapperError::ImageError(error)
    }
}

#[cfg(target_arch = "wasm32")]
impl From<PixelSnapperError> for wasm_bindgen::JsValue {
    fn from(err: PixelSnapperError) -> wasm_bindgen::JsValue {
        wasm_bindgen::JsValue::from_str(&err.to_string())
    }
}

type Result<T> = std::result::Result<T, PixelSnapperError>;

/// CLI entry point
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn main() -> Result<()> {
    let config = parse_args().unwrap_or_default();
    process_image(&config)
}

fn process_image_bytes_common(input_bytes: &[u8], config: Option<Config>) -> Result<Vec<u8>> {
    let config = config.unwrap_or_default();

    let img = image::load_from_memory(input_bytes)?;
    let (width, height) = img.dimensions();

    validate_image_dimensions(width, height)?;

    if let Some(px) = config.pixel_size_override {
        if !px.is_finite() || px < 1.0 || px > (width.min(height) as f64 / 2.0) {
            return Err(PixelSnapperError::InvalidInput(format!(
                "pixel_size_override {:.1} is out of valid range [1, {}]",
                px,
                width.min(height) / 2
            )));
        }
    }
    if let (Some(target_w), Some(target_h)) = (config.target_grid_width, config.target_grid_height)
    {
        validate_target_grid_size(target_w, target_h, width as usize, height as usize)?;
    } else if config.target_grid_width.is_some() || config.target_grid_height.is_some() {
        return Err(PixelSnapperError::InvalidInput(
            "target grid width and height must both be provided".to_string(),
        ));
    }
    if config.resample_mode == ResampleMode::PaletteAware && config.palette.is_none() {
        return Err(PixelSnapperError::InvalidInput(
            "palette-aware resampling requires --palette".to_string(),
        ));
    }

    let rgba_img = img.to_rgba8();

    let quantized_img = quantize_image(&rgba_img, &config)?;
    let (profile_x, profile_y) = compute_profiles(&quantized_img)?;

    // Estimate step sizes
    let step_x_opt = estimate_step_size(&profile_x, &config);
    let step_y_opt = estimate_step_size(&profile_y, &config);
    let autocorr_step_x_opt = estimate_step_size_autocorr(&profile_x);
    let autocorr_step_y_opt = estimate_step_size_autocorr(&profile_y);

    // Resolve step sizes. Some instabilities so use sibling axis if one fails, or fallback if both fail
    let (step_x, step_y) = resolve_step_sizes(step_x_opt, step_y_opt, width, height, &config);

    println!(
        "Pixel size: {:.1}px ({})",
        step_x,
        if config.pixel_size_override.is_some() {
            "override"
        } else {
            "auto-detected"
        }
    );

    let (col_cuts, row_cuts) = if let (Some(target_w), Some(target_h)) =
        (config.target_grid_width, config.target_grid_height)
    {
        println!("Target grid size: {}x{}", target_w, target_h);
        (
            snap_grid_size_cuts(&profile_x, width as usize, target_w, &config),
            snap_grid_size_cuts(&profile_y, height as usize, target_h, &config),
        )
    } else if config.pixel_size_override.is_none() {
        select_grid_candidate(
            &profile_x,
            &profile_y,
            step_x,
            step_y,
            width as usize,
            height as usize,
            &rgba_img,
            &config,
            autocorr_step_x_opt,
            autocorr_step_y_opt,
        )?
    } else {
        let raw_col_cuts = walk(&profile_x, step_x, width as usize, &config)?;
        let raw_row_cuts = walk(&profile_y, step_y, height as usize, &config)?;

        // Two-pass stabilization: first pass with raw cuts, then cross-validate
        stabilize_both_axes(
            &profile_x,
            &profile_y,
            raw_col_cuts,
            raw_row_cuts,
            width as usize,
            height as usize,
            &config,
        )
    };

    println!("Output size: {}x{}", col_cuts.len() - 1, row_cuts.len() - 1);

    println!("Resample mode: {}", config.resample_mode.label());
    let output_img = resample(&quantized_img, &col_cuts, &row_cuts, &config)?;

    // Returns bytes for both implementations
    let mut output_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut output_bytes);
    output_img
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| PixelSnapperError::ImageError(e))?;

    Ok(output_bytes)
}

/// WASM entry point
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn process_image(
    input_bytes: &[u8],
    k_colors: Option<u32>,
    pixel_size_override: Option<f64>,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let mut config = Config::default();
    if let Some(k) = k_colors {
        if k == 0 {
            return Err(wasm_bindgen::JsValue::from_str(
                "k_colors must be greater than 0",
            ));
        }
        config.k_colors = k as usize;
    }

    config.pixel_size_override = pixel_size_override;

    process_image_bytes_common(input_bytes, Some(config))
        .map_err(|e| wasm_bindgen::JsValue::from(e))
}

/// WASM entry point with an explicit output grid size.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn process_image_with_grid(
    input_bytes: &[u8],
    k_colors: Option<u32>,
    pixel_size_override: Option<f64>,
    target_grid_width: u32,
    target_grid_height: u32,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let mut config = Config::default();
    if let Some(k) = k_colors {
        if k == 0 {
            return Err(wasm_bindgen::JsValue::from_str(
                "k_colors must be greater than 0",
            ));
        }
        config.k_colors = k as usize;
    }

    config.pixel_size_override = pixel_size_override;
    config.target_grid_width = Some(target_grid_width as usize);
    config.target_grid_height = Some(target_grid_height as usize);

    process_image_bytes_common(input_bytes, Some(config))
        .map_err(|e| wasm_bindgen::JsValue::from(e))
}

/// WASM entry point with explicit grid and resampling options.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn process_image_with_options(
    input_bytes: &[u8],
    k_colors: Option<u32>,
    pixel_size_override: Option<f64>,
    target_grid_width: Option<u32>,
    target_grid_height: Option<u32>,
    resample_mode: Option<String>,
    edge_weight: Option<f64>,
    palette: Option<String>,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let mut config = Config::default();
    if let Some(k) = k_colors {
        if k == 0 {
            return Err(wasm_bindgen::JsValue::from_str(
                "k_colors must be greater than 0",
            ));
        }
        config.k_colors = k as usize;
    }

    config.pixel_size_override = pixel_size_override;
    config.target_grid_width = target_grid_width.map(|v| v as usize);
    config.target_grid_height = target_grid_height.map(|v| v as usize);
    if let Some(mode) = resample_mode {
        config.resample_mode = ResampleMode::parse(&mode).ok_or_else(|| {
            wasm_bindgen::JsValue::from_str(
                "resample_mode must be one of: majority, center, mean, edge-aware",
            )
        })?;
    }
    if let Some(weight) = edge_weight {
        if !weight.is_finite() || weight < 0.0 {
            return Err(wasm_bindgen::JsValue::from_str(
                "edge_weight must be a finite number greater than or equal to 0",
            ));
        }
        config.edge_weight = weight.min(5.0);
    }
    if let Some(palette) = palette {
        config.palette =
            Some(parse_palette_colors(&palette).map_err(|e| {
                wasm_bindgen::JsValue::from_str(&format!("invalid palette: {}", e))
            })?);
    }

    process_image_bytes_common(input_bytes, Some(config))
        .map_err(|e| wasm_bindgen::JsValue::from(e))
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn parse_args() -> Option<Config> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        return None;
    }

    let mut config = Config {
        input_path: args[1].clone(),
        output_path: args[2].clone(),
        ..Default::default()
    };

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--pixel-size" => {
                let Some(val) = args.get(i + 1) else {
                    eprintln!("Warning: --pixel-size requires a value");
                    break;
                };

                match val.parse::<f64>() {
                    Ok(px) if px.is_finite() && px > 0.0 => config.pixel_size_override = Some(px),
                    _ => eprintln!("Warning: invalid --pixel-size '{}', ignoring", val),
                }
                i += 2;
            }
            "--grid-size" => {
                let Some(val) = args.get(i + 1) else {
                    eprintln!("Warning: --grid-size requires a value like 32x32");
                    break;
                };

                match parse_grid_size(val) {
                    Some((w, h)) => {
                        config.target_grid_width = Some(w);
                        config.target_grid_height = Some(h);
                    }
                    None => eprintln!("Warning: invalid --grid-size '{}', ignoring", val),
                }
                i += 2;
            }
            "--resample" => {
                let Some(val) = args.get(i + 1) else {
                    eprintln!(
                        "Warning: --resample requires one of majority, center, mean, edge-aware, palette-aware"
                    );
                    break;
                };

                match ResampleMode::parse(val) {
                    Some(mode) => config.resample_mode = mode,
                    None => eprintln!("Warning: invalid --resample '{}', ignoring", val),
                }
                i += 2;
            }
            "--edge-weight" => {
                let Some(val) = args.get(i + 1) else {
                    eprintln!("Warning: --edge-weight requires a finite non-negative value");
                    break;
                };

                match val.parse::<f64>() {
                    Ok(weight) if weight.is_finite() && weight >= 0.0 => {
                        config.edge_weight = weight.min(5.0)
                    }
                    _ => eprintln!("Warning: invalid --edge-weight '{}', ignoring", val),
                }
                i += 2;
            }
            "--palette" => {
                let Some(val) = args.get(i + 1) else {
                    eprintln!("Warning: --palette requires a file path or #rrggbb color list");
                    break;
                };

                match load_palette_arg(val) {
                    Ok(palette) => config.palette = Some(palette),
                    Err(err) => eprintln!("Warning: invalid --palette '{}': {}", val, err),
                }
                i += 2;
            }
            arg if arg.starts_with("--") => {
                eprintln!("Warning: unknown argument '{}', ignoring", arg);
                i += 1;
            }
            k_arg => {
                match k_arg.parse::<usize>() {
                    Ok(k) if k > 0 => config.k_colors = k,
                    _ => eprintln!(
                        "Warning: invalid k_colors '{}', falling back to default ({})",
                        k_arg, config.k_colors
                    ),
                }
                i += 1;
            }
        }
    }

    Some(config)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_grid_size(value: &str) -> Option<(usize, usize)> {
    let normalized = value.trim().to_ascii_lowercase();
    let (w, h) = normalized.split_once('x')?;
    let w = w.parse::<usize>().ok()?;
    let h = h.parse::<usize>().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

#[cfg(not(target_arch = "wasm32"))]
fn load_palette_arg(value: &str) -> Result<Vec<[u8; 3]>> {
    let content = if std::path::Path::new(value).exists() {
        std::fs::read_to_string(value).map_err(|e| {
            PixelSnapperError::InvalidInput(format!("failed to read palette file: {}", e))
        })?
    } else {
        value.to_string()
    };
    parse_palette_colors(&content)
}

fn parse_palette_colors(value: &str) -> Result<Vec<[u8; 3]>> {
    let mut colors = Vec::new();
    for line in value.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.contains(',') {
            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
            if parts.len() >= 6 {
                if let (Ok(r), Ok(g), Ok(b)) = (
                    parts[3].parse::<u8>(),
                    parts[4].parse::<u8>(),
                    parts[5].parse::<u8>(),
                ) {
                    colors.push([r, g, b]);
                    continue;
                }
            }
            if parts.len() >= 3 {
                if let (Ok(r), Ok(g), Ok(b)) = (
                    parts[0].parse::<u8>(),
                    parts[1].parse::<u8>(),
                    parts[2].parse::<u8>(),
                ) {
                    colors.push([r, g, b]);
                    continue;
                }
            }
        }

        for token in line.split(|c: char| c.is_whitespace() || c == ';' || c == ',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            if let Some(color) = parse_hex_color(token) {
                colors.push(color);
            }
        }
    }

    colors.sort_unstable();
    colors.dedup();
    if colors.is_empty() {
        return Err(PixelSnapperError::InvalidInput(
            "palette must contain at least one RGB color".to_string(),
        ));
    }
    Ok(colors)
}

fn parse_hex_color(value: &str) -> Option<[u8; 3]> {
    let hex = value.trim().trim_start_matches('#');
    if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some([r, g, b])
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn process_image(config: &Config) -> Result<()> {
    println!("Processing: {}", config.input_path);

    let img_bytes = std::fs::read(&config.input_path).map_err(|e| {
        PixelSnapperError::ProcessingError(format!("Failed to read input file: {}", e))
    })?;

    let output_bytes = process_image_bytes_common(&img_bytes, Some(config.clone()))?;

    std::fs::write(&config.output_path, output_bytes).map_err(|e| {
        PixelSnapperError::ProcessingError(format!("Failed to write output file: {}", e))
    })?;

    println!("Saved to: {}", config.output_path);
    Ok(())
}

fn validate_image_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Image dimensions cannot be zero".to_string(),
        ));
    }
    if width > 10000 || height > 10000 {
        return Err(PixelSnapperError::InvalidInput(
            "Image dimensions too large (max 10000x10000)".to_string(),
        ));
    }
    Ok(())
}

fn validate_target_grid_size(
    target_w: usize,
    target_h: usize,
    input_w: usize,
    input_h: usize,
) -> Result<()> {
    if target_w == 0 || target_h == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Target grid dimensions must be greater than zero".to_string(),
        ));
    }
    if target_w > input_w || target_h > input_h {
        return Err(PixelSnapperError::InvalidInput(format!(
            "Target grid size {}x{} cannot exceed input size {}x{}",
            target_w, target_h, input_w, input_h
        )));
    }
    Ok(())
}

fn quantize_image(img: &RgbaImage, config: &Config) -> Result<RgbaImage> {
    if config.k_colors == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Number of colors must be greater than 0".to_string(),
        ));
    }
    if let Some(palette) = &config.palette {
        return Ok(map_image_to_palette(img, palette));
    }

    let opaque_pixels: Vec<[f32; 3]> = img
        .pixels()
        .filter_map(|p| {
            if p[3] == 0 {
                None
            } else {
                Some([p[0] as f32, p[1] as f32, p[2] as f32])
            }
        })
        .collect();
    let n_pixels = opaque_pixels.len();
    if n_pixels == 0 {
        return Ok(img.clone());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(config.k_seed);
    let k = config.k_colors.min(n_pixels);

    fn sample_index(rng: &mut ChaCha8Rng, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let upper = upper as u64;
        rng.gen_range(0..upper) as usize
    }

    fn dist_sq(p: &[f32; 3], c: &[f32; 3]) -> f32 {
        let dr = p[0] - c[0];
        let dg = p[1] - c[1];
        let db = p[2] - c[2];
        dr * dr + dg * dg + db * db
    }

    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    let first_idx = sample_index(&mut rng, n_pixels);
    centroids.push(opaque_pixels[first_idx]);
    let mut distances = vec![f32::MAX; n_pixels];

    // Maybe try a faster algorithm for this? like https://crates.io/crates/kmeans_colors
    for _ in 1..k {
        let last_c = centroids.last().unwrap();
        let mut sum_sq_dist = 0.0;

        for (i, p) in opaque_pixels.iter().enumerate() {
            let d_sq = dist_sq(p, last_c);
            if d_sq < distances[i] {
                distances[i] = d_sq;
            }
            sum_sq_dist += distances[i];
        }

        if sum_sq_dist <= 0.0 {
            let idx = sample_index(&mut rng, n_pixels);
            centroids.push(opaque_pixels[idx]);
        } else {
            let dist = WeightedIndex::new(&distances).map_err(|e| {
                PixelSnapperError::ProcessingError(format!("Failed to sample new centroid: {}", e))
            })?;
            let idx = dist.sample(&mut rng);
            centroids.push(opaque_pixels[idx]);
        }
    }

    let mut prev_centroids = centroids.clone();
    for iteration in 0..config.max_kmeans_iterations {
        let mut sums = vec![[0.0f32; 3]; k];
        let mut counts = vec![0usize; k];

        for p in &opaque_pixels {
            let mut min_dist = f32::MAX;
            let mut best_k = 0;

            for (i, c) in centroids.iter().enumerate() {
                let d = dist_sq(p, c);
                if d < min_dist {
                    min_dist = d;
                    best_k = i;
                }
            }
            sums[best_k][0] += p[0];
            sums[best_k][1] += p[1];
            sums[best_k][2] += p[2];
            counts[best_k] += 1;
        }

        for i in 0..k {
            if counts[i] > 0 {
                let fcount = counts[i] as f32;
                centroids[i] = [
                    sums[i][0] / fcount,
                    sums[i][1] / fcount,
                    sums[i][2] / fcount,
                ];
            }
        }

        if iteration > 0 {
            let mut max_movement = 0.0f32;
            for (new_c, old_c) in centroids.iter().zip(prev_centroids.iter()) {
                let movement = dist_sq(new_c, old_c);
                if movement > max_movement {
                    max_movement = movement;
                }
            }

            if max_movement < 0.01 {
                break;
            }
        }

        prev_centroids.copy_from_slice(&centroids);
    }

    let mut new_img = RgbaImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel[3] == 0 {
            new_img.put_pixel(x, y, *pixel);
            continue;
        }
        let p = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
        let mut min_dist = f32::MAX;
        let mut best_c = [pixel[0], pixel[1], pixel[2]];

        for c in &centroids {
            let d = dist_sq(&p, c);
            if d < min_dist {
                min_dist = d;
                best_c = [c[0].round() as u8, c[1].round() as u8, c[2].round() as u8];
            }
        }
        new_img.put_pixel(x, y, Rgba([best_c[0], best_c[1], best_c[2], pixel[3]]));
    }
    Ok(new_img)
}

fn map_image_to_palette(img: &RgbaImage, palette: &[[u8; 3]]) -> RgbaImage {
    let mut new_img = RgbaImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel[3] == 0 {
            new_img.put_pixel(x, y, *pixel);
            continue;
        }
        let best =
            nearest_palette_color([pixel[0] as f64, pixel[1] as f64, pixel[2] as f64], palette);
        new_img.put_pixel(x, y, Rgba([best[0], best[1], best[2], pixel[3]]));
    }
    new_img
}

fn nearest_palette_color(rgb: [f64; 3], palette: &[[u8; 3]]) -> [u8; 3] {
    let mut best = palette.first().copied().unwrap_or([0, 0, 0]);
    let mut best_dist = f64::MAX;
    for &candidate in palette {
        let dr = rgb[0] - candidate[0] as f64;
        let dg = rgb[1] - candidate[1] as f64;
        let db = rgb[2] - candidate[2] as f64;
        let dist = dr * dr + dg * dg + db * db;
        if dist < best_dist {
            best_dist = dist;
            best = candidate;
        }
    }
    best
}

fn compute_profiles(img: &RgbaImage) -> Result<(Vec<f64>, Vec<f64>)> {
    let (w, h) = img.dimensions();

    if w < 3 || h < 3 {
        return Err(PixelSnapperError::InvalidInput(
            "Image too small (minimum 3x3)".to_string(),
        ));
    }

    let mut col_proj = vec![0.0; w as usize];
    let mut row_proj = vec![0.0; h as usize];

    let gray = |x, y| {
        let p = img.get_pixel(x, y);
        if p[3] == 0 {
            0.0
        } else {
            0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64
        }
    };

    // kernels: [-1, 0, 1]
    for y in 0..h {
        for x in 1..w - 1 {
            let left = gray(x - 1, y);
            let right = gray(x + 1, y);
            let grad = (right - left).abs();
            col_proj[x as usize] += grad;
        }
    }
    for x in 0..w {
        for y in 1..h - 1 {
            let top = gray(x, y - 1);
            let bottom = gray(x, y + 1);
            let grad = (bottom - top).abs();
            row_proj[y as usize] += grad;
        }
    }

    Ok((col_proj, row_proj))
}

fn estimate_step_size(profile: &[f64], config: &Config) -> Option<f64> {
    if profile.is_empty() {
        return None;
    }

    let max_val = profile.iter().cloned().fold(0.0 / 0.0, f64::max);
    if max_val == 0.0 {
        return None; // Decide later
    }
    let threshold = max_val * config.peak_threshold_multiplier;

    let mut peaks = Vec::new();
    for i in 1..profile.len() - 1 {
        if profile[i] > threshold && profile[i] > profile[i - 1] && profile[i] > profile[i + 1] {
            peaks.push(i);
        }
    }

    if peaks.len() < 2 {
        return None;
    }

    let mut clean_peaks = vec![peaks[0]];
    for &p in peaks.iter().skip(1) {
        if p - clean_peaks.last().unwrap() > (config.peak_distance_filter - 1) {
            clean_peaks.push(p);
        }
    }

    if clean_peaks.len() < 2 {
        return None;
    }

    // Compute diffs
    let mut diffs: Vec<f64> = clean_peaks
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .collect();

    // Median
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Some(diffs[diffs.len() / 2])
}

fn estimate_step_size_autocorr(profile: &[f64]) -> Option<f64> {
    if profile.len() < 6 {
        return None;
    }
    let mean = profile.iter().sum::<f64>() / profile.len() as f64;
    let centered: Vec<f64> = profile.iter().map(|value| value - mean).collect();
    let energy = centered.iter().map(|value| value * value).sum::<f64>();
    if energy <= 0.0 {
        return None;
    }

    let max_lag = (profile.len() / 2).min(256);
    let mut best_lag = 0usize;
    let mut best_score = f64::MIN;
    for lag in 2..=max_lag {
        let mut score = 0.0;
        let mut count = 0usize;
        for i in 0..profile.len() - lag {
            score += centered[i] * centered[i + lag];
            count += 1;
        }
        if count > 0 {
            score /= count as f64;
        }
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    (best_lag > 0 && best_score > 0.0).then_some(best_lag as f64)
}

fn resolve_step_sizes(
    step_x_opt: Option<f64>,
    step_y_opt: Option<f64>,
    width: u32,
    height: u32,
    config: &Config,
) -> (f64, f64) {
    if let Some(px) = config.pixel_size_override {
        return (px, px);
    }

    match (step_x_opt, step_y_opt) {
        (Some(sx), Some(sy)) => {
            let ratio = if sx > sy { sx / sy } else { sy / sx };
            if ratio > config.max_step_ratio {
                let smaller = sx.min(sy);
                (smaller, smaller)
            } else {
                let avg = (sx + sy) / 2.0;
                (avg, avg)
            }
        }

        (Some(sx), None) => (sx, sx),

        (None, Some(sy)) => (sy, sy),

        (None, None) => {
            let fallback_step =
                ((width.min(height) as f64) / config.fallback_target_segments as f64).max(1.0);
            (fallback_step, fallback_step)
        }
    }
}

fn stabilize_both_axes(
    profile_x: &[f64],
    profile_y: &[f64],
    raw_col_cuts: Vec<usize>,
    raw_row_cuts: Vec<usize>,
    width: usize,
    height: usize,
    config: &Config,
) -> (Vec<usize>, Vec<usize>) {
    let col_cuts_pass1 = stabilize_cuts(
        profile_x,
        raw_col_cuts.clone(),
        width,
        &raw_row_cuts,
        height,
        config,
    );
    let row_cuts_pass1 = stabilize_cuts(
        profile_y,
        raw_row_cuts.clone(),
        height,
        &raw_col_cuts,
        width,
        config,
    );

    // Check if the results are coherent
    let col_cells = col_cuts_pass1.len().saturating_sub(1).max(1);
    let row_cells = row_cuts_pass1.len().saturating_sub(1).max(1);
    let col_step = width as f64 / col_cells as f64;
    let row_step = height as f64 / row_cells as f64;

    let step_ratio = if col_step > row_step {
        col_step / row_step
    } else {
        row_step / col_step
    };

    if step_ratio > config.max_step_ratio {
        let target_step = col_step.min(row_step);

        let final_col_cuts = if col_step > target_step * 1.2 {
            snap_uniform_cuts(
                profile_x,
                width,
                target_step,
                config,
                config.min_cuts_per_axis,
            )
        } else {
            col_cuts_pass1
        };

        let final_row_cuts = if row_step > target_step * 1.2 {
            snap_uniform_cuts(
                profile_y,
                height,
                target_step,
                config,
                config.min_cuts_per_axis,
            )
        } else {
            row_cuts_pass1
        };

        (final_col_cuts, final_row_cuts)
    } else {
        (col_cuts_pass1, row_cuts_pass1)
    }
}

struct GridCandidate {
    col_cuts: Vec<usize>,
    row_cuts: Vec<usize>,
    source: &'static str,
    score: f64,
}

fn select_grid_candidate(
    profile_x: &[f64],
    profile_y: &[f64],
    step_x: f64,
    step_y: f64,
    width: usize,
    height: usize,
    source_img: &RgbaImage,
    config: &Config,
    autocorr_step_x_opt: Option<f64>,
    autocorr_step_y_opt: Option<f64>,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut candidates = Vec::new();

    if let Some(candidate) = build_walk_candidate(
        profile_x, profile_y, step_x, step_y, width, height, config, step_x, step_y, "detected",
        source_img,
    )? {
        candidates.push(candidate);
    }

    if autocorr_step_x_opt.is_some() || autocorr_step_y_opt.is_some() {
        let (autocorr_step_x, autocorr_step_y) = resolve_step_sizes(
            autocorr_step_x_opt,
            autocorr_step_y_opt,
            width as u32,
            height as u32,
            config,
        );
        let ratio_x = autocorr_step_x / step_x.max(1.0);
        let ratio_y = autocorr_step_y / step_y.max(1.0);
        if (0.75..=1.25).contains(&ratio_x) && (0.75..=1.25).contains(&ratio_y) {
            if let Some(candidate) = build_walk_candidate(
                profile_x,
                profile_y,
                autocorr_step_x,
                autocorr_step_y,
                width,
                height,
                config,
                step_x,
                step_y,
                "autocorr",
                source_img,
            )? {
                candidates.push(candidate);
            }
        }
    }

    if let Some(candidate) = build_line_mesh_candidate(
        profile_x, profile_y, step_x, step_y, width, height, config, source_img,
    ) {
        candidates.push(candidate);
    }

    if let Some(candidate) = build_reconstruction_candidate(
        source_img, profile_x, profile_y, step_x, step_y, width, height,
    ) {
        candidates.push(candidate);
    }

    for &(multiplier, source) in &[
        (0.9, "slightly-smaller-step"),
        (1.1, "slightly-larger-step"),
    ] {
        let candidate_step_x = step_x * multiplier;
        let candidate_step_y = step_y * multiplier;
        if candidate_step_x >= 1.0 && candidate_step_y >= 1.0 {
            if let Some(candidate) = build_walk_candidate(
                profile_x,
                profile_y,
                candidate_step_x,
                candidate_step_y,
                width,
                height,
                config,
                step_x,
                step_y,
                source,
                source_img,
            )? {
                candidates.push(candidate);
            }
        }
    }

    let expected_segments = (width.min(height) as f64 / step_x.max(step_y).max(1.0)).round();
    for &(segments, source) in &[(16usize, "fixed-16"), (32, "fixed-32"), (64, "fixed-64")] {
        let ratio = segments as f64 / expected_segments.max(1.0);
        if segments <= width.min(height) && (0.75..=1.25).contains(&ratio) {
            let candidate_step_x = width as f64 / segments as f64;
            let candidate_step_y = height as f64 / segments as f64;
            if let Some(candidate) = build_uniform_candidate(
                profile_x,
                profile_y,
                candidate_step_x,
                candidate_step_y,
                width,
                height,
                config,
                step_x,
                step_y,
                source,
                source_img,
            ) {
                candidates.push(candidate);
            }
        }
    }

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    candidates.dedup_by(|a, b| {
        a.col_cuts.len() == b.col_cuts.len() && a.row_cuts.len() == b.row_cuts.len()
    });

    let best = candidates.into_iter().next().ok_or_else(|| {
        PixelSnapperError::ProcessingError("No valid grid candidates found".to_string())
    })?;
    println!(
        "Grid candidate: {} (score {:.3}, size {}x{})",
        best.source,
        best.score,
        best.col_cuts.len().saturating_sub(1),
        best.row_cuts.len().saturating_sub(1)
    );
    Ok((best.col_cuts, best.row_cuts))
}

fn build_walk_candidate(
    profile_x: &[f64],
    profile_y: &[f64],
    step_x: f64,
    step_y: f64,
    width: usize,
    height: usize,
    config: &Config,
    score_expected_step_x: f64,
    score_expected_step_y: f64,
    source: &'static str,
    source_img: &RgbaImage,
) -> Result<Option<GridCandidate>> {
    let raw_col_cuts = walk(profile_x, step_x, width, config)?;
    let raw_row_cuts = walk(profile_y, step_y, height, config)?;
    let (col_cuts, row_cuts) = stabilize_both_axes(
        profile_x,
        profile_y,
        raw_col_cuts,
        raw_row_cuts,
        width,
        height,
        config,
    );
    Ok(make_grid_candidate(
        profile_x,
        profile_y,
        col_cuts,
        row_cuts,
        score_expected_step_x,
        score_expected_step_y,
        width,
        height,
        source,
        source_img,
    ))
}

fn build_uniform_candidate(
    profile_x: &[f64],
    profile_y: &[f64],
    step_x: f64,
    step_y: f64,
    width: usize,
    height: usize,
    config: &Config,
    score_expected_step_x: f64,
    score_expected_step_y: f64,
    source: &'static str,
    source_img: &RgbaImage,
) -> Option<GridCandidate> {
    let col_cuts = snap_uniform_cuts(profile_x, width, step_x, config, config.min_cuts_per_axis);
    let row_cuts = snap_uniform_cuts(profile_y, height, step_y, config, config.min_cuts_per_axis);
    make_grid_candidate(
        profile_x,
        profile_y,
        col_cuts,
        row_cuts,
        score_expected_step_x,
        score_expected_step_y,
        width,
        height,
        source,
        source_img,
    )
}

fn build_line_mesh_candidate(
    profile_x: &[f64],
    profile_y: &[f64],
    expected_step_x: f64,
    expected_step_y: f64,
    width: usize,
    height: usize,
    _config: &Config,
    source_img: &RgbaImage,
) -> Option<GridCandidate> {
    let col_lines = detect_strong_line_positions(profile_x, width)?;
    let row_lines = detect_strong_line_positions(profile_y, height)?;
    let col_step = median_spacing(&col_lines).unwrap_or(expected_step_x);
    let row_step = median_spacing(&row_lines).unwrap_or(expected_step_y);
    if !step_close(col_step, expected_step_x) || !step_close(row_step, expected_step_y) {
        return None;
    }

    let col_cuts = homogenize_lines(&col_lines, col_step, width);
    let row_cuts = homogenize_lines(&row_lines, row_step, height);
    make_grid_candidate(
        profile_x,
        profile_y,
        col_cuts,
        row_cuts,
        expected_step_x,
        expected_step_y,
        width,
        height,
        "line-mesh",
        source_img,
    )
}

fn build_reconstruction_candidate(
    source_img: &RgbaImage,
    profile_x: &[f64],
    profile_y: &[f64],
    expected_step_x: f64,
    expected_step_y: f64,
    width: usize,
    height: usize,
) -> Option<GridCandidate> {
    let base_step = ((expected_step_x + expected_step_y) / 2.0).round();
    if !base_step.is_finite() || base_step < 2.0 {
        return None;
    }

    let mut best: Option<(Vec<usize>, Vec<usize>, f64)> = None;
    for step_delta in [-1.0, 0.0, 1.0] {
        let step = (base_step + step_delta).max(2.0);
        if !step_close(step, expected_step_x) || !step_close(step, expected_step_y) {
            continue;
        }
        let max_offset = (step as usize).min(8);
        let stride = (max_offset / 3).max(1);
        for offset_x in (0..max_offset).step_by(stride) {
            let col_cuts = make_uniform_offset_cuts(step, offset_x as f64, width);
            for offset_y in (0..max_offset).step_by(stride) {
                let row_cuts = make_uniform_offset_cuts(step, offset_y as f64, height);
                let error = reconstruction_error(source_img, &col_cuts, &row_cuts);
                if best
                    .as_ref()
                    .map(|(_, _, best_error)| error < *best_error)
                    .unwrap_or(true)
                {
                    best = Some((col_cuts.clone(), row_cuts, error));
                }
            }
        }
    }

    let (col_cuts, row_cuts, _) = best?;
    make_grid_candidate(
        profile_x,
        profile_y,
        col_cuts,
        row_cuts,
        expected_step_x,
        expected_step_y,
        width,
        height,
        "reconstruction",
        source_img,
    )
}

fn make_grid_candidate(
    profile_x: &[f64],
    profile_y: &[f64],
    col_cuts: Vec<usize>,
    row_cuts: Vec<usize>,
    expected_step_x: f64,
    expected_step_y: f64,
    width: usize,
    height: usize,
    source: &'static str,
    source_img: &RgbaImage,
) -> Option<GridCandidate> {
    if col_cuts.len() < 2 || row_cuts.len() < 2 {
        return None;
    }
    let score = score_grid_candidate(
        profile_x,
        profile_y,
        &col_cuts,
        &row_cuts,
        expected_step_x,
        expected_step_y,
        width,
        height,
        source_img,
    );
    if !score.is_finite() {
        return None;
    }
    Some(GridCandidate {
        col_cuts,
        row_cuts,
        source,
        score,
    })
}

fn score_grid_candidate(
    profile_x: &[f64],
    profile_y: &[f64],
    col_cuts: &[usize],
    row_cuts: &[usize],
    expected_step_x: f64,
    expected_step_y: f64,
    width: usize,
    height: usize,
    source_img: &RgbaImage,
) -> f64 {
    let edge_score =
        normalized_edge_score(profile_x, col_cuts) + normalized_edge_score(profile_y, row_cuts);
    let regularity_penalty = cut_irregularity(col_cuts) + cut_irregularity(row_cuts);
    let size_penalty = grid_size_penalty(col_cuts, expected_step_x, width)
        + grid_size_penalty(row_cuts, expected_step_y, height);
    let reconstruction_penalty = reconstruction_error(source_img, col_cuts, row_cuts);
    edge_score - 0.75 * regularity_penalty - 1.5 * size_penalty - 2.0 * reconstruction_penalty
}

fn normalized_edge_score(profile: &[f64], cuts: &[usize]) -> f64 {
    if profile.is_empty() || cuts.len() < 3 {
        return 0.0;
    }
    let mean = profile.iter().sum::<f64>() / profile.len() as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for &cut in &cuts[1..cuts.len() - 1] {
        if let Some(value) = profile.get(cut) {
            total += *value / mean;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn cut_irregularity(cuts: &[usize]) -> f64 {
    if cuts.len() < 3 {
        return 0.0;
    }
    let spans: Vec<f64> = cuts
        .windows(2)
        .filter_map(|w| (w[1] > w[0]).then_some((w[1] - w[0]) as f64))
        .collect();
    if spans.len() < 2 {
        return 0.0;
    }
    let mean = spans.iter().sum::<f64>() / spans.len() as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let variance = spans
        .iter()
        .map(|span| {
            let delta = span - mean;
            delta * delta
        })
        .sum::<f64>()
        / spans.len() as f64;
    variance.sqrt() / mean
}

fn grid_size_penalty(cuts: &[usize], expected_step: f64, limit: usize) -> f64 {
    if expected_step <= 0.0 || !expected_step.is_finite() || cuts.len() < 2 {
        return 0.0;
    }
    let expected_cells = limit as f64 / expected_step;
    let actual_cells = cuts.len().saturating_sub(1) as f64;
    if expected_cells <= 0.0 || actual_cells <= 0.0 {
        return 0.0;
    }
    let ratio = actual_cells / expected_cells;
    if (0.75..=1.25).contains(&ratio) {
        0.0
    } else {
        ratio.log2().abs()
    }
}

fn detect_strong_line_positions(profile: &[f64], limit: usize) -> Option<Vec<usize>> {
    if profile.len() < 3 || limit < 2 {
        return None;
    }
    let mean = profile.iter().sum::<f64>() / profile.len() as f64;
    let variance = profile
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / profile.len() as f64;
    let threshold = mean + variance.sqrt();

    let mut lines = vec![0];
    let mut i = 1usize;
    while i + 1 < profile.len() {
        if profile[i] < threshold {
            i += 1;
            continue;
        }
        let start = i;
        let mut best = i;
        let mut best_value = profile[i];
        while i + 1 < profile.len() && profile[i] >= threshold {
            if profile[i] > best_value {
                best = i;
                best_value = profile[i];
            }
            i += 1;
        }
        if best > 0 && best < limit {
            let last = *lines.last().unwrap();
            if best.saturating_sub(last) > 1 {
                lines.push(best);
            } else if let Some(last_mut) = lines.last_mut() {
                *last_mut = ((*last_mut + start + best) / 3).min(limit);
            }
        }
    }
    if *lines.last().unwrap() != limit {
        lines.push(limit);
    }
    (lines.len() >= 4).then_some(lines)
}

fn median_spacing(lines: &[usize]) -> Option<f64> {
    let mut diffs: Vec<usize> = lines
        .windows(2)
        .filter_map(|w| (w[1] > w[0]).then_some(w[1] - w[0]))
        .collect();
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_unstable();
    Some(diffs[diffs.len() / 2] as f64)
}

fn homogenize_lines(lines: &[usize], step: f64, limit: usize) -> Vec<usize> {
    if lines.len() < 2 || step < 1.0 || !step.is_finite() {
        return vec![0, limit];
    }
    let offset = lines
        .iter()
        .copied()
        .find(|line| *line > 0 && *line < limit)
        .unwrap_or(0) as f64;
    make_uniform_offset_cuts(step, offset, limit)
}

fn make_uniform_offset_cuts(step: f64, offset: f64, limit: usize) -> Vec<usize> {
    let mut cuts = vec![0, limit];
    let mut pos = offset.rem_euclid(step);
    if pos < 1.0 {
        pos += step;
    }
    while pos < limit as f64 {
        let cut = pos.round() as usize;
        if cut > 0 && cut < limit {
            cuts.push(cut);
        }
        pos += step;
    }
    sanitize_cuts(cuts, limit)
}

fn step_close(candidate: f64, expected: f64) -> bool {
    if candidate <= 0.0 || expected <= 0.0 || !candidate.is_finite() || !expected.is_finite() {
        return false;
    }
    let ratio = candidate / expected;
    (0.75..=1.25).contains(&ratio)
}

fn reconstruction_error(img: &RgbaImage, col_cuts: &[usize], row_cuts: &[usize]) -> f64 {
    if col_cuts.len() < 2 || row_cuts.len() < 2 {
        return f64::INFINITY;
    }
    let width = img.width() as usize;
    let height = img.height() as usize;
    let mut total_error = 0.0;
    let mut total_count = 0usize;

    for rows in row_cuts.windows(2) {
        let y0 = rows[0].min(height);
        let y1 = rows[1].min(height);
        if y1 <= y0 {
            continue;
        }
        for cols in col_cuts.windows(2) {
            let x0 = cols[0].min(width);
            let x1 = cols[1].min(width);
            if x1 <= x0 {
                continue;
            }

            let mut sums = [0.0f64; 3];
            let mut sq_sums = [0.0f64; 3];
            let mut count = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = img.get_pixel(x as u32, y as u32).0;
                    if p[3] == 0 {
                        continue;
                    }
                    for channel in 0..3 {
                        let value = p[channel] as f64 / 255.0;
                        sums[channel] += value;
                        sq_sums[channel] += value * value;
                    }
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            for channel in 0..3 {
                total_error += sq_sums[channel] - sums[channel] * sums[channel] / count as f64;
            }
            total_count += count;
        }
    }

    if total_count == 0 {
        return f64::INFINITY;
    }
    let cells = col_cuts.len().saturating_sub(1) * row_cuts.len().saturating_sub(1);
    total_error / total_count as f64 + 0.02 * cells as f64 / total_count as f64
}

// Tried uniform grid instead of an elastic-ish walker, but the result was a bit worse.
// Keeping the walker for now. But some distortions might happen...
fn walk(profile: &[f64], step_size: f64, limit: usize, config: &Config) -> Result<Vec<usize>> {
    if profile.is_empty() {
        return Err(PixelSnapperError::ProcessingError(
            "Cannot walk on empty profile".to_string(),
        ));
    }

    let mut cuts = vec![0];
    let mut current_pos = 0.0;
    let search_window =
        (step_size * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val: f64 = profile.iter().sum::<f64>() / profile.len() as f64;

    while current_pos < limit as f64 {
        let target = current_pos + step_size;
        if target >= limit as f64 {
            cuts.push(limit);
            break;
        }

        let start_search = ((target - search_window) as usize).max((current_pos + 1.0) as usize);
        let end_search = ((target + search_window) as usize).min(limit);

        if end_search <= start_search {
            current_pos = target;
            continue;
        }

        let mut max_val = -1.0;
        let mut max_idx = start_search;
        for i in start_search..end_search {
            if profile[i] > max_val {
                max_val = profile[i];
                max_idx = i;
            }
        }

        if max_val > mean_val * config.walker_strength_threshold {
            cuts.push(max_idx);
            current_pos = max_idx as f64;
        } else {
            cuts.push(target as usize);
            current_pos = target;
        }
    }
    Ok(cuts)
}

fn stabilize_cuts(
    profile: &[f64],
    cuts: Vec<usize>,
    limit: usize,
    sibling_cuts: &[usize],
    sibling_limit: usize,
    config: &Config,
) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }

    let cuts = sanitize_cuts(cuts, limit);
    let min_required = config.min_cuts_per_axis.max(2).min(limit.saturating_add(1));
    let axis_cells = cuts.len().saturating_sub(1);
    let sibling_cells = sibling_cuts.len().saturating_sub(1);
    let sibling_has_grid =
        sibling_limit > 0 && sibling_cells >= min_required.saturating_sub(1) && sibling_cells > 0;
    let steps_skewed = sibling_has_grid && axis_cells > 0 && {
        let axis_step = limit as f64 / axis_cells as f64;
        let sibling_step = sibling_limit as f64 / sibling_cells as f64;
        let step_ratio = axis_step / sibling_step;
        step_ratio > config.max_step_ratio || step_ratio < 1.0 / config.max_step_ratio
    };
    let has_enough = cuts.len() >= min_required;

    if has_enough && !steps_skewed {
        return cuts;
    }

    let mut target_step = if sibling_has_grid {
        sibling_limit as f64 / sibling_cells as f64
    } else if config.fallback_target_segments > 1 {
        limit as f64 / config.fallback_target_segments as f64
    } else if axis_cells > 0 {
        limit as f64 / axis_cells as f64
    } else {
        limit as f64
    };
    if !target_step.is_finite() || target_step <= 0.0 {
        target_step = 1.0;
    }

    snap_uniform_cuts(profile, limit, target_step, config, min_required)
}

fn sanitize_cuts(mut cuts: Vec<usize>, limit: usize) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }

    let mut has_zero = false;
    let mut has_limit = false;

    for value in cuts.iter_mut() {
        if *value == 0 {
            has_zero = true;
        }
        if *value >= limit {
            *value = limit;
        }
        if *value == limit {
            has_limit = true;
        }
    }

    if !has_zero {
        cuts.push(0);
    }
    if !has_limit {
        cuts.push(limit);
    }

    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

fn snap_uniform_cuts(
    profile: &[f64],
    limit: usize,
    target_step: f64,
    config: &Config,
    min_required: usize,
) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }
    if limit == 1 {
        return vec![0, 1];
    }

    // Get desired cells
    let mut desired_cells = if target_step.is_finite() && target_step > 0.0 {
        (limit as f64 / target_step).round() as usize
    } else {
        0
    };
    desired_cells = desired_cells
        .max(min_required.saturating_sub(1))
        .max(1)
        .min(limit);

    let cell_width = limit as f64 / desired_cells as f64;
    let search_window =
        (cell_width * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val = if profile.is_empty() {
        0.0
    } else {
        profile.iter().sum::<f64>() / profile.len() as f64
    };

    let mut cuts = Vec::with_capacity(desired_cells + 1);
    cuts.push(0);
    for idx in 1..desired_cells {
        let target = cell_width * idx as f64;
        let prev = *cuts.last().unwrap();
        if prev + 1 >= limit {
            break;
        }
        let mut start = ((target - search_window).floor() as isize)
            .max(prev as isize + 1)
            .max(0);
        let mut end = ((target + search_window).ceil() as isize).min(limit as isize - 1);
        if end < start {
            start = prev as isize + 1;
            end = start;
        }
        let start = start as usize;
        let end = end as usize;
        let mut best_idx = start.min(profile.len().saturating_sub(1));
        let mut best_val = -1.0;
        for i in start..=end.min(profile.len().saturating_sub(1)) {
            let v = profile.get(i).copied().unwrap_or(0.0);
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        let strength_threshold = mean_val * config.walker_strength_threshold;
        if best_val < strength_threshold {
            let mut fallback_idx = target.round() as isize;
            if fallback_idx <= prev as isize {
                fallback_idx = prev as isize + 1;
            }
            if fallback_idx >= limit as isize {
                fallback_idx = (limit as isize - 1).max(prev as isize + 1);
            }
            best_idx = fallback_idx as usize;
        }
        cuts.push(best_idx);
    }
    if *cuts.last().unwrap() != limit {
        cuts.push(limit);
    }
    cuts = sanitize_cuts(cuts, limit);
    cuts
}

fn snap_grid_size_cuts(
    profile: &[f64],
    limit: usize,
    target_cells: usize,
    config: &Config,
) -> Vec<usize> {
    debug_assert!(target_cells > 0);
    debug_assert!(target_cells <= limit);

    let cell_width = limit as f64 / target_cells as f64;
    let search_window =
        (cell_width * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val = if profile.is_empty() {
        0.0
    } else {
        profile.iter().sum::<f64>() / profile.len() as f64
    };
    let strength_threshold = mean_val * config.walker_strength_threshold;

    let mut cuts = Vec::with_capacity(target_cells + 1);
    cuts.push(0);

    for idx in 1..target_cells {
        let target = cell_width * idx as f64;
        let prev = *cuts.last().unwrap();
        let min_cut = prev + 1;
        let max_cut = limit - (target_cells - idx);

        let start = ((target - search_window).floor() as isize)
            .max(min_cut as isize)
            .max(0) as usize;
        let end = ((target + search_window).ceil() as isize)
            .min(max_cut as isize)
            .max(start as isize) as usize;

        let mut best_idx = target.round().clamp(min_cut as f64, max_cut as f64) as usize;
        let mut best_val = -1.0;
        for i in start..=end.min(profile.len().saturating_sub(1)) {
            let v = profile.get(i).copied().unwrap_or(0.0);
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        if best_val < strength_threshold {
            best_idx = target.round().clamp(min_cut as f64, max_cut as f64) as usize;
        }

        cuts.push(best_idx);
    }

    cuts.push(limit);
    cuts
}

fn resample(img: &RgbaImage, cols: &[usize], rows: &[usize], config: &Config) -> Result<RgbaImage> {
    if cols.len() < 2 || rows.len() < 2 {
        return Err(PixelSnapperError::ProcessingError(
            "Insufficient grid cuts for resampling".to_string(),
        ));
    }

    let out_w = (cols.len().max(1) - 1) as u32;
    let out_h = (rows.len().max(1) - 1) as u32;
    let mut final_img: RgbaImage = ImageBuffer::new(out_w, out_h);
    let img_w = img.width() as usize;
    let img_h = img.height() as usize;

    for (y_i, w_y) in rows.windows(2).enumerate() {
        for (x_i, w_x) in cols.windows(2).enumerate() {
            let ys = w_y[0].min(img_h);
            let ye = w_y[1].min(img_h);
            let xs = w_x[0].min(img_w);
            let xe = w_x[1].min(img_w);

            if xe <= xs || ye <= ys {
                continue;
            }

            let best_pixel = match config.resample_mode {
                ResampleMode::Majority => resample_majority(img, xs, xe, ys, ye),
                ResampleMode::Center => resample_center(img, xs, xe, ys, ye),
                ResampleMode::Mean => resample_mean(img, xs, xe, ys, ye),
                ResampleMode::EdgeAware => {
                    resample_edge_aware(img, xs, xe, ys, ye, config.edge_weight)
                }
                ResampleMode::PaletteAware => {
                    resample_palette_aware(img, xs, xe, ys, ye, config.palette.as_deref().unwrap())
                }
            };

            final_img.put_pixel(x_i as u32, y_i as u32, Rgba(best_pixel));
        }
    }
    Ok(final_img)
}

fn resample_majority(img: &RgbaImage, xs: usize, xe: usize, ys: usize, ye: usize) -> [u8; 4] {
    let mut counts: HashMap<[u8; 4], usize> = HashMap::new();

    for y in ys..ye {
        for x in xs..xe {
            let p = img.get_pixel(x as u32, y as u32).0;
            *counts.entry(p).or_insert(0) += 1;
        }
    }

    let mut candidates: Vec<([u8; 4], usize)> = counts.into_iter().collect();
    candidates.sort_by(|a, b| {
        let count_cmp = b.1.cmp(&a.1);
        if count_cmp == Ordering::Equal {
            a.0.cmp(&b.0)
        } else {
            count_cmp
        }
    });

    candidates
        .first()
        .map(|winner| winner.0)
        .unwrap_or([0, 0, 0, 0])
}

fn resample_center(img: &RgbaImage, xs: usize, xe: usize, ys: usize, ye: usize) -> [u8; 4] {
    let cx = ((xs + xe.saturating_sub(1)) / 2).min(img.width().saturating_sub(1) as usize);
    let cy = ((ys + ye.saturating_sub(1)) / 2).min(img.height().saturating_sub(1) as usize);
    let pixel = img.get_pixel(cx as u32, cy as u32).0;
    if pixel[3] == 0 {
        resample_majority(img, xs, xe, ys, ye)
    } else {
        pixel
    }
}

fn resample_mean(img: &RgbaImage, xs: usize, xe: usize, ys: usize, ye: usize) -> [u8; 4] {
    let mut weighted_rgb = [0.0f64; 3];
    let mut total_alpha = 0.0f64;
    let mut alpha_sum = 0.0f64;
    let mut count = 0usize;

    for y in ys..ye {
        for x in xs..xe {
            let p = img.get_pixel(x as u32, y as u32).0;
            let alpha = p[3] as f64 / 255.0;
            weighted_rgb[0] += p[0] as f64 * alpha;
            weighted_rgb[1] += p[1] as f64 * alpha;
            weighted_rgb[2] += p[2] as f64 * alpha;
            total_alpha += alpha;
            alpha_sum += p[3] as f64;
            count += 1;
        }
    }

    if total_alpha <= 0.0 || count == 0 {
        return [0, 0, 0, 0];
    }

    [
        clamp_u8(weighted_rgb[0] / total_alpha),
        clamp_u8(weighted_rgb[1] / total_alpha),
        clamp_u8(weighted_rgb[2] / total_alpha),
        clamp_u8(alpha_sum / count as f64),
    ]
}

fn resample_edge_aware(
    img: &RgbaImage,
    xs: usize,
    xe: usize,
    ys: usize,
    ye: usize,
    edge_weight: f64,
) -> [u8; 4] {
    let mut counts: HashMap<[u8; 4], f64> = HashMap::new();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let edge_weight = edge_weight.max(0.0);

    for y in ys..ye {
        for x in xs..xe {
            let p = img.get_pixel(x as u32, y as u32).0;
            let center = luminance(p);
            let left = luminance(img.get_pixel(x.saturating_sub(1) as u32, y as u32).0);
            let right = luminance(img.get_pixel((x + 1).min(width - 1) as u32, y as u32).0);
            let up = luminance(img.get_pixel(x as u32, y.saturating_sub(1) as u32).0);
            let down = luminance(img.get_pixel(x as u32, (y + 1).min(height - 1) as u32).0);
            let grad = (center - left).abs()
                + (center - right).abs()
                + (center - up).abs()
                + (center - down).abs();
            let normalized = (grad / (4.0 * 255.0)).min(1.0);
            let vote_weight = 1.0 + edge_weight * normalized;
            *counts.entry(p).or_insert(0.0) += vote_weight;
        }
    }

    let mut candidates: Vec<([u8; 4], f64)> = counts.into_iter().collect();
    candidates.sort_by(|a, b| {
        let count_cmp = b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal);
        if count_cmp == Ordering::Equal {
            a.0.cmp(&b.0)
        } else {
            count_cmp
        }
    });

    candidates
        .first()
        .map(|winner| winner.0)
        .unwrap_or([0, 0, 0, 0])
}

fn resample_palette_aware(
    img: &RgbaImage,
    xs: usize,
    xe: usize,
    ys: usize,
    ye: usize,
    palette: &[[u8; 3]],
) -> [u8; 4] {
    let mean = resample_mean_rgb_alpha(img, xs, xe, ys, ye);
    let Some((rgb, alpha)) = mean else {
        return [0, 0, 0, 0];
    };
    let best = nearest_palette_color(rgb, palette);
    [best[0], best[1], best[2], alpha]
}

fn resample_mean_rgb_alpha(
    img: &RgbaImage,
    xs: usize,
    xe: usize,
    ys: usize,
    ye: usize,
) -> Option<([f64; 3], u8)> {
    let mut weighted_rgb = [0.0f64; 3];
    let mut total_alpha = 0.0f64;
    let mut alpha_sum = 0.0f64;
    let mut count = 0usize;

    for y in ys..ye {
        for x in xs..xe {
            let p = img.get_pixel(x as u32, y as u32).0;
            let alpha = p[3] as f64 / 255.0;
            weighted_rgb[0] += p[0] as f64 * alpha;
            weighted_rgb[1] += p[1] as f64 * alpha;
            weighted_rgb[2] += p[2] as f64 * alpha;
            total_alpha += alpha;
            alpha_sum += p[3] as f64;
            count += 1;
        }
    }

    if total_alpha <= 0.0 || count == 0 {
        return None;
    }

    Some((
        [
            weighted_rgb[0] / total_alpha,
            weighted_rgb[1] / total_alpha,
            weighted_rgb[2] / total_alpha,
        ],
        clamp_u8(alpha_sum / count as f64),
    ))
}

fn luminance(pixel: [u8; 4]) -> f64 {
    if pixel[3] == 0 {
        0.0
    } else {
        0.299 * pixel[0] as f64 + 0.587 * pixel[1] as f64 + 0.114 * pixel[2] as f64
    }
}

fn clamp_u8(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}
