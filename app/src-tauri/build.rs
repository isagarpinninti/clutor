use std::path::Path;

// The engine is plain C (../../c/src). It is compiled into this program, so the app needs no separate process and runs the same way on every
// system, Android included.
fn main() {
    let src = Path::new("../../c/src");
    let mut b = cc::Build::new();
    for f in ["gguf", "cache", "route", "kernel", "quant", "meta", "xs", "forward", "tokenizer", "stream", "chat", "bench", "vision"] {
        b.file(src.join(format!("{f}.c")));
    }
    b.include(src).opt_level(2).warnings(false).flag_if_supported("-std=gnu11");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    match arch.as_str() {
        // Every Apple-silicon Mac and every Android phone of the last several years has the int8 dot-product instruction the kernels use.
        "aarch64" => { b.flag_if_supported("-march=armv8.2-a+dotprod"); }
        "x86_64" => { b.flag_if_supported("-mavx2"); b.flag_if_supported("-mfma"); b.flag_if_supported("-mf16c"); }
        _ => {}
    }
    b.compile("clutor_engine");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os == "linux" { println!("cargo:rustc-link-lib=m"); println!("cargo:rustc-link-lib=pthread"); }
    if os == "windows" { println!("cargo:rustc-link-lib=psapi"); }
    println!("cargo:rerun-if-changed=../../c/src");
    // The image engine (stable-diffusion.cpp, MIT) is linked in when its built libraries are found: CLUTOR_SD_DIR, else a checkout next to this repository. Without them the app builds
    // without it and reports that the image engine is not included.
    println!("cargo::rustc-check-cfg=cfg(has_image)");
    println!("cargo:rerun-if-env-changed=CLUTOR_SD_DIR");
    println!("cargo:rerun-if-changed=image");
    let sd = std::env::var("CLUTOR_SD_DIR").ok().map(std::path::PathBuf::from).unwrap_or_else(|| Path::new("../../../clutor-tools/stable-diffusion.cpp").to_path_buf());
    let lib = sd.join("build").join("libstable-diffusion.a");
    if os == "macos" && arch == "aarch64" && lib.exists() {
        let mut s = cc::Build::new();
        s.file("image/clutor_image.c").include("image").include(sd.join("include")).opt_level(2).warnings(false).flag_if_supported("-std=gnu11");
        s.compile("clutor_image");
        let g = sd.join("build").join("ggml").join("src");
        for dir in [sd.join("build"), g.clone(), g.join("ggml-metal"), g.join("ggml-blas")] { println!("cargo:rustc-link-search=native={}", dir.display()); }
        for l in ["stable-diffusion", "ggml", "ggml-metal", "ggml-blas", "ggml-cpu", "ggml-base"] { println!("cargo:rustc-link-lib=static={l}"); }
        for f in ["Accelerate", "Metal", "MetalKit", "Foundation", "CoreFoundation"] { println!("cargo:rustc-link-lib=framework={f}"); }
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=objc");
        println!("cargo:rustc-cfg=has_image");
    }
    tauri_build::build();
}
