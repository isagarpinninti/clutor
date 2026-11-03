use std::path::Path;

// The engine is plain C (../../c/src). It is compiled into this program, so the app needs no separate process and runs the same way on every
// system, Android included.
fn main() {
    let src = Path::new("../../c/src");
    let mut b = cc::Build::new();
    for f in ["gguf", "cache", "route", "kernel", "quant", "meta", "xs", "forward", "tokenizer", "stream", "chat"] {
        b.file(src.join(format!("{f}.c")));
    }
    b.include(src).opt_level(2).warnings(false).flag_if_supported("-std=gnu11");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    match arch.as_str() {
        // Every Apple-silicon Mac and every Android phone of the last several years has the int8 dot-product instruction the kernels use.
        "aarch64" => { b.flag_if_supported("-march=armv8.2-a+dotprod"); }
        "x86_64" => { b.flag_if_supported("-mavx2"); b.flag_if_supported("-mfma"); }
        _ => {}
    }
    b.compile("clutor_engine");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os == "linux" { println!("cargo:rustc-link-lib=m"); println!("cargo:rustc-link-lib=pthread"); }
    if os == "windows" { println!("cargo:rustc-link-lib=psapi"); }
    println!("cargo:rerun-if-changed=../../c/src");
    tauri_build::build();
}
