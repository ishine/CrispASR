use std::env;

fn main() {
    let mut cfg = cmake::Config::new("..");
    cfg.define("BUILD_SHARED_LIBS", "OFF")
       .define("WHISPER_BUILD_EXAMPLES", "OFF")
       .define("WHISPER_BUILD_TESTS", "OFF");

    if cfg!(feature = "cuda") {
        cfg.define("GGML_CUDA", "ON");
    }
    if cfg!(feature = "metal") {
        cfg.define("GGML_METAL", "ON")
           .define("GGML_METAL_EMBED_LIBRARY", "ON");
    }
    if cfg!(feature = "vulkan") {
        cfg.define("GGML_VULKAN", "ON");
    }

    let dst = cfg.build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    // Backend .a files live in the cmake build tree, not the install prefix
    println!("cargo:rustc-link-search=native={}/build/src", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src", dst.display());
    println!("cargo:rustc-link-lib=static=whisper");
    // Backend model libraries (linked into libwhisper in shared builds,
    // but separate .a files in static builds)
    for lib in &[
        "parakeet", "canary", "canary_ctc", "cohere", "qwen3_asr",
        "voxtral", "voxtral4b", "granite_speech", "wav2vec2-ggml",
        "crispasr-core", "pyannote-seg", "silero-lid", "ctc-align",
    ] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    // Platform libs
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            if cfg!(feature = "metal") {
                println!("cargo:rustc-link-lib=framework=Metal");
                println!("cargo:rustc-link-lib=framework=Foundation");
                println!("cargo:rustc-link-lib=framework=MetalKit");
            }
        }
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=gomp"); // OpenMP
        }
        _ => {}
    }
}
