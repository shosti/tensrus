use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
