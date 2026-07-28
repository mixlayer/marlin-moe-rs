use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const SOURCES: &[&str] = &[
    "csrc/marlin_moe_c_api.cu",
    "vendor/marlin/moe/marlin_moe_wna16/ops.cu",
    "vendor/marlin/moe/marlin_moe_wna16/sm80_kernel_bfloat16_u4_bfloat16.cu",
    "vendor/marlin/moe/marlin_moe_wna16/sm80_kernel_bfloat16_u4b8_bfloat16.cu",
    "vendor/marlin/moe/marlin_moe_wna16/sm80_kernel_float16_u4_float16.cu",
    "vendor/marlin/moe/marlin_moe_wna16/sm80_kernel_float16_u4b8_float16.cu",
    "vendor/marlin/quantization/marlin/gptq_marlin_repack.cu",
    "vendor/marlin/moe/moe_support.cu",
    "vendor/marlin/moe/moe_sum_reduce.cu",
];

const FP8_SOURCES: &[&str] =
    &["vendor/marlin/moe/marlin_moe_wna16/sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu"];

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    for variable in [
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_PATH",
        "CUDA_COMPUTE_CAP",
        "MARLIN_MOE_ARCHS",
        "MARLIN_MOE_BUILD_DIR",
        "CANDLE_NVCC_CCBIN",
        "NVCC_THREADS",
        "MARLIN_MOE_NO_CCACHE",
        "MARLIN_MOE_PTXAS_VERBOSE",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    println!("cargo:rerun-if-changed=build.rs");
    for source in SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    for source in FP8_SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-changed=csrc/marlin_moe_c_api.h");
    println!("cargo:rerun-if-changed=vendor/marlin");
    println!(
        "cargo:rustc-env=MARLIN_MOE_SOURCE_ROOT={}",
        manifest_dir.join("vendor/marlin").display()
    );

    let archs = match selected_archs() {
        Ok(archs) => archs,
        Err(message) => {
            println!("cargo:warning={message}; compiling unsupported-architecture Rust stubs");
            println!("cargo:rustc-cfg=feature=\"unsupported_arch\"");
            return;
        }
    };
    let enable_fp8 = archs.iter().all(|arch| matches!(arch, 120 | 121));
    if !enable_fp8 && archs.iter().any(|arch| matches!(arch, 120 | 121)) {
        panic!(
            "MARLIN_MOE_ARCHS cannot mix SM120/SM121 with SM80-SM90 because the FP8 selector \
             is architecture-gated; build separate archives for legacy and SM12x targets"
        );
    }

    let cuda_root = discover_cuda_root();
    let nvcc = cuda_root.join("bin/nvcc");
    let build_dir = env::var_os("MARLIN_MOE_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("marlin-moe-build"));
    fs::create_dir_all(&build_dir)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", build_dir.display()));

    let includes = [
        manifest_dir.join("csrc"),
        manifest_dir.join("vendor/marlin"),
        manifest_dir.join("vendor/marlin/moe/marlin_moe_wna16"),
        cuda_root.join("include"),
    ];
    let sources = SOURCES
        .iter()
        .copied()
        .chain(FP8_SOURCES.iter().copied().filter(|_| enable_fp8));
    let mut objects = Vec::with_capacity(SOURCES.len() + FP8_SOURCES.len());
    for (index, relative) in sources.enumerate() {
        let source = manifest_dir.join(relative);
        let object = build_dir.join(format!(
            "{index:02}_{}.o",
            source.file_stem().unwrap().to_string_lossy()
        ));
        compile(&nvcc, &source, &object, &includes, &archs, enable_fp8);
        objects.push(object);
    }

    let archive = build_dir.join("libmarlin_moe.a");
    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&archive).args(&objects);
    run(ar, "archive Marlin MoE objects");

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=marlin_moe");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_root.join("lib64").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_root.join("lib64/stubs").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn selected_archs() -> Result<Vec<u32>, String> {
    let raw = env::var("MARLIN_MOE_ARCHS")
        .ok()
        .or_else(|| env::var("CUDA_COMPUTE_CAP").ok())
        .unwrap_or_else(|| "90".to_string());
    let archs: Vec<u32> = raw
        .split(',')
        .map(|part| {
            part.trim()
                .to_ascii_lowercase()
                .trim_start_matches("sm_")
                .trim_start_matches("sm")
                .replace('.', "")
                .trim_end_matches('a')
                .parse::<u32>()
                .map_err(|_| format!("invalid Marlin architecture {part:?}"))
        })
        .collect::<Result<_, _>>()?;
    if archs.is_empty()
        || archs
            .iter()
            .any(|arch| !matches!(arch, 80 | 86 | 89 | 90 | 120 | 121))
    {
        return Err(format!(
            "Marlin MoE supports SM80, SM86, SM89, SM90, SM120, and SM121; requested {raw:?}"
        ));
    }
    Ok(archs)
}

fn discover_cuda_root() -> PathBuf {
    for variable in ["CUDA_HOME", "CUDA_ROOT", "CUDA_PATH"] {
        if let Some(path) = env::var_os(variable) {
            return validate_cuda_root(PathBuf::from(path), variable);
        }
    }
    let nvcc = find_in_path("nvcc")
        .unwrap_or_else(|| panic!("nvcc not found; set CUDA_HOME, CUDA_ROOT, or CUDA_PATH"));
    let root = nvcc.parent().and_then(Path::parent).unwrap().to_path_buf();
    validate_cuda_root(root, "PATH")
}

fn validate_cuda_root(path: PathBuf, source: &str) -> PathBuf {
    let path = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("{source} CUDA root {}: {error}", path.display()));
    if !path.join("bin/nvcc").is_file() || !path.join("include/cuda_runtime_api.h").is_file() {
        panic!(
            "{source} does not identify a CUDA Toolkit at {}",
            path.display()
        );
    }
    path
}

fn find_in_path(program: &str) -> Option<PathBuf> {
    env::split_paths(&env::var_os("PATH")?)
        .map(|directory| directory.join(program))
        .find(|candidate| candidate.is_file())
}

fn compile(
    nvcc: &Path,
    source: &Path,
    object: &Path,
    includes: &[PathBuf],
    archs: &[u32],
    enable_fp8: bool,
) {
    let mut command = if !truthy("MARLIN_MOE_NO_CCACHE") {
        find_in_path("sccache")
            .or_else(|| find_in_path("ccache"))
            .map(|cache| {
                let mut command = Command::new(cache);
                command.arg(nvcc);
                command
            })
            .unwrap_or_else(|| Command::new(nvcc))
    } else {
        Command::new(nvcc)
    };
    command.arg("-c").arg(source).arg("-o").arg(object).args([
        "-O3",
        "-std=c++20",
        "-DNDEBUG",
        "-static-global-template-stub=false",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
        "-Xcompiler=-fPIC",
        "-Xcompiler=-fvisibility=hidden",
    ]);
    for arch in archs {
        command
            .arg("-gencode")
            .arg(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    if truthy("MARLIN_MOE_PTXAS_VERBOSE") {
        command.arg("--ptxas-options=-v");
    }
    if enable_fp8 {
        command.arg("-DMARLIN_MOE_ENABLE_FP8_MOE=1");
    }
    if let Some(ccbin) = env::var_os("CANDLE_NVCC_CCBIN") {
        command
            .arg("-allow-unsupported-compiler")
            .arg("-ccbin")
            .arg(ccbin);
    }
    if let Some(threads) = env::var_os("NVCC_THREADS") {
        command.arg("--threads").arg(threads);
    }
    for include in includes {
        command.arg("-I").arg(include);
    }
    run(command, &format!("compile {}", source.display()));
}

fn truthy(variable: &str) -> bool {
    env::var(variable)
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no"
            )
        })
        .unwrap_or(false)
}

fn run(mut command: Command, context: &str) {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to {context}: {error}"));
    if !output.status.success() {
        panic!(
            "failed to {context}\ncommand: {command:?}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
