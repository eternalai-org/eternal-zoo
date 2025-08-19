class LlamaCpp < Formula
  desc "LLM inference in C/C++"
  homepage "https://github.com/ggml-org/llama.cpp"
  # CMake uses Git to generate version information.
  url "https://github.com/ggml-org/llama.cpp.git",
      tag:      "b6190",
      revision: "ae532eac2c1df1d8edc3d2719145895b966de1bf"
  license "MIT"
  head "https://github.com/ggml-org/llama.cpp.git", branch: "master"

  # llama.cpp publishes new tags too often
  # Having multiple updates in one day is not very convenient
  # Update formula only after 10 new tags (1 update per ≈2 days)
  #
  # `throttle 10` doesn't work
  livecheck do
    url :stable
    regex(/^v?b(\d+(?:\.\d+)*0)$/i)
  end

  bottle do
    sha256 cellar: :any,                 arm64_sequoia: "eabf0a0538aefda2639dc924dafb15844a244e441c98a6783a9c1c4f9456bd1c"
    sha256 cellar: :any,                 arm64_sonoma:  "7450bfeda4e2f3d236700e71e9e6e35caa11788b29035b4d7b3dce2b06ca6273"
    sha256 cellar: :any,                 arm64_ventura: "a1512c9bcfef0037dbec4602d1a36bd7b94d4d3580ec5bb78ed26812386449d1"
    sha256 cellar: :any,                 sonoma:        "86379e3904656649ba16873dafe80e13b443d6f4addfdac23c4f3cffa02e1b1b"
    sha256 cellar: :any,                 ventura:       "48fca96a7658edc16651e9d08ff824cac386b4d98afc996aeafd64fdb8d59e6e"
    sha256 cellar: :any_skip_relocation, arm64_linux:   "376648435b2faf3cf8c0294af794fcb3d7ccf67c01f5820358243d96fee099b8"
    sha256 cellar: :any_skip_relocation, x86_64_linux:  "4360d76c331e115aef71d9683e25915388b71973df0d7e315d053490648a46a4"
  end

  depends_on "cmake" => :build
  depends_on "pkgconf" => :build
  uses_from_macos "curl"

  on_linux do
    depends_on "openblas"
  end

  def install
    args = %W[
      -DBUILD_SHARED_LIBS=ON
      -DCMAKE_INSTALL_RPATH=#{rpath}
      -DGGML_ACCELERATE=#{OS.mac? ? "ON" : "OFF"}
      -DGGML_ALL_WARNINGS=OFF
      -DGGML_BLAS=ON
      -DGGML_BLAS_VENDOR=#{OS.mac? ? "Apple" : "OpenBLAS"}
      -DGGML_CCACHE=OFF
      -DGGML_LTO=ON
      -DGGML_METAL=#{(OS.mac? && !Hardware::CPU.intel?) ? "ON" : "OFF"}
      -DGGML_METAL_EMBED_LIBRARY=#{OS.mac? ? "ON" : "OFF"}
      -DGGML_NATIVE=#{build.bottle? ? "OFF" : "ON"}
      -DLLAMA_ALL_WARNINGS=OFF
      -DLLAMA_CURL=ON
    ]
    args << "-DLLAMA_METAL_MACOSX_VERSION_MIN=#{MacOS.version}" if OS.mac?

    system "cmake", "-S", ".", "-B", "build", *args, *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"

    libexec.install bin.children
    bin.install_symlink libexec.children.select { |file|
                          file.executable? && !file.basename.to_s.start_with?("test-")
                        }
  end

  test do
    system libexec/"test-sampling"
    # The test below is flaky on slower hardware.
    return if OS.mac? && Hardware::CPU.intel? && MacOS.version <= :monterey

    system bin/"llama-cli", "--hf-repo", "ggml-org/tiny-llamas",
                            "-m", "stories260K.gguf",
                            "-n", "400", "-p", "I", "-ngl", "0"
  end
end