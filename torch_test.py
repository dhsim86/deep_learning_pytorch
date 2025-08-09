import torch
import platform
import os


def print_header(title):
    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50)


def print_kv(key, value, indent=0):
    print("  " * indent + f"{key:<20}: {value}")


def print_torch_info():
    print_header("PyTorch 환경 정보")
    print_kv("PyTorch 버전", torch.__version__)
    print_kv("설치 경로", os.path.dirname(torch.__file__))
    print_kv("Python 버전", platform.python_version())
    print_kv("OS", platform.platform())


def print_cuda_info():
    print_header("CUDA 환경 정보")
    print_kv("CUDA 사용 가능", torch.cuda.is_available())
    print_kv("CUDA 버전", torch.version.cuda)
    print_kv("디바이스 수", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\n  [GPU {i}] {torch.cuda.get_device_name(i)}")
            print_kv(
                "총 메모리(MB)",
                f"{torch.cuda.get_device_properties(i).total_memory // (1024 ** 2):,}",
                2,
            )
            print_kv(
                "멀티프로세서 수",
                torch.cuda.get_device_properties(i).multi_processor_count,
                2,
            )
            print_kv(
                "Compute Capability",
                f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
                2,
            )
            print_kv("현재 사용 중", torch.cuda.memory_allocated(i) // (1024**2), 2)
            print_kv("최대 할당", torch.cuda.max_memory_allocated(i) // (1024**2), 2)
    else:
        print("  CUDA를 지원하는 GPU가 없습니다.")


def print_cudnn_info():
    print_header("cuDNN 환경 정보")
    cudnn_available = torch.backends.cudnn.is_available()
    print_kv("cuDNN 사용 가능", cudnn_available)
    print_kv("cuDNN 버전", torch.backends.cudnn.version())


def print_env_info():
    print_header("추가 환경 정보")
    print_kv("PyTorch BLAS", torch.backends.openmp.is_available())
    print_kv("MKL 사용 가능", torch.backends.mkl.is_available())
    print_kv("OpenMP 사용 가능", torch.backends.openmp.is_available())


if __name__ == "__main__":
    print_torch_info()
    print_cuda_info()
    print_cudnn_info()
    print_env_info()
