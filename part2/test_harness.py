import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
from neuronxcc.nki import benchmark

from conv2d import fused_conv2d_maxpool as conv2d

from conv2d_numpy import conv2d_cpu_torch
import argparse

import subprocess

def save_trace(profile_name):
    """Run the profiler and save the NEFF and NTFF files with the specified name."""
    subprocess.run(
        [
            "neuron-profile",
            "capture",
            "-n",
            profile_name + ".neff",
            "-s",
            profile_name + ".ntff",
        ],
        check=True,
    )

    print(
        f"\nNEFF / NTFF files generated with names: {profile_name + '.neff'}, {profile_name + '.ntff'}"
    )


def test_correctness_conv2d_kernel(
    kernel,
    simulate=False,
    use_larger_images=False,
    use_bias=False,
    use_maxpool=False,
):
    if not simulate:
        kernel = baremetal(kernel)
    ref_impl = conv2d_cpu_torch

    input_channels_list = [128, 256]
    output_channels_list = [128, 256]
    kernel_size_list = [3]
    batch_size_list = [4]
    image_dims_list = [(32, 16)]
    pool_size = 2 if use_maxpool else 1

    if use_larger_images:
        input_channels_list = [256]
        output_channels_list = [256]
        image_dims_list = [(224, 224)]

    for input_channels in input_channels_list:
        for output_channels in output_channels_list:
            for kernel_size in kernel_size_list:
                for batch_size in batch_size_list:
                    for image_dims in image_dims_list:
                        X = np.random.rand(
                            batch_size, input_channels, image_dims[0], image_dims[1]
                        ).astype(np.float32)
                        W = np.random.rand(
                            output_channels, input_channels, kernel_size, kernel_size
                        ).astype(np.float32)
                        bias = (
                            np.zeros(output_channels).astype(np.float32)
                            if not use_bias
                            else np.random.rand(output_channels).astype(np.float32)
                        )

                        args = [X, W, bias]
                        kwargs = {"pool_size": pool_size}

                        out = kernel(*args, **kwargs)
                        out_ref = ref_impl(*args, **kwargs)

                        if not np.allclose(out, out_ref):
                            print(
                                f"Output mismatch for {input_channels=}, {output_channels=}, {kernel_size=}, "
                                f"{batch_size=}, {image_dims=}, {use_bias=}, {use_maxpool=}"
                            )
                            return False

    return True


def test_performance_conv2d_kernel(
    kernel,
    dtype=np.float32,
    batch_size=1,
    in_channels=256,
    out_channels=256,
    image_height=224,
    image_width=224,
    kernel_height=3,
    kernel_width=3,
    pool_size=1,
    profile=None
):
    # a performance requirement map (dtype, image_height) ->
    # [relaxed performance threshold, optimized performance threshold]
    performance_requirements_by_dtype_size = {
        (np.float32, 224): [4964, 4596],
        (np.float16, 224): [1365, 1002],
        (np.float32, 32): [110, 110],
        (np.float16, 32): [84, 84],
    }

    X = np.random.rand(batch_size, in_channels, image_height, image_width).astype(dtype)
    W = np.random.rand(out_channels, in_channels, kernel_height, kernel_width).astype(
        dtype
    )
    bias = np.random.rand(out_channels).astype(dtype)

    args = [X, W, bias]
    kwargs = {"pool_size": pool_size}

    bench_kwargs = {}
    if profile:
        bench_kwargs = {
            "save_neff_name": profile,
            "additional_compile_opt": "--disable-dge",
        }
    bench_func = nki.benchmark(warmup=5, iters=20, **bench_kwargs)(kernel)
    bench_func(*args, **kwargs)
    p99_us = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)
    print(f"\n\nExecution Time for student implementation: {p99_us} μs")

    if p99_us > (thresh := performance_requirements_by_dtype_size[(dtype, image_height)][0]):
        print(f"Performance requirement not met: must be under {thresh} μs")
        return False, False
    elif p99_us > (thresh := performance_requirements_by_dtype_size[(dtype, image_height)][1]):
        print(f"Performance requirement partially met (90% credit): for full credit, must be under {thresh} μs")
        return True, False
    else:
        return True, True


def get_performance_score(test_result, total_score):
    relaxed_result, optimized_result = test_result
    if optimized_result:
        print("Performance test passed 😍")
        return total_score
    elif relaxed_result:
        print("Can you make it faster? 🧐")
        # students get most of the score with meeting the relaxed time constraint
        return total_score * 0.95
    else:
        print("Performance test failed 😢")
        return 0


# write a function g which when passed a function f, returns a new function that when called with some *args
# and **kwargs, calls nki.simulate_kernel(f, *args, **kwargs) and returns the result
def simulate_kernel_wrapper(kernel):
    def temp_func(*args, **kwargs):
        return nki.simulate_kernel(kernel, *args, **kwargs)

    return temp_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_maxpool", action="store_true", help="Run the kernel with pool_size=2"
    )
    parser.add_argument(
        "--profile", type=str, default=None, help="File to save the .neff file"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use nki.simulate_kernel to run student implementation on CPU",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generation"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.simulate:
        conv2d = simulate_kernel_wrapper(conv2d)

    correctness_score = 0.0
    performance_score = 0.0
    ec = 0.0

    # --------- CORRECTNESS TESTS ---------
    correctness_tests = [
        {
            "use_larger_images": False,
            "use_bias": False,
            "use_maxpool": False,
        },
        {
            "use_larger_images": True,
            "use_bias": False,
            "use_maxpool": False,
        },
        {
            "use_larger_images": True,
            "use_bias": True,
            "use_maxpool": False,
        },
    ]
    if args.test_maxpool:
        correctness_tests.append({
            "use_larger_images": True,
            "use_bias": True,
            "use_maxpool": True,
        })
    
    for test_case in correctness_tests:
        print("\nRunning correctness test for conv2d kernel with "
              f"{'larger' if test_case['use_larger_images'] else 'smaller'} image"
              f"{' + bias' if test_case['use_bias'] else ''}"
              f"{' + maxpool' if test_case['use_maxpool'] else ''}"
              f"{' [simulated]' if args.simulate else ''}...", end=" ", flush=True)
            
        test_result = test_correctness_conv2d_kernel(conv2d, simulate=args.simulate, **test_case)
        if test_result:
            correctness_score += 2.5
            print("Passed 😎")
        else:
            print("Failed 😢")
    
    if correctness_score < 2.5 * len(correctness_tests):
        print("Correctness failed, skipping performance tests.")
        exit()
    
    # --------- PERFORMANCE TESTS ---------
    performance_tests = [
        {
            "pool_size": 1,
            "dtype": np.float32,
        },
        {
            "pool_size": 1,
            "dtype": np.float16,
        },
    ]
    if args.test_maxpool:
        performance_tests.extend([{
            "pool_size": 2,
            "dtype": np.float32,
        }, {
            "pool_size": 2,
            "dtype": np.float16,
        }])
    
    for test_case in performance_tests:
        pool_str = "with maxpool" if test_case['pool_size'] == 2 else "no maxpool"
        dtype_str = "float16" if test_case['dtype'] == np.float16 else "float32"
        print(f"\nComparing performance with reference kernel ({pool_str}, {dtype_str})...", end=" ", flush=True)

        profile = None
        if args.profile is not None:
            profile = f"{args.profile}{'_pool' if test_case['pool_size'] == 2 else ''}_{dtype_str}.neff"
        
        test_result = test_performance_conv2d_kernel(conv2d, profile=profile, **test_case)
        performance_score += get_performance_score(test_result, 17.5 if test_case['pool_size'] == 1 else 7.5)

        if profile:
            save_trace(profile.replace(".neff", ""))

    # --------- EXTRA CREDIT TESTS ---------
    ec_tests = [test | {"image_height": 32, "image_width": 16} for test in performance_tests]
    for test_case in ec_tests:
        pool_str = "with maxpool" if test_case['pool_size'] == 2 else "no maxpool"
        dtype_str = "float16" if test_case['dtype'] == np.float16 else "float32"
        print(f"\nComparing performance with reference kernel ({pool_str},"
              f" {dtype_str}, smaller image)... [EC] ", end=" ", flush=True)

        profile = None
        if args.profile is not None:
            profile = f"{args.profile}{'_pool' if test_case['pool_size'] == 2 else ''}_{dtype_str}_smaller.neff"
        
        test_result = test_performance_conv2d_kernel(conv2d, profile=profile, **test_case)
        ec += get_performance_score(test_result, 1.25)

        if profile:
            save_trace(profile.replace(".neff", ""))
    
    print(
        f"Your final score is {'' if args.test_maxpool else '(without maxpool)'}: ",
        f"{correctness_score + performance_score + ec} / {60.0 if args.test_maxpool else 42.5}"
    )
    print(
        f"Correctness: {correctness_score}\tTotal obtainable: {10.0 if args.test_maxpool else 7.5}"
    )
    print(
        f"Performance: {performance_score}\tTotal obtainable: {50.0 if args.test_maxpool else 35}"
    )
    print(f"Extra Credit: {ec}\tTotal obtainable: {5.0 if args.test_maxpool else 2.5}")
