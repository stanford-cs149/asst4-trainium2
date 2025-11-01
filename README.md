# Assignment 4: Programming a Machine Learning Accelerator #

**Due Thursday Nov 13, 11:59pm**

**100 points total**

## Overview ##

In this assignment, you will learn how to implement and optimize kernels for the [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/) architecture, which features multiple tensor-oriented accelerated processing engines as well as software-managed on-chip storage that provides these engines high-bandwidth access to data. 

The assignment is organized into two parts.  In part 1 you'll familiarize yourself with the Trainium architecture and data movement patterns by studying some simple kernels for vector addition and writing your own matrix transpose kernel. In part 2 you will implement a fused convolution layer for Trainium.

Overall, this assignment will:

1) Give you experience with the low-level details of tensor processing and managing on-chip SRAM on the accelerator.

2) Show the value of key locality-preserving optimizations like loop blocking and loop fusion.

## Environment Setup ##

You will be programming and testing your code on an AWS VM featuring Trainium accelerators. Please follow the instructions in [cloud_readme.md](cloud_readme.md) for setting up a machine to run the assignment.

Once you have logged in to your AWS machine, you should download the assignment starter code from the course Github using:

`git clone https://github.com/stanford-cs149/asst4-trainium2`

After downloading the Assignment 4 repository, move to the `asst4-trainium2` directory and **run the install script we have provided**:
```
cd asst4-trainium2
source install.sh
```
The install script will activate a Python [virtual environment](https://builtin.com/data-science/python-virtual-environment) with all the needed assignment dependencies. It will also modify your `~/.bashrc` file so the virtual environment is activated upon future logins to your machine. Finally, the script sets up your InfluxDB credentials so that you may use `neuron-profile`.

## Part 0: Getting familiar with Trainium and Neuron Core Architecture

### Trainium Architecture Overview

First, let's get you acquainted with Trainium.

The `Trn2.3xlarge` instance used in this assignment features a single Trainium device, which comprises eight NeuronCores. Each core is equipped with its own dedicated HBM (high-bandwidth memory), as seen in the images below. Each NeuronCore can be considered a standalone processing unit, which contains its own on-chip storage as well as a collection of specialized compute engines for performing 128x128 matrix operations (Tensor Engine), 128-wide vector operations (Vector Engine), etc. While each Trainium device has eight NeuronCores, in this assignment we will be writing kernels that execute on a single NeuronCore.

<p align="center">
  <img src="handout/trainium_chip.png" width=60% height=60%>
  <img src="handout/neuroncore_v3.png" width=30% height=30%>
</p>

More details on the four distinct compute engines that exist in a NeuronCore can be found [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html).

### Trainium Memory Hierarchy

In Assignment 3, one of the key concepts was learning about the GPU memory hierarchy presented by CUDA, where there was main host memory, GPU device global memory, per-thread block shared memory, and private per-CUDA-thread memory.  On Trainium, the memory hierarchy consists of four levels: **host memory (DRAM)**, **device memory (HBM)**, and two fast on-chip memory types, **SBUF (State Buffer)** and **PSUM (Partial Sum Buffer)**. In this assignment, we'll be writing kernels that only target device/on-chip memory, so we can ignore DRAM (which is external to the Trainium device) and focus on HBM, SBUF, and PSUM.

<p align="center">
  <img src="/handout/memory_hierarchy.png" width=80% height=80%>
</p>

* __HBM__ is high-bandwidth memory located on the Trainium device. HBM serves as the device's primary memory, offering large storage (96 GiB). Most data types created outside kernels (e.g. NumPy arrays) are allocated in HBM by default.
* __SBUF__ is on-chip storage on the NeuronCore. In comparison, SBUF is significantly smaller than HBM (28 MiB) but offers much higher bandwidth (~20x than HBM). The programmer must explicitly move data to and from SBUF in order to perform computations using the NeuronCore.
* __PSUM__ is a small, specialized memory bank (2 MiB) dedicated to holding matrix multiplication results produced by the tensor engine.

<p align="center">
  <img src="/handout/neuron_core.png" width=40% height=40%>
</p>

Recall that in a system that features a traditional data cache, decisions about what data from off-chip memories is replicated and stored in on-chip storage are made by the cache (based on cache organization and eviction policies). Software loads data at a given memory address, and the hardware is responsible for fetching that data from memory and managing what data is stored in the cache for efficient future access. In other words, from the perspective of software correctness, the cache does not exist--it is a hardware implementation detail. 

In contrast, the memories available to a NeuronCore are *software managed*. This means that a software must explicitly move data to and from these memories using data movement commands. Either the programmer must explicitly describe data movement in their program, or the NKI compiler must analyze the application and generate the appropriate data movement operations.  Some of the biggest challenges of efficiently using the NeuronCore architecture involve efficiently orchestrating the movement of data through the machine.

## Part 1: Learning the Neuron Kernel Interface with Vector Add and Matrix Transpose (30 points)

In this section, we introduce the basics of the Trainium programming model by providing several different implementations of an application that adds the elements of two vectors. We'll then write a simple kernel to transpose a 2D matrix.

The corresponding code is organized within the `/part1` directory. Specifically, the vector addition kernels discussed here can be found in `kernels.py`. Additionally, we provide a script, `run_benchmark.py`, which offers a convenient command-line interface for executing these kernels with different vector sizes. The script also includes an optional flag for collecting profiling metrics.

```
usage: run_benchmark.py [-h] --kernel {naive,tiled,stream,transpose} -n N [-m M] [--profile_name PROFILE_NAME]

options:
  -h, --help            show this help message and exit
  --kernel {naive,tiled,stream,transpose}
  -n N
  -m M
  --profile_name PROFILE_NAME
                        Name used to save .NEFF and .NTFF files
```

### NKI Programming Model:

The Neuron Kernel Interface (NKI) is a language and compiler for developing kernels that run on Trainium devices. NKI kernels are written in Python, and make use of three types of NKI operations:
1. **Loading data** from HBM to the on-chip SBUF.
2. **Computation** executed on the NeuronCore compute engines.
3. **Storing outputs** from SBUF back to HBM.

As an example, the following kernel defines how to perform vector addition using NKI. Note that the `@nki.jit` is a Python decorator that indicates that a function should be compiled to run on NeuronDevices, much like how the `__global__` function decorator in CUDA C++ designates that a function is to be compiled as a device-side function and run on the GPU.

Similar to how arguments to CUDA kernels are arrays in CUDA device global memory, the arguments to Python functions decorated with `@nki.jit` are tensors that reside in HBM accessible to the NeuronCore. The `@nki.compiler.skip_middle_end_transformations` decorator disables some compiler optimizations that can transform kernels in unexpected ways, which will make debugging easier.

In the following code, `a_vec` and `b_vec` are assumed to be length 128 vectors in HBM. (The code will not work for vectors that are larger than 128. We'll explain why shortly.)
```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_naive(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Allocate space for the input vectors in SBUF and copy them from HBM
    a_sbuf = nl.ndarray(shape=(a_vec.shape[0], 1), dtype=a_vec.dtype, buffer=nl.sbuf)
    b_sbuf = nl.ndarray(shape=(b_vec.shape[0], 1), dtype=b_vec.dtype, buffer=nl.sbuf)
    
    nisa.dma_copy(src=a_vec, dst=a_sbuf)
    nisa.dma_copy(src=b_vec, dst=b_sbuf)

    # Add the input vectors
    res = nisa.tensor_scalar(a_sbuf, nl.add, b_sbuf)

    # Store the result into HBM
    nisa.dma_copy(src=res, dst=out)

    return out
```

In the code above...

- `a_vec` and `b_vec` are NumPy arrays created outside the kernel residing in HBM.
- `a_sbuf` and `b_sbuf` are arrays explicitly allocated in SBUF with the same shape and dtype as `a_vec` and `b_vec`.
- `nisa.tensor_scalar(..., nl.add, ...)` performs vector addition using the vector engine. The signature `tensor_scalar` means that the second operand is expected to be a vector, i.e. of shape (N, 1), or a constant scalar, which makes it a bit faster than a general `tensor_tensor` operation.
- `nisa.dma_copy` moves the relevant data between HBM and SBUF.

<p align="center">
  <img src="/handout/sbuf_layout.png" width=60% height=60%>
</p>

**When looking at the code above, notice that NKI operations operate on tensors, not scalar values.** Specifically, the on-chip memories, SBUF and PSUM, store data that is arranged as 2D memory arrays. The first dimension of the 2D array is called the "partition dimension" `P`. The second dimension is referred to as the "free dimension" `F`.  NeuronCores are able to load and process data along the partition dimension in parallel, *but the architecture also places a restriction that the size of the partition dimension is 128 or smaller.* 
In other words, when loading a tensor from HBM to SBUF, the partition dimension of the tensor can be at most 128.  We will talk about the restrictions of the free dimension later.

As a result, in the code above, since `a_vec` and `b_vec` are 1D vectors, their only dimension is the partition dimension, and thus their size is limited to 128 elements.  In other words the code only works for vector sizes 128 or less.

### Step 1: Chunking Vectors to Parallelize Across 128 Compute Lanes

To fix the code to work for vectors with a size greater than 128, we need to load the vectors in chunks (subsets of the original tensor). 

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_tiled(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Get the total number of vector rows
    M = a_vec.shape[0]
    
    # TODO: You should modify this variable for Step 1
    ROW_CHUNK = 1

    # Loop over the total number of chunks, we can use affine_range
    # because there are no loop-carried dependencies
    for m in nl.affine_range(M // ROW_CHUNK):

        # Allocate row-chunk sized tiles for the input vectors
        a_tile = nl.ndarray((ROW_CHUNK, 1), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((ROW_CHUNK, 1), dtype=b_vec.dtype, buffer=nl.sbuf)
        
        # Load a chunk of rows
        nisa.dma_copy(src=a_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=a_tile)
        nisa.dma_copy(src=b_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=b_tile)

        # Add the row chunks together
        res = nisa.tensor_scalar(a_tile, nl.add, b_tile)

        # Store the result chunk into HBM
        nisa.dma_copy(src=res, dst=out[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])
    
    return out
```

The above example breaks the vector rows into single-element chunks (the chunk size is 1 element of the vector---yes, this is inefficient, we'll come back to this in a second). This is achieved by indexing the vector using the standard Python slicing syntax `Tensor[Index:Index:...]`. More details regarding Tensor indexing in NKI can be found [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html#nki-tensor-indexing). 

In the code above [affine_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.affine_range.html) used here generates a sequence of numbers for loop iterators, similar to Python’s `range` function, but it requires that there are no loop-carried dependencies across iterations. For cases where there are loop-carried dependencies, NKI also provides [sequential_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.sequential_range.html).

Normally, `affine_range` lets the NKI compiler more aggressively optimize loop iterations to allow for increased pipelining across compute engines. Since we're disabling compiler optimizations for transparency/reproducibility, however, both of these constructs are effectively the same.

**What you need to do:**
1. Run the above `vector_add_tiled` implementation where *row_chunk = 1* with a vector size of 25600 (*this may take a couple minutes*). You may do so using the following command:

   ```
   python run_benchmark.py --kernel tiled -n 25600
   ```

   What was the execution time in microseconds (μs)?

3. Remember that the maximum partition size (number of rows) that can be loaded at once on a NeuronDevice is 128. Inside `kernels.py`, change `vector_add_tiled` so that it uses *row_chunk = 128*. Record the execution time in microseconds (μs) for `vector_add_tiled` with the *row_chunk = 128* operating on a vector size of 25600. How much faster is `vector_add_tiled` on a vector size of 25600 when *row_chunk = 128* compared to when *row_chunk = 1*? Why do you think it is faster?  (*Hint:* you should think of the execution as loading `ROW_CHUNK` elements in parallel from HBM and then performing a `ROW_CHUNK` wide vector add on the vectors in SBUF.)
	
4. Try running `vector_add_tiled` on vector sizes of 25600 when *row_chunk = 256*. You should see an error. In one sentence, explain why you get an error when trying to run *row_chunk = 256*.

### Step 2a: Improved Data Streaming

So far, we have been able to exploit the fact that the Vector Engine can perform computation with all 128 vector lanes in parallel, with each lane streaming a single element to/from a single SBUF/PSUM partition.

However, we can improve performance further by streaming more elements across the free dimension. To do that, let's think more about Direct Memory Access (DMA) transfers. You should think of a DMA transfer, i.e. a call to `nisa.dma_copy`, as a single asynchronous operation that moves a block of data from HBM to SBUF or vice versa.

Each NeuronCore has 16 DMA engines that can all work on different data transfer operations in parallel. The caveat is that there is an overhead cost when setting up a DMA transfer and assigning DMA engines to work on them. In order to reduce this setup overhead, efficient implementations should aim to move a large amount of data in each transfer to amortize DMA transfer overhead.

Although the first dimension (partition dimension) of a SBUF tensor can be no greater than 128, the second dimension for a single SBUF vector instruction can be up to 64K elements. This means that it is possible to use a single instruction to load 128 * 64k = 8192k elements from HBM to SBUF. Furthermore, we can perform vector addition on two 8192k element SBUF tiles in a single `nisa.tensor_tensor` instruction. Therefore, rather than performing a `nisa.dma_copy` for each 128 element chunk of a vector, we should should instead move many 128-row chunks with each DMA transfer request. This streamlined approach allows us to amortize the setup time required for transferring data.

In order to improve DMA transfer overhead, we will need to reshape our vectors so they are two-dimensional tiles, rather than linearized arrays. In Assignment 3, we worked with CUDA thread blocks partitioned across an entire image, and in order to map CUDA threads to image pixels we flattened our grid by calculating a thread’s global linear index. You can think about the reshaping process for the NeuronCore as the inverse: the goal is to turn a single-dimension vector into a dense 2D matrix. NumPy comes with a built-in [reshape function](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) allowing you to reshape arrays into the shape of your choosing. 

<p align="center">
  <img src="/handout/non_reshaped_DMA.png" width=48% height=48%>
  <img src="/handout/reshaped_DMA.png" width=48% height=48%>
</p>


Take a look at `vector_add_stream`, which extends `vector_add_tiled` so that there are less DMA transfers:
```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_stream(a_vec, b_vec):

    # Get the total number of vector rows
    M = a_vec.shape[0]

    # TODO: You should modify this variable for Step 2a
    FREE_DIM = 2

    # The maximum size of our Partition Dimension
    PARTITION_DIM = 128

    a_vec_re = a_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    b_vec_re = b_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    out = nl.ndarray(shape=a_vec_re.shape, dtype=a_vec_re.dtype, buffer=nl.hbm)

    # Loop over the total number of tiles
    for m in nl.affine_range(M // (PARTITION_DIM * FREE_DIM)):

        # Allocate space for a reshaped tile
        a_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=b_vec.dtype, buffer=nl.sbuf)

        # Load the input tiles
        nisa.dma_copy(src=a_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=a_tile)
        nisa.dma_copy(src=b_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=b_tile)

        # Add the tiles together. Note that we must switch to tensor_tensor instead of tensor_scalar
        res = nisa.tensor_tensor(a_tile, b_tile, op=nl.add)

        # Store the result tile into HBM
        nisa.dma_copy(src=res, dst=out[:, m * FREE_DIM : (m + 1) * FREE_DIM])

    # Reshape the output vector into its original shape
    out = out.reshape(a_vec.shape)

    return out
```

**What you need to do:**
1. Run the above `vector_add_stream` implementation where *FREE_DIM = 2*. How many microseconds (μs) did it take to run for a vector size of 25600? How much faster is this compared to `vector_add_tiled` with *row_chunk = 128* from Step 1?
2. The current `vector_add_stream` implementation reduces the number of DMA transfers slightly, but the number of DMA transfers can be reduced further. Inside `kernels.py`, change the value of *FREE_DIM* for `vector_add_stream` to reduce the number of DMA transfers as much as possible on a vector of size 25600.

   What value of *FREE_DIM* did you choose? What was the execution time in microseconds (μs) on a vector size of 25600 for this value of *FREE_DIM*?
   
   How much faster is `vector_add_stream` with the *FREE_DIM* number you chose than `vector_add_stream` with *FREE_DIM = 2*? How much faster is `vector_add_stream` with the *FREE_DIM* number you chose than `vector_add_tiled` with *row_chunk = 128*?

### Step 2b: Learning to Use Neuron-Profile

There is a trade-off in choosing a tile's free dimension size:
1. Too small of a tile size exposes significant instruction overhead leading to inefficient engine execution.
2. Too large of a tile size often leads to inefficient pipelining between engines and high memory pressure in SBUF in cases of data reuse ("memory pressure" means that SBUF may fill up).

Currently, we have explored the benefits of increasing tile sizes to their maximum amount in order for us to amortize instruction overhead and DMA transfer setup / teardown. Now, we will explore why making the free dimension as large as possible is not always the best solution.

For this task, you will need to use the profiling tool for NeuronDevices: `neuron-profile`, which can provide detailed analysis of the performance of an application running on a NeuronCore. In order to run the profiling tool, you must make sure that you ran the install script as detailed in [Environment Setup](#environment-setup) and that you forwarded ports 3001 and 8086 when you ssh'd into your machine. To reiterate on the latter, the command you should have ran is:

 `ssh -i path/to/key_name.pem ubuntu@<public_dns_name> -L 3001:localhost:3001 -L 8086:localhost:8086`
 
 More details about why this is needed can be found in the [cloud_readme.md](/cloud_readme.md).

**What you need to do:**
1.  This time, we are going to increase the vector sizes by a factor of 10 so that instead of adding 25600 elements we will be adding 256000 elements. This will allow us to see trade offs that comes from dealing with tile sizes that are too large.  

    First, set *FREE_DIM = 2000* in `vector_add_stream`. Now, just like the prior steps we are going to execute our kernel, but this time we are going to save the compiled kernel 
    into a **.neff** file and the kernel execution trace into a **.ntff** trace file. Let's run `vector_add_stream` on a vector_size of 256000 and save the compiled kernel and 
    trace into files prefixed with `stream_256k_fd2k` with the following command:

    ```
    python run_benchmark.py --kernel stream -n 256000 --profile_name stream_256k_fd2k
    ```

    You should have generated two files: ***stream_256k_fd2k.neff*** and ***stream_256k_fd2k.ntff***. (You might see an error in stdout saying "hw profiler overview not found" -- this is safe to ignore, don't worry.)
    
    Now, using a similar workflow run `vector_add_stream` with *FREE_DIM = 1000* on a vector_size of 256000 and save the compiled kernel and trace into files prefixed with 
    `stream_256k_fd1k`.
2.  These generated files will allow us to collect kernel execution metrics using the `neuron-profile` tool. These profiling metrics will be very useful for analyzing the 
    performance of your kernels. Let's look at a brief summary of execution metrics for `vector_add_stream` with *FREE_DIM = 2000* by running the following command:

    ```
    neuron-profile view --output-format summary-text -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
    ```

    You will see a summarized output consisting of various execution metrics in alphabetical order. Let's look at two specific metrics: 
    
     * **dma_transfer_count**: The number of DMA transfers
     * **total_time**: Kernel execution time in seconds

    What was kernel execution time in seconds when *FREE_DIM = 2000*? How many DMA transfers were made when *FREE_DIM = 2000*?
    
    Using the same workflow as before, look at the summary of execution metrics when *FREE_DIM = 1000*.
    
    What was kernel execution time in seconds when *FREE_DIM = 1000*? How many DMA transfers were made when *FREE_DIM = 1000*?

3. Although the kernel with *FREE_DIM = 1000* had more DMA transfers, it was faster! Let's analyze why.
  
   We can dive deeper into the kernel execution metrics using the GUI functionality of `neuron-profile`. Let's launch the GUI for `vector_add_stream` with *FREE_DIM = 2000* by 
   running the following command:

   ```
   neuron-profile view -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
   ```

   After running the command, you will see an output like the following:

   `View profile at http://localhost:3001/profile/...`

   Paste this *http* link into a browser of your choice to view more in-depth profiler analytics. (Feel free to ignore any warnings that come up at the top of the page.)
   
> [!NOTE]
> You will only be able to view this if you have correctly forwarded ports 3001 and 8086 when you ssh'd into your machine.

   You should see a graph generated from the profiler depicting instructions issued to different engines laid out across time.
   
   To make things easier to see for our purposes, go to `View Settings` at the bottom and do the following:
   * Change `Instructions color group` to `Instruction Type`
   * Turn off `Show individual NeuronCore layout` under `Timeline display options`
   * Turn off `Show expanded DMA` under `DMA display options`
   * Click `Save` at the very bottom.
   
   After these steps, the profiler graph should look like this:

   ![Profiler GUI Example](/handout/profiler_gui.png)
   
   You can also hover over various events in the graph to see more info. Try hovering over events in the following categories:
   
   * **DMA-E79**: Shows DMA Engines moving input and output data to/from the appropriate buffers (count the number of instructions -- does this match the expected number of calls to `nisa.dma_copy`?)
   * **VectorE**: Shows the Vector Engine adding the two input vectors via `nisa.tensor_tensor` (this should be highlighted in green)
   * **Pending DMA Count**: Shows the number of pending DMA transfers over time
   * **DMA Throughput**: Shows the device bandwidth utilization over time

   Now, in your terminal press `ctrl-c` to exit the current `neuron-profile view`. Note that you can still view the GUI analytics for `vector_add_stream` with *FREE_DIM = 2000* in your browser as they have been temporally stored in a database. Following the same workflow, launch the GUI analytics for `vector_add_stream` with *FREE_DIM = 1000*.

4. After analyzing the GUI analytics graph for `vector_add_stream` with both *FREE_DIM = 2000* and *FREE_DIM = 1000*, briefly explain why FREE_DIM = 1000 has a faster execution 
   time than FREE_DIM = 2000 even though it required more DMA transfers (*Hint:* pipelining).

   Feel free to also play around with various functionalities in the `neuron-profile` GUI. You may also want to look at the `Summary` tab located at the bottom toolbar. This tab displays the same brief summary of execution metrics we saw when running  
   `neuron-profile view --output-format summary-text ...` in Question 2. Feel free to learn more about `neuron-profile` functionality from the [user guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html) and interesting performance metrics for NKI kernels from the [NKI performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_perf_guide.html).

### Step 3: Matrix Transpose
### Matrix Operations on a NeuronCore
Before you begin, we will demonstrate how to perform matrix operations on a NeuronCore. As discussed earlier, a NeuronCore is equipped with various compute engines, each optimized for specific types of arithmetic operations. The Tensor Engine on Trainium is specifically designed to accelerate these matrix operations, such as matrix multiplication and matrix transpose. 

<p align="center">
  <img src="/handout/tensor_engine.png" width=60% height=60%>
</p>

The above image depicts the architecture of the Tensor Engine. The Tensor Engine is built around a 128x128 [systolic processing array](https://gfxcourses.stanford.edu/cs149/fall25/lecture/proghardware/slide_10) which streams matrix data input from SBUF (on-chip storage) and writes the output to PSUM (also on-chip storage). Like SBUF, PSUM is fast on-chip memory, however it is much smaller than SBUF (2MiB vs 28 MiB) and serves a dedicated purpose of storing matrix multiplication results computed by the Tensor Engine. The Tensor Engine is able to read-add-write to every address in PSUM. Therefore, PSUM is useful when executing large matrix multiplications in a tiled manner, where the results of each matrix multiply are accumulated into the same output tile.

### Writing the kernel
Here you'll try writing your own baby kernel for transposing a matrix using the tensor engine before moving on to a more complicated kernel involving actual matmuls in Part 2. Take a look at the starter code in `kernels.py`. Your kernel should accept a single 2D tensor of shape (M, N) as input and return a 2D tensor of shape (N, M). The only restriction on M and N is that both are divisible by 128, the maximum partition dimension.

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def matrix_transpose(a_tensor):
    M, N = a_tensor.shape
    out = nl.ndarray((N, M), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    tile_dim = 128

    assert M % tile_dim == N % tile_dim == 0, "Matrix dimensions not divisible by tile dimension!"

    # TODO: Your implementation here. The only compute instruction you should use is `nisa.nc_transpose`.

    return out
```

To actually perform the transpose, you must call [nisa.nc_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_transpose.html#nki.isa.nc_transpose), which is a built-in instruction that uses the Tensor Engine to transpose tiles of size up to 128x128, storing the result in PSUM. You are **not** allowed to use other compute instructions, including `nisa.dma_tranpose` or `nl.transpose`. (Memory instructions, including `nisa.dma_copy` and `nl.ndarray`, are of course allowed.)

Since you will be transposing matrices much larger than 128x128, your kernel should manage the movement of data tiles to and from HBM/SBUF. It might be useful to revisit the vector addition kernels from earlier to see how they allocate and move data.

> [!TIP]
> `nisa.dma_copy` only works on tensors in SBUF/HBM. Since the output of `nisa.nc_transpose` is a PSUM tile, you'll need to copy it to SBUF first. You might find [`nisa.tensor_copy`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_copy.html#nki.isa.tensor_copy) useful for this.

**What you need to do:**
1.  Fill in the kernel with your implementation. Then test it on a 1024x1024 matrix by running
    ```
    python run_benchmark.py --kernel transpose -n 1024
    ```
    and record the execution time in microseconds (μs).
2. Without using a profiler, do you think your kernel is memory-bound or compute-bound? Explain your answer. Then, confirm it by profiling your code the same way you did with `vector_add_stream`. (You may include a screenshot, but please provide a written description of how it validates your answer.)
3.  **(Extra credit, 1 pt)** Optimize your implementation to minimize latency. To obtain credit, you should be able to hit <700 μs on a 4096x4096 transpose. Make sure you measure latency *without* passing `--profile_name` (the profiler changes execution time).

    Feel free to experiment with other APIs besides `nisa.nc_transpose` for this part. Please also submit a brief writeup explaining how you identified performance bottlenecks and addressed them.

## Part 2: Implementing a Fused Convolution - Max Pool Layer (70 points)

Now that you’ve learned how to efficiently move data on a NeuronCore, it is time to program an actual Trainium kernel yourself. In this section, your task is to implement a kernel that performs both convolution and an operation called "max pooling". As we discussed in class, these two operations are a fundamental component of modern Convolutional Neural Networks (CNNs), which are extensively used for computer vision tasks. An important detail is that your implementation of these two operations will be "fused", mean you will implement the computation on Trainium without dumping intermediate values to off-chip HBM. 

### An NKI Matrix Multiplication Kernel

Recall that the Vector Engine has the capability to operate on SBUF tiles of size (128, 64k). However, the Tensor Engine contains unique SBUF tile size constraints which differ to that of the Vector Engine. Suppose we want the Tensor Engine to perform the matrix multiplication C = A x B, where A and B are located in SBUF, and the result C is stored in PSUM. Trainium imposes the following constraints. 
  - Matrix A - the left-hand side tile - can be no bigger than (128, 128)
  - Matrix B - the right-hand side tile - can be no bigger than (128, 512).
  - The output tile C in PSUM is restricted to a size of (128, 512).

Given the constraints of the Tensor Engine, implementing matrix multiplication for arbitrary matrix dimensions on Trainium requires tiling the computation so it is performed as a sequence of matrix multiplications on fixed-size tiles. (This is similar to how vector addition in part 1 was tiled to work for large input vector sizes). The example below, which we modified from the [NKI tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/tutorials), demonstrates how to implement matrix multiplication using a tiled approach, where the tiles are sized to meet the Trainium Tensor engines tile size constraints. Note: a description of the code is provided after the code listing.

```
def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner"""

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

  # Maximum partition dimension of a tile
  TILE_K = nl.tile_size.pmax  # 128

  # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        res_psum += nisa.nc_matmul(lhsT_tile[...], rhs_tile[...])

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
```

Let's break down the components of the kernel which computes the matrix multiply: `result = lhsT x rhs`.

  - Input Tensors:
      - `lhsT` is the left-hand-side matrix. But the matrix is provided in a __transposed format__ with shape `[K,M]`, where both `K` and `M` are multiples of 128.
      - `rhs` is the right-hand-side matrix, with shape `[K,N]`, where `K` is a multiple of 128, and `N` is a multiple of 512.
      - `result` is the output matrix of shape `[M,N]`
      - In matrix multiplication, the **contraction dimension** refers to the column dimension of the left-hand matrix and the row dimension of the right-hand matrix. For example, say 
        we have the following matrix multiplication: `A x B = C`. The matrix `A` has shape  
        `[M, N]` and the matrix `B` has shape `[N, M]`. The shape of `C` is then `[M, M]`. 
        Thus, the dimensions that were eliminated was the column dimension of `A` and the row dimension of `B`.
      - Please note that in the `nki_matmul_tiled_` example above, the matrix is in transposed form, where `lhsT=A^T`. The `nisa.nc_matmul` takes `lhsT=A^T` and `rhs=B` as argument and returns `A x B`.
  - Tile Dimensions:
      - The tile sizes are set based on the constraints of the tensor engine matrix multiplication operation, as described [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.isa.nc_matmul.html).
        - `TILE_M`: 128 — Tile size for the `M` dimension.
        - `TILE_K`: 128 — Tile size for the `K` dimension.
        - `TILE_N`: 512 — Tile size for the `N` dimension.
  - Looping Over Tiles:
      - The kernel uses `affine_range` loops to iterate over tiles along the `M` and `N` dimensions of the `result` matrix.
      - For each output tile of shape `(TILE_M, TILE_N)`, it allocates a temporary partial sum tensor `res_psum` in PSUM memory.
  - Loading Tiles:
      - For each output tile, tiles of `lhsT` and `rhs` are loaded into the on-chip SBUF memory for efficient access.
      - `lhsT_tile` is loaded with a slice of shape `[TILE_K, TILE_M]`, and `rhs_tile` is loaded with a slice of shape `[TILE_K, TILE_N]`.
  - Matrix Multiplication:
      - A partial matrix multiplication is performed using the loaded tiles and partial results are accumulated into `res_psum`.
  - Storing Results:
      - Once the tiles for a given result block are fully computed, the partial sums in `res_psum` are copied to SBUF and cast to the required data type.
      - The final result is stored back into the `result` tensor at the corresponding position.

> Note that we've replaced `nl.matmul()` and `nl.load()/nl.store()` with `nisa.nc_matmul()` and `nisa.dma_copy()` from the online tutorial. This lowers the nki.lang APIs to nki.nisa. We recommend using nki.isa APIs for any compute instructions. This has more deterministic behaviour on how it is lowered, and less unexpected behavior that may cause bogus compilation errors. 

In summary, this tiled implementation handles large matrix dimensions by breaking them into hardware-compatible tile sizes. It leverages specialized memory buffers (i.e., PSUM) to minimize memory latency and optimize matrix multiplication performance. You can read more about NKI matrix multiplication [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/tutorials/matrix_multiplication.html).

### Convolution Layer Overview

Let’s now turn our focus to the convolution layer. Recall the [convolution operation](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dnninference/slide_26) discussed in class. It involves sliding a filter across an __input feature map__, where at each position the filter interacts with the overlapping input region. In each overlapping region, element-wise multiplications are performed between the filter weights and the input region region values. The results of these element-wise multiplications are then added together, producing a single value for the corresponding position in the output feature map. This process captures local spatial patterns and relationships among neighboring features.

<p align="center">
  <img src="/handout/convolution.png" width=55% height=55%>
</p>

The input feature map often consists of multiple channels. For example, an image usually contains three RGB channels (red, green, and blue). In this case, instead of only computing a weighted sum over the 2D spatial region, the convolution computes the weighted sum of both the 2D spatial region and channel depth. The image below depicts an example of a convolution layer being performed on a 32x32 input image with three RGB channels. In the image, a 5x5x3 filter is applied on the 32x32x3 image to produce a 28x28x1 output feature map.

<p align="center">
  <img src="/handout/cs231n_convolution.png" width=55% height=55%>
  <br>
  <em>Source: CS231N https://cs231n.stanford.edu/slides/2025/lecture_5.pdf </em>
</p>

__As seen in the image, each filter produces a single channel of output.__ To generate multiple output channels, multiple filters are applied to the input featuer map. In addition to this, each convolution filter also contains a scalar bias value that is to be added to each weighted sum. 

The input and output of the convolution operator can be summarized as follows (ignoring bias for now):

<p align="left">
  <img src="/handout/conv2d_summary.png" width=50% height=50%>
</p>

Additionally, a [convolution layer](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html) can take in additional hyper-parameters such as padding and stride in addition to an input feature map, filter weights, and scalar bias. However, we have *simplified the constraints of your convolution* to make implementation easier for you. You need **only to support a stride of 1**, and you do **not have to worry about padding** as we will pad the input feature map for you before it is passed into your kernel.

### Mapping Convolution to Matrix Multiplication

Now, our objective is to map the convolution operator onto the high-performance matrix operations supported by the Trainium's Tensor engine. To do this, we can compare the mathematical formulation of convolution with matrix multiplication.

**Conv2D:**

<p align="center">
  <img src="/handout/conv2d_formula.png" width=65% height=65%>
</p>

**Matrix Multiplication:**

<p align="center">
  <img src="/handout/matmul_formula.png" width=25% height=25%>
</p>

In class we discussed one way to convert convolution with many filters into a single large matrix multiplication.  We'll do the same thing here, but take a different approach that yields an efficient implementation on Trainium.  In this approach the convolution operation is formulated as a series of independent matrix multiplications. A visual illustration of this formulation is shown below.

<p align="center">
  <img src="/handout/conv2d_matmul_diagram.png" width=100% height=100%>
</p>

In this approach, the height and width dimensions of the input feature map are flattened into a single dimension, reshaping the input to `(Height × Width) × Input Channels​`. This reshaped input is then multiplied by each position of the filters, where `i` and `j` respectively range from `0` to `Filter Height - 1` and from `0` to `Filter Width - 1`. Each filter slice has a shape of `Input Channels × Output Channels`, and the resulting matrix multiplication contracts along the `Input Channels` dimension. To align the input with each filter slice, the input must be shifted by an offset corresponding to the filter’s current position `(i, j)`. The results of these matrix multiplications are accumulated to produce the output tensor.

Below is the pseudocode for the described algorithm:
```
- Have the input image with shape (Input Channels, Image Height * Image Width)
- Have the filter weights with shape (Filter Height, Filter Weight, Input Channels, Output Channels)
- Initialize the output to appropriate shape of (Output Channels, Output Height * Output Width)

# Iterate over the filter height
for i in range(Filter_Height):
    # Iterate over the filter width
    for j in range(Filter_Width):

        # Shift the Input tensor by (i, j) to align with the filter's current position
        input_shifted = shift(input, (i, j))

        # Perform matrix multiplication between the input and the filter slice
        # Note that this is a full matmul, without limit on input sizes
        output += matmul(transpose(weight[i,j,:,:]), input_shifted)
```

> [!NOTE]
> **This is just an algorithmic description, and the purpose of this assignment is for you to figure out to map this algorithmic description to an efficient implementation on this hardware!** Think about how the loops over filter size fit into the loops for computing your tiled matrix multiplication. Consider different ways of rearranging those loops while keeping data locality in mind. You’ll want to keep intermediate results in PSUM at all times until the computation for that tile is fully completed, so that each part of the result array in SBUF is written only once.

### Max Pool Layer Overview
Max pooling layers are commonly used in CNNs between successive convolutional layers to reduce the size of the feature maps. Not only does this prevent excessively large feature maps which can pose a problem for computational resources, but it also reduces the amount of parameters in the CNN which effectively reduces model overfitting.

A max pooling layer operates similarly to a convolution layer in that it slides a filter spatially over an input feature map. However, instead of computing a weighted sum for each overlapping region, the max pooling layer selects the maximum value from each region and stores it in the output feature map. This operation is applied independently to each channel of the feature map, thus the number of channels remains unchanged. For instance, consider a 4x4 input image with three RGB channels passing through a max pooling layer with a 2x2 filter. The resulting output is a 2x2 image with three RGB channels, showing that the spatial dimensions are reduced by a factor of 2 while the number of channels remains the same.

<p align="center">
  <img src="/handout/maxpool.png" width=37% height=37%>
</p>

As shown above, a [max pool layer](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html#torch.nn.functional.max_pool2d) typically has separate stride and filter size hyperparameters. Similar to the convolution layer, we have simplified the constraints for the max pooling layer you are required to implement. Instead of defining both parameters, your kernel will use a single parameter, `pool_size`, which corresponds to both the filter size and the stride. The `pool_size` can only be set to either 1 or 2. When `pool_size` is 2, the max pooling operation behaves as shown in the image above. When `pool_size` is 1, the max pooling layer functions as a no-op, producing an output identical to the input. While a `pool_size` of 1 might seem pointless, it actually offers added flexibility for your fused layer, as you will soon see. 

### Fusing Convolution and Max Pool
You will implement an NKI kernel that combines the Convolution and Max Pool layers into a single, fused operation. Below, we will outline the detailed specifications and requirements for your fused layer.

<p align="center">
  <img src="/handout/fused_kernel.png" width=95% height=95%>
</p>

The diagram above illustrates the calculations your fused kernel would perform on a 6x6 input with a single input channel. The fused kernel performs a standard convolution with one filter and stride of 1. The fused kernel then performs a max pool on the convolution result using a 2x2 pooling filter.

Your fused kernel takes in the following parameters:
  - `X` - A batch of input images. `X` has shape `(Batch Size, Input Channels, Input Height, Input Width)`. You are guaranteed that `Input Channels` will be a multiple of 128.
  - `W` - The convolution filter weights. `W` has shape `(Output Channels, Input Channels, Filter Height, Filter Width)`. You are guaranteed that `Filter Height == Filter Width`. You are also guaranteed that `Output Channels` is a multiple of 128. Moreover, you can assume that the size of the weights would always be such that it can completely fit inside SBUF.
  - `bias` - The convolution filter biases. `bias` has shape `(Output Channels)`
  - `pool_size` - The size of the max pooling filter and pooling stride. You are guaranteed that the size of the input, the size of the filter, and the `pool_size` would be such that everything is nicely divisible. More concretely, `(Input Height - Filter Height + 1) % Pool Size == 0`.  Notice that if the value of `pool_size` is `1`, then the fused kernel operates as a normal convolution kernel. This gives us the flexibility to choose whether we want max pooling or not.

Feel free to use the [course slides](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dnninference/slide_57) on a convolution layer implementation as a starting point. If you are referencing the course slides, `INPUT_DEPTH` is synonymous with `Input Channels` and `LAYER_NUM_FILTERS` is synonymous with `Output Channels` in our naming scheme. Note that the input parameters to your fused kernel have different shapes than depicted in the convolution course slides. You are free to reshape the inputs into whatever shapes you desire by using the [NumPy reshape method](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) just as was done in `vector_add_stream kernel` from Part 1. We have also given you the NumPy implementations of a convolution layer and a maxpool layer in `part2/conv2d_numpy.py`. The NumPy implementations should give you a general outline of the programming logic for each layer. It might be a good exercise to think about how you would be able to fuse the NumPy implementations into a single layer, which is what you will do in your kernel. Feel free to look over the [NKI tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html) to learn more about additional optimizations or other API functions. You can also view the [API Reference Manual](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/index.html) to see all of the API functions that are available and their usage. You may find some of them useful. *Hint:* [nisa.tensor_reduce(nl.max, ...)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_reduce.html) should be helpful for max pooling. [nisa.tensor_tensor](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor.html) should be helpful for adding bias.

### What You Need To Do
For this part of the assignment, focus exclusively on the file `part2/conv2d.py`. We've provided basic starter code; your task is to complete the implementation of the (fused) Conv2D kernel within the function `fused_conv2d_maxpool`.
#### General Tips
Prioritize Correctness First. Before optimizing for performance, ensure your implementation is correct. While you may choose to implement things differently, we recommend starting by creating a kernel that works for small image sizes. Then, once your kernel works for small images, extend its functionality to handle images that are too large to fit entirely in the SBUF buffer. Following that, incorporate bias addition. Proceed to optimize performance once you achieve correctness. The test harness is written such that you can optimize your conv2D implementation without having to fuse a maxpool operation with it. Thus, you may choose to fuse a max pool operation with the conv2d kernel after you have an implementation which is correct and meets the performance requirements.

#### Testing
Use the test harness script provided to validate your implementation. To run the tests, navigate to the `part2/` directory and execute:
```
python3 test_harness.py
```

To check the correctness and performance of your implementation of Conv2D kernel with a fused maxpool, invoke the test harness with the `--test_maxpool` flag. 

The test harness will run correctness tests first, and run performance checks next. A full-credit solution must achieve performance within 150% of the reference kernel while maintaining correctness. It will invoke your kernel with input tensors having data types float32 and float16: with the performance requirements for float16 being more strict. Make sure you write your kernels keeping this in mind!

#### Writeup and Profiling
Students are required to submit a write up briefly describing their implementations. Also describe how you went about optimizing your implementation. Make sure to profile your implementation, and report the achieved MFU (Model FLOPs Utilization), with both `float16` and `float32` data types. You can so by running `neuron-profile view -n [your_profile_name].neff -s [your_profile_name].ntff`. Run the test harness with the `--profile <profile_name>` flag to capture a trace.

> [!TIP]
> It’s expected that you will see the warnings below when using the profiler, which is a known issue due to changes in the NKI compiler stack. Some benchmarking parameters will be missing, but the MFU value is still available by mousing over the "Cumulative Utilization" line in the Estimated MFU section of the GUI.

<p align="center">
  <img src="handout/profiler-warning.png" alt="Profiler warning" width="90%">
</p>

### Tips on Using NKI
* Prefer to use nki.isa APIs in the following scenarios:
    * For all compute operations
      * nisa.nc_matmul instead of nl.matmul
      * nisa.tensor_scalar(op = nl.add, <>) instead of nl.add
    * Prefer to use nisa.dma_copy() instead of nl.load()/nl.store().
    * When invoking nisa compute operations, make sure to only pass op=nl.* codes as the arguments to these. For example, don’t pass op=math.sin.
* Avoid using nested functions. Define all functions at module level. 
* To debug your implementations, you could run the test harness with the `--simulate` flag. This wraps your implementation with a call to `nki.simulate_kernel()`: you can read more about it [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate_kernel.html#nki.simulate_kernel). When running in simulation mode, you can insert calls to `nl.device_print()` to print intermediate values of the device tensors. However, there __could be__ some divergence between CPU simulate and on-device execution. If you are unsure about the result, it's recommended to debug by returning intermediate tensors.
* Be careful when mutating tensor assignments. Some nisa APIs take the dst tensor as arguments, such as nisa.dma_copy(src=<>, dst=<>). Other APIs produce dst tensor through the function itself, and likely need to be used to modify an existing tensor. In future NKI releases all ISA APIs will take dst as an argument. For example:
  * x_sbuf = nl.zeros(shape=hbm_tensor.shape, buffer = nl.sbuf)  (create the array)
  * nisa.dma_copy(src = hbm_tensor, dst = x_sbuf) (copy into the array)
  * Specifically, if you choose to use `nl.load(...)`, `x = nl.load(...)` (which creates a new array) is different from `x[...] = nl.load(...)` (which modifieds an existing array).
* Avoid using block dimension, it's a pure software construct and does not impact hardware. (Don't worry about it if you don't know what it is.) Either put it in free dimensions or use lists of tensors. See public [documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_block_dimension_migration_guide.html#nki-block-dimension-migration-guide).
* For tensor indexing, prefer to do it with integer slicing. When more advanced indexing is required, use [`nl.mgrid`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.language.mgrid.html). Do not use nested slicing/mgrid. (e.g. t[0:128, 128:256][0:64, 0:64]). Do not use nl.arrange().


## Extra Credit
Run neuron-profile again on smaller images. Is there a difference in MFU between smaller and larger images? If so, how would you optimize your fused convolution layer for smaller images? Consider how the dimensions of each tile are mapped onto the chip.

Up to five points of extra credit will be awarded for solutions that meet the performance goal for smaller images (a stricter target). Your write-up must clearly explain your approach and the steps you took to optimize your solution.

It’s possible that your current solution already meets this requirement without the hint—great job if that’s the case! Please still make sure to thoroughly discuss your approach.

## Grading Guidelines

For the correctness test, we use two types of images. The first type is a small image with dimensions of 32×16. The second type is a large image with dimensions of 224×224, which exceeds the capacity of the SBUF and cannot fit within it.

For the performance test, we evaluate the performance under different configurations: with and without maxpool, and using float16 versus float32 precision. We will compare the performance of your program with the reference solution. You will pass the test if your p99 latency is within 150% of the reference latency.

**Write Up: 40 Points**
  - Part 1 Questions: 30 Points
  - Part 2 Questions: 10 Points

**Correctness of Fused Convolution - MaxPool Kernel: 10 Points**
  - With Small Images: 2.5 points
  - With Large Images: 2.5 points
  - With Bias Addition: 2.5 points
  - With Max Pool: 2.5 points

**Performance of Fused Convolution - MaxPool Kernel: 50 Points + 5 Points EC**
  - Without Max Pool (Float 16): 17.5 points
  - Without Max Pool (Float 32): 17.5 points
  - With Max Pool (Float 16): 7.5 points
  - With Max Pool (Float 32): 7.5 points
  - Without Max Pool on Smaller Images (Float 16): 1.25 points EC
  - Without Max Pool on Smaller Images (Float 32): 1.25 points EC
  - With Max Pool on Smaller Images (Float 16): 1.25 points EC
  - With Max Pool on Smaller Images (Float 32): 1.25 points EC

## Hand-in Instructions

Please submit your work using Gradescope. If you are working with a partner please remember to tag your partner on gradescope.

1. **Please submit your writeup as the file `writeup.pdf`.**
2. **Please submit `conv2d.py` from part 2.**
