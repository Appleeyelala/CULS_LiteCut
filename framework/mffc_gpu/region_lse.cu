// Project-specific Region DAG LSE forward/backward; no third-party kernel code.
// One warp owns a ready region. All graph dependencies remain on the device.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdint.h>
#include <math.h>

using device_int = cuda::atomic_ref<int, cuda::thread_scope_device>;
using device_double = cuda::atomic_ref<double, cuda::thread_scope_device>;
constexpr unsigned ALL = 0xffffffffu;

__device__ void add_adjoint(double &location, double value) {
    // CUDA 11.8 atomic_ref<double> has generic CAS but no floating fetch_add.
    device_double slot(location);
    double previous = slot.load(cuda::memory_order_relaxed);
    while (!slot.compare_exchange_weak(previous, previous + value, cuda::memory_order_acq_rel)) {}
}

__device__ int take_job(int *queue, int *state, int regions) {
    int previous_done = device_int(state[2]).load(cuda::memory_order_acquire);
    unsigned long long last_progress = clock64();
    while (!device_int(state[3]).load(cuda::memory_order_acquire)) {
        int h = device_int(state[0]).load(cuda::memory_order_acquire);
        int t = device_int(state[1]).load(cuda::memory_order_acquire);
        if (h < t && h >= regions) {
            device_int(state[3]).store(2, cuda::memory_order_release);
            return -1;
        }
        if (h < t && device_int(state[0]).compare_exchange_weak(
                h, h + 1, cuda::memory_order_acq_rel)) {
            // A producer reserves tail before publishing this slot.
            int value;
            do { value = device_int(queue[h]).load(cuda::memory_order_acquire); }
            while (value < 0 && !device_int(state[3]).load(cuda::memory_order_acquire));
            return value;
        }
        int done = device_int(state[2]).load(cuda::memory_order_acquire);
        if (done == regions) return -1;
        if (done != previous_done) {
            previous_done = done;
            last_progress = clock64();
        } else if (clock64() - last_progress > 5000000000ULL) {
            // A malformed dependency graph must fail rather than hang a GPU.
            device_int(state[3]).store(1, cuda::memory_order_release);
            return -1;
        }
        __nanosleep(64);
    }
    return -1;
}

__device__ void publish(int node, int *queue, int *state, int regions) {
    int slot = device_int(state[1]).fetch_add(1, cuda::memory_order_acq_rel);
    if (slot >= regions) {
        device_int(state[3]).store(2, cuda::memory_order_release);
        return;
    }
    device_int(queue[slot]).store(node, cuda::memory_order_release);
}

__global__ void region_forward(int regions, int edges, int batch,
        const int64_t *in_ptr, const int64_t *in_edge, const int64_t *source,
        const int64_t *out_ptr, const int64_t *out_target,
        const double *cost, double *arrival, int *pending, int *queue,
        int *state, double temperature) {
    const int lane = threadIdx.x & 31;
    for (;;) {
        int r = lane == 0 ? take_job(queue, state, regions) : -1;
        r = __shfl_sync(ALL, r, 0);
        if (r < 0) return;
        for (int b = 0; b < batch; ++b) {
            double largest = -INFINITY;
            for (int64_t p = in_ptr[r] + lane; p < in_ptr[r + 1]; p += 32) {
                int64_t e = in_edge[p], s = source[e];
                double upstream = s < 0 ? 0.0 : device_double(arrival[(int64_t)b * regions + s]).load(cuda::memory_order_acquire);
                largest = fmax(largest, upstream + cost[(int64_t)b * edges + e]);
            }
            for (int offset = 16; offset; offset >>= 1)
                largest = fmax(largest, __shfl_down_sync(ALL, largest, offset));
            largest = __shfl_sync(ALL, largest, 0);
            double total = 0.0;
            for (int64_t p = in_ptr[r] + lane; p < in_ptr[r + 1]; p += 32) {
                int64_t e = in_edge[p], s = source[e];
                double upstream = s < 0 ? 0.0 : device_double(arrival[(int64_t)b * regions + s]).load(cuda::memory_order_acquire);
                total += exp((upstream + cost[(int64_t)b * edges + e] - largest) / temperature);
            }
            for (int offset = 16; offset; offset >>= 1)
                total += __shfl_down_sync(ALL, total, offset);
            if (lane == 0)
                device_double(arrival[(int64_t)b * regions + r]).store(largest + temperature * log(total), cuda::memory_order_release);
        }
        __syncwarp();
        for (int64_t p = out_ptr[r] + lane; p < out_ptr[r + 1]; p += 32) {
            int target = out_target[p];
            if (device_int(pending[target]).fetch_sub(1, cuda::memory_order_acq_rel) == 1)
                publish(target, queue, state, regions);
        }
        __syncwarp();
        if (lane == 0) device_int(state[2]).fetch_add(1, cuda::memory_order_release);
    }
}

__global__ void region_backward(int regions, int edges, int batch,
        const int64_t *in_ptr, const int64_t *in_edge, const int64_t *source,
        const double *cost, double *arrival, double *adjoint, double *gradient,
        int *pending, int *queue, int *state, double temperature) {
    const int lane = threadIdx.x & 31;
    for (;;) {
        int r = lane == 0 ? take_job(queue, state, regions) : -1;
        r = __shfl_sync(ALL, r, 0);
        if (r < 0) return;
        for (int64_t p = in_ptr[r] + lane; p < in_ptr[r + 1]; p += 32) {
            int64_t e = in_edge[p], s = source[e];
            for (int b = 0; b < batch; ++b) {
                double upstream = s < 0 ? 0.0 : device_double(arrival[(int64_t)b * regions + s]).load(cuda::memory_order_acquire);
                double current = device_double(arrival[(int64_t)b * regions + r]).load(cuda::memory_order_acquire);
                double signal = device_double(adjoint[(int64_t)b * regions + r]).load(cuda::memory_order_acquire);
                double value = signal * exp((upstream + cost[(int64_t)b * edges + e] - current) / temperature);
                gradient[(int64_t)b * edges + e] = value;
                if (s >= 0)
                    add_adjoint(adjoint[(int64_t)b * regions + s], value);
            }
            if (s >= 0 && device_int(pending[s]).fetch_sub(1, cuda::memory_order_acq_rel) == 1)
                publish((int)s, queue, state, regions);
        }
        __syncwarp();
        if (lane == 0) device_int(state[2]).fetch_add(1, cuda::memory_order_release);
    }
}

extern "C" int mffc_region_lse(int reverse, int regions, int edges, int batch,
        const int64_t *in_ptr, const int64_t *in_edge, const int64_t *source,
        const int64_t *out_ptr, const int64_t *out_target,
        const double *cost, double *arrival, double *adjoint, double *gradient,
        int *pending, int *queue, int *state, double temperature,
        int blocks, void *stream_pointer) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    if (regions <= 0 || edges < 0 || batch <= 0 || temperature <= 0 || blocks <= 0)
        return (int)cudaErrorInvalidValue;
    if (reverse)
        region_backward<<<blocks, 128, 0, stream>>>(regions, edges, batch, in_ptr,
            in_edge, source, cost, arrival, adjoint, gradient, pending, queue, state, temperature);
    else
        region_forward<<<blocks, 128, 0, stream>>>(regions, edges, batch, in_ptr,
            in_edge, source, out_ptr, out_target, cost, arrival, pending, queue, state, temperature);
    return (int)cudaGetLastError();
}

extern "C" const char *mffc_region_error(int error) {
    return cudaGetErrorString((cudaError_t)error);
}
