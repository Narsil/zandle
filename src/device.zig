const std = @import("std");
pub const cu = @cImport({
    @cInclude("cuda.h");
});
const err = @import("error.zig");
const tensor = @import("tensor.zig");
const Dim = tensor.Dim;
const Tensor = tensor.Tensor;
const gemm = @cImport({
    @cInclude("gemm.h");
});
const kernel_comptime = @import("kernel_comptime.zig");

fn rtotal_size(shape: []const Dim) usize {
    var total: usize = 1;
    for (shape) |dim| {
        total *= dim.value.get();
    }
    return total;
}

pub fn total_size(shape: []const Dim) ?usize {
    var total: usize = 1;
    for (shape) |dim| {
        switch (dim.value) {
            .static => |v| {
                total *= v;
            },
            .dynamic => return null,
        }
    }
    return total;
}

pub fn RuntimePtr(comptime T: type) type {
    return struct {
        devptr: cu.CUdeviceptr,
        n: usize,
        pub const Ptr = cu.CUdeviceptr;

        const Self = @This();
        pub fn ptr(self: Self) Ptr {
            return self.devptr;
        }

        pub fn byte_size(self: Self) usize {
            return self.n * @sizeOf(T);
        }
    };
}
pub fn SizedPtr(comptime T: type, comptime n: usize) type {
    return struct {
        devptr: cu.CUdeviceptr,
        pub const Ptr = cu.CUdeviceptr;

        const Self = @This();
        pub fn ptr(self: Self) Ptr {
            return self.devptr;
        }

        pub fn byte_size(self: Self) usize {
            _ = self;
            return n * @sizeOf(T);
        }
    };
}

pub fn CpuRuntimePtr(comptime T: type) type {
    return struct {
        dataptr: []T,
        pub const Ptr = []T;
        const Self = @This();
        pub fn ptr(self: Self) Ptr {
            return self.dataptr;
        }

        pub fn byte_size(self: Self) usize {
            return self.dataptr.len * @sizeOf(T);
        }
    };
}
pub fn CpuSizedPtr(comptime T: type, comptime n: usize) type {
    return struct {
        dataptr: []T,
        pub const Ptr = []T;
        const Self = @This();
        pub fn ptr(self: Self) Ptr {
            return self.dataptr;
        }

        pub fn byte_size(self: Self) usize {
            _ = self;
            return n * @sizeOf(T);
        }
    };
}

pub const Device = union(enum) {
    cpu,
    cuda: usize,
    const Self = @This();
    pub fn device_type(comptime self: Self) type {
        switch (self) {
            .cpu => return Cpu,
            .cuda => return Cuda(self.cuda),
        }
    }
    pub fn data_type(comptime self: Self, comptime T: type, comptime cn: ?usize) type {
        if (cn) |n| {
            switch (self) {
                .cpu => return CpuSizedPtr(T, n),
                .cuda => return SizedPtr(T, n),
            }
        } else {
            switch (self) {
                .cpu => return CpuRuntimePtr(T),
                .cuda => return RuntimePtr(T),
            }
        }
    }
};

pub fn Cuda(comptime device_id: c_int) type {
    return struct {
        device_id: c_int,
        device: cu.CUdevice,
        context: cu.CUcontext,
        stream: cu.CUstream,
        const Self = @This();

        pub const device_enum = Device{ .cuda = device_id };
        pub fn create() !Self {
            var device: cu.CUdevice = undefined;
            var context: cu.CUcontext = undefined;
            var stream: cu.CUstream = undefined;

            try err.Cuda(cu.cuInit(0));
            try err.Cuda(cu.cuDeviceGet(&device, device_id));
            try err.Cuda(cu.cuDevicePrimaryCtxRetain(&context, device));
            try err.Cuda(cu.cuCtxSetCurrent(context));
            try err.Cuda(cu.cuStreamCreate(&stream, cu.CU_STREAM_DEFAULT));
            return @This(){
                .device_id = device_id,
                .device = device,
                .context = context,
                .stream = stream,
            };
        }

        pub fn copy_from(self: Self, ptr: cu.CUdeviceptr, otherdevice: Device, other: anytype) !void {
            switch (otherdevice) {
                .cpu => {
                    try err.Cuda(cu.cuMemcpyHtoDAsync_v2(ptr, @ptrCast(other.ptr()), other.byte_size(), self.stream));
                },
                .cuda => {
                    @panic("Not implemented yet");
                },
            }
        }

        pub fn copy_to_cpu(self: Self, ptr: anytype, devptr: cu.CUdeviceptr, byte_size: usize) !void {
            try err.Cuda(cu.cuMemcpyDtoHAsync_v2(@ptrCast(ptr), devptr, byte_size, self.stream));
        }

        pub const Allocator = struct {
            device: Self,

            fn dynalloc(self: Allocator, comptime T: type, n: usize) !RuntimePtr(T) {
                var ptr: cu.CUdeviceptr = undefined;
                const bytes = n * @sizeOf(T);
                try err.Cuda(cu.cuMemAllocAsync(&ptr, bytes, self.device.stream));
                return RuntimePtr(T){ .n = n, .devptr = ptr };
            }

            fn alloc(self: Allocator, comptime T: type, comptime n: usize) !SizedPtr(T, n) {
                var ptr: cu.CUdeviceptr = undefined;
                const bytes = n * @sizeOf(T);
                try err.Cuda(cu.cuMemAllocAsync(&ptr, bytes, self.device.stream));
                return SizedPtr(T, n){ .devptr = ptr };
            }

            // fn zero_alloc(self: Allocator, comptime T: type, comptime n: usize) !SizedPtr(T, n) {
            //     const ptr = try self.alloc(T, n);
            //     try err.Cuda(cu.cuMemsetD8Async(ptr.ptr, 0, ptr.len(), self.device.stream));
            //     return ptr;
            // }

            pub fn empty(self: Allocator, comptime T: type, comptime rank: usize, comptime shape: [rank]Dim) !Tensor(device_enum, T, rank, shape) {
                const comptime_nelements = comptime total_size(&shape);
                if (comptime_nelements) |n| {
                    const data = try self.alloc(T, n);
                    return Tensor(device_enum, T, rank, shape).init(data, self.device);
                } else {
                    const nelements = rtotal_size(&shape);
                    const data = try self.dynalloc(T, nelements);
                    return Tensor(device_enum, T, rank, shape).init(data, self.device);
                }
            }

            pub fn free(self: Allocator, ptr: anytype) void {
                const e = cu.cuMemFreeAsync(ptr.data.devptr, self.device.stream);
                _ = e;
            }
        };

        pub fn allocator(self: Self) Allocator {
            return Allocator{ .device = self };
        }

        pub fn synchronize(self: Self) !void {
            _ = self;
            try err.Cuda(cu.cuCtxSynchronize());
        }

        pub fn matmul_t(self: Self, comptime T: type, m: usize, n: usize, k: usize, alpha: T, a_data: anytype, lda: usize, b_data: anytype, ldb: usize, beta: T, out_data: anytype, ldc: usize) void {
            if (kernel_comptime.IS_COMPILING) {
                kernel_comptime.emit(
                    \\void launch_simple_gemm_tt_half(size_t m, size_t n, size_t k,
                    \\                                            __half const* alpha,
                    \\                                            __half const* A, size_t lda,
                    \\                                            __half const* B, size_t ldb,
                    \\                                            __half const* beta, __half* C,
                    \\                                            size_t ldc, cudaStream_t stream){
                    \\    launch_simple_gemm_tt<__half>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
                    \\}
                );
            } else {
                gemm.launch_simple_gemm_tt_half(m, n, k, @ptrCast(@alignCast(&alpha)), a_data.devptr, lda, b_data.devptr, ldb, @ptrCast(@alignCast(&beta)), out_data.devptr, ldc, @ptrCast(self.stream));
            }
        }
    };
}

pub const Cpu = struct {
    pub const device_enum = Device{ .cpu = undefined };
    const Self = @This();
    const Allocator = struct {
        alloc: std.mem.Allocator,
        device: Cpu,
        pub fn empty(self: Allocator, comptime T: type, comptime rank: usize, comptime shape: [rank]Dim) !Tensor(device_enum, T, rank, shape) {
            const cn = comptime total_size(&shape);
            if (cn) |n| {
                const slice = try self.alloc.alloc(T, n);
                const DataType = CpuSizedPtr(T, n);
                const data = DataType{ .dataptr = slice };
                return Tensor(device_enum, T, rank, shape).init(data, self.device);
            } else {
                const n = rtotal_size(&shape);
                const slice = try self.alloc.alloc(T, n);
                const data = CpuRuntimePtr(T){ .dataptr = slice };

                return Tensor(device_enum, T, rank, shape).init(data, self.device);
            }
        }

        pub fn free(self: Allocator, data: anytype) void {
            self.alloc.free(data.data.dataptr);
        }
    };

    pub fn allocator(self: Self, alloc: std.mem.Allocator) Allocator {
        _ = self;
        return Allocator{ .alloc = alloc, .device = Cpu{} };
    }

    pub fn copy_from(self: Self, ptr: anytype, otherdevice: Device, other: anytype) !void {
        _ = self;
        switch (otherdevice) {
            .cuda => {
                try other.device.copy_to_cpu(ptr, other.ptr(), other.byte_size());
            },
            .cpu => {
                @panic("Not implemented yet");
            },
        }
    }
};
