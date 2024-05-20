// const z = @import("lib.zig");
const cuda_allocator = @import("cuda_allocator.zig");
const std = @import("std");
pub const cu = @cImport({
    @cInclude("cuda.h");
});
pub const gemm = @cImport({
    @cInclude("gemm.h");
});
pub const err = @import("error.zig");
pub fn SizedPtr(comptime T: type, comptime n: usize) type {
    return struct {
        devptr: cu.CUdeviceptr,
        pub const Ptr = cu.CUdeviceptr;

        const Self = @This();
        fn ptr(self: Self) Ptr {
            return self.devptr;
        }

        fn byte_size(self: Self) usize {
            _ = self;
            return n * @sizeOf(T);
        }
    };
}

pub fn CpuSizedPtr(comptime T: type, comptime n: usize) type {
    return struct {
        dataptr: []T,
        pub const Ptr = []T;
        const Self = @This();
        fn ptr(self: Self) Ptr {
            return self.dataptr;
        }
        fn byte_size(self: Self) usize {
            _ = self;
            return n * @sizeOf(T);
        }
    };
}

pub const DeviceEnum = enum { cpu, cuda };
pub const Device = union(DeviceEnum) {
    cpu,
    cuda: usize,
    const Self = @This();
    pub fn device_type(comptime self: Self) type {
        switch (self) {
            .cpu => return Cpu,
            .cuda => return Cuda(self.cuda),
        }
    }
    pub fn data_type(comptime self: Self, comptime T: type, comptime n: usize) type {
        switch (self) {
            .cpu => return CpuSizedPtr(T, n),
            .cuda => return SizedPtr(T, n),
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

        const device_enum = Device{ .cuda = device_id };
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

        pub const Allocator = struct {
            device: Self,

            fn alloc(self: Allocator, comptime T: type, comptime n: usize) !SizedPtr(T, n) {
                var ptr: cu.CUdeviceptr = undefined;
                const bytes = n * @sizeOf(T);
                try err.Cuda(cu.cuMemAllocAsync(&ptr, bytes, self.device.stream));
                return SizedPtr(T, n){ .devptr = ptr };
            }

            fn zero_alloc(self: Allocator, comptime T: type, comptime n: usize) !SizedPtr(T, n) {
                const ptr = try self.alloc(T, n);
                try err.Cuda(cu.cuMemsetD8Async(ptr.ptr, 0, ptr.len(), self.device.stream));
                return ptr;
            }

            pub fn empty(self: Allocator, comptime T: type, comptime rank: usize, comptime shape: [rank]usize) !Tensor(device_enum, T, rank, shape) {
                const nelements = comptime total_size(&shape);
                const data = try self.alloc(T, nelements);
                return Tensor(device_enum, T, rank, shape).init(data, self.device);
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
            gemm.launch_simple_gemm_tt_half(m, n, k, @ptrCast(@alignCast(&alpha)), a_data.devptr, lda, b_data.devptr, ldb, @ptrCast(@alignCast(&beta)), out_data.devptr, ldc, @ptrCast(self.stream));
        }
    };
}

pub const Cpu = struct {
    const device_enum = Device{ .cpu = undefined };
    const Self = @This();
    const Allocator = struct {
        alloc: std.mem.Allocator,
        device: Cpu,
        pub fn empty(self: Allocator, comptime T: type, comptime rank: usize, comptime shape: [rank]usize) !Tensor(device_enum, T, rank, shape) {
            const n = comptime total_size(&shape);
            const slice = try self.alloc.alloc(T, n);
            // @memset(slice, 0);
            const data = CpuSizedPtr(T, n){ .dataptr = slice };
            return Tensor(device_enum, T, rank, shape).init(data, self.device);
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
                try err.Cuda(cu.cuMemcpyDtoHAsync_v2(@ptrCast(ptr), other.ptr(), other.byte_size(), other.device.stream));
            },
            .cpu => {
                @panic("Not implemented yet");
            },
        }
    }
};

fn total_size(shape: []const usize) usize {
    var total = 1;
    for (shape) |dim| {
        total *= dim;
    }
    return total;
}

pub fn Tensor(comptime device: Device, comptime T: type, comptime rank: usize, comptime shape: [rank]usize) type {
    const nelements = total_size(&shape);
    const DataType = device.data_type(T, nelements);
    const DeviceType = device.device_type();
    return struct {
        device: DeviceType,
        data: DataType,
        const Self = @This();

        fn init(data: device.data_type(T, nelements), realdevice: anytype) !Self {
            return Self{ .data = data, .device = realdevice };
        }

        fn ptr(self: Self) DataType.Ptr {
            return self.data.ptr();
        }

        fn byte_size(self: Self) usize {
            return self.data.byte_size();
        }

        // fn _copy_from_device(self: Self, other: anytype) !void {
        //     const device_enum = @typeName(@typeInfo(@typeInfo(@TypeOf(other)).Pointer.child).Struct.fields[0].type);
        //     return self.copy_from_device(device_enum, other);
        // }

        fn copy_from_device(self: Self, comptime otherdevice: Device, other: *const Tensor(otherdevice, T, rank, shape)) !void {
            return self.device.copy_from(self.data.ptr(), otherdevice, other);
        }

        fn matmul_t(self: Self, comptime outdim: usize, other: Tensor(device, T, 2, [2]usize{ outdim, shape[0] }), out: Tensor(device, T, 2, [2]usize{ shape[0], outdim }), realdevice: anytype) !void {
            std.debug.assert(rank == 2);
            const m = shape[0];
            const k = shape[shape.len - 1];
            const n = outdim;
            const lda = (k + 16 - 1) / 16;
            const ldb = (n + 16 - 1) / 16;
            const ldc = (n + 16 - 1) / 16;
            _ = ldc;
            const alpha: T = 1.0;
            const beta: T = 0.0;
            realdevice.matmul_t(T, m, n, k, alpha, self.data, lda, other.data, ldb, beta, out.data, ldb);
        }
    };
}

pub fn main() !void {
    const dtype = f16;
    const rank = 2;
    const M = 8;
    const N = 8;
    const K = 8;

    const cuda_0 = try Cuda(0).create();
    const gpu_allocator = cuda_0.allocator();
    const cpu = Cpu{};
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        //fail test; can't try in defer as defer is executed after we return
        if (deinit_status == .leak) @panic("Leak detected");
    }
    const cpu_allocator = cpu.allocator(gpa.allocator());

    const cpu_A = try cpu_allocator.empty(f16, rank, [rank]usize{ M, K });
    defer cpu_allocator.free(cpu_A);
    for (cpu_A.data.dataptr, 0..) |*ap, i| {
        ap.* = @as(dtype, @floatFromInt(i));
    }

    const A = try gpu_allocator.empty(f16, rank, [rank]usize{ M, K });
    defer gpu_allocator.free(A);
    try A.copy_from_device(@TypeOf(cpu).device_enum, &cpu_A);
    const cpu_B = try cpu_allocator.empty(f16, rank, [rank]usize{ N, K });
    defer cpu_allocator.free(cpu_B);
    for (cpu_B.data.dataptr, 0..) |*bp, i| {
        const j = i % K;
        const ii = i / K;
        const i_transposed = j * N + ii;
        bp.* = @as(dtype, @floatFromInt(i_transposed));
    }

    const B = try gpu_allocator.empty(f16, rank, [rank]usize{ M, K });
    defer gpu_allocator.free(B);
    try B.copy_from_device(@TypeOf(cpu).device_enum, &cpu_B);

    const C = try gpu_allocator.empty(f16, 2, [2]usize{ M, N });
    defer gpu_allocator.free(C);

    try cuda_0.synchronize();

    try A.matmul_t(N, B, C, cuda_0);

    const cpu_C = try cpu_allocator.empty(f16, rank, [rank]usize{ M, N });
    defer cpu_allocator.free(cpu_C);
    try cpu_C.copy_from_device(@TypeOf(cuda_0).device_enum, &C);

    std.debug.print("{any}\n", .{cpu_A.data.dataptr});
    std.debug.print("{any}\n", .{cpu_B.data.dataptr});
    std.debug.print("{any}\n", .{cpu_C.data.dataptr});
}
