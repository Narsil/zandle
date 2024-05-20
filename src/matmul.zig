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
        ptr: cu.CUdeviceptr,

        fn len(self: @This()) usize {
            _ = self;
            return n * @sizeOf(T);
        }
    };
}

pub const DeviceEnum = enum { cpu, cuda };
pub const Device = union(DeviceEnum) {
    cpu,
    cuda: usize,

    pub fn data_type(comptime self: @This(), comptime T: type, comptime n: usize) type {
        switch (self) {
            .cpu => return [n]T,
            .cuda => return SizedPtr(T, n),
        }
    }
};

pub fn Cuda(comptime device_id: c_int) type {
    const device_enum = Device{ .cuda = device_id };
    return struct {
        device_id: c_int,
        device: cu.CUdevice,
        context: cu.CUcontext,
        stream: cu.CUstream,

        const Self = @This();
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

        pub fn synchronize(self: Self) !void {
            _ = self;
            try err.Cuda(cu.cuCtxSynchronize());
        }

        pub fn zeros(self: Self, comptime T: type, comptime rank: usize, comptime shape: [rank]usize) !Tensor(device_enum, T, rank, shape) {
            const nelements = comptime total_size(&shape);
            const data = try self.static_zero_alloc(T, nelements);
            return Tensor(device_enum, T, rank, shape).init(data);
        }

        pub fn matmul_t(self: Self, comptime T: type, m: usize, n: usize, k: usize, alpha: T, a_data: anytype, lda: usize, b_data: anytype, ldb: usize, beta: T, out_data: anytype, ldc: usize) void {
            gemm.launch_simple_gemm_tt_half(m, n, k, @ptrCast(@alignCast(&alpha)), a_data.ptr, lda, b_data.ptr, ldb, @ptrCast(@alignCast(&beta)), out_data.ptr, ldc, @ptrCast(self.stream));
        }

        // pub fn alloc(self: Self, comptime T: type, n: usize) !SizedPtr(T) {
        //     var ptr: cu.CUdeviceptr = undefined;
        //     const bytes = n * @sizeOf(T);
        //     std.debug.print("Allocate {} elements {} bytes\n", .{ n, bytes });
        //     try err.Cuda(cu.cuMemAllocAsync(&ptr, bytes, self.stream));
        //     std.debug.print("Allocated {}\n", .{ptr});
        //     return SizedPtr(T){ .n = n, .ptr = ptr };
        // }

        // pub fn zero_alloc(self: Self, comptime T: type, n: usize) !SizedPtr(T) {
        //     const ptr = try self.alloc(T, n);
        //     try err.Cuda(cu.cuMemsetD8Async(ptr.ptr, 0, n * @sizeOf(T), self.stream));
        //     return ptr;
        // }

        pub fn static_alloc(self: Self, comptime T: type, comptime n: usize) !SizedPtr(T, n) {
            var ptr: cu.CUdeviceptr = undefined;
            const bytes = n * @sizeOf(T);
            std.debug.print("Allocate {} elements {} bytes\n", .{ n, bytes });
            try err.Cuda(cu.cuMemAllocAsync(&ptr, bytes, self.stream));
            std.debug.print("Allocated {}\n", .{ptr});
            return SizedPtr(T, n){ .ptr = ptr };
        }

        pub fn static_zero_alloc(self: Self, comptime T: type, comptime n: usize) !SizedPtr(T, n) {
            const ptr = try self.static_alloc(T, n);
            try err.Cuda(cu.cuMemsetD8Async(ptr.ptr, 0, ptr.len(), self.stream));
            return ptr;
        }

        pub fn free(self: Self, ptr: anytype) void {
            const e = cu.cuMemFreeAsync(ptr.ptr, self.stream);
            _ = e;
        }
    };
}

pub const Cpu = struct {
    pub fn zeros(alloc: std.mem.Allocator, comptime T: type, n: usize) ![]T {
        const newptr = try alloc.alloc(T, n);
        @memset(newptr, 0);
        return newptr;
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
    return struct {
        data: device.data_type(T, nelements),
        const Self = @This();

        fn init(data: device.data_type(T, nelements)) !Self {
            return Self{ .data = data };
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

fn static_copy_htod(comptime T: type, comptime n: usize, cpu: []T, gpu: SizedPtr(T, n), stream: cu.CUstream) !void {
    std.debug.print("HtoD {} {}\n", .{ &cpu, gpu.ptr });
    try err.Cuda(cu.cuMemcpyHtoDAsync_v2(gpu.ptr, @ptrCast(cpu), @sizeOf(T) * cpu.len, stream));
}

fn static_copy_dtoh(comptime T: type, comptime n: usize, gpu: SizedPtr(T, n), stream: cu.CUstream, cpu: []T) !void {
    std.debug.print("DtoH {} {}\n", .{ &cpu, gpu.ptr });
    try err.Cuda(cu.cuMemcpyDtoHAsync_v2(@ptrCast(cpu), gpu.ptr, @sizeOf(T) * cpu.len, stream));
}

pub fn main() !void {
    const cuda_0 = try Cuda(0).create();
    const cpu = Cpu;
    const dtype = f16;
    const M = 8;
    const N = 8;
    const K = 8;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        //fail test; can't try in defer as defer is executed after we return
        if (deinit_status == .leak) @panic("Leak detected");
    }
    const cpu_allocator = gpa.allocator();
    const cpu_A = try cpu.zeros(cpu_allocator, dtype, M * K);
    defer cpu_allocator.free(cpu_A);
    for (cpu_A, 0..) |*ap, i| {
        ap.* = @as(dtype, @floatFromInt(i));
    }

    const A = try cuda_0.zeros(f16, 2, [2]usize{ M, K });
    defer cuda_0.free(A.data);

    try static_copy_htod(dtype, M * K, cpu_A, A.data, cuda_0.stream);
    try cuda_0.synchronize();

    const cpu_B = try cpu.zeros(cpu_allocator, dtype, N * K);
    defer cpu_allocator.free(cpu_B);
    for (cpu_B, 0..) |*bp, i| {
        const j = i % K;
        const ii = i / K;
        const i_transposed = j * N + ii;
        bp.* = @as(dtype, @floatFromInt(i_transposed));
    }

    const cpu_C = try cpu.zeros(cpu_allocator, dtype, M * N);
    defer cpu_allocator.free(cpu_C);

    const B = try cuda_0.zeros(f16, 2, [2]usize{ M, K });
    defer cuda_0.free(B.data);
    try static_copy_htod(dtype, N * K, cpu_B, B.data, cuda_0.stream);

    const C = try cuda_0.zeros(f16, 2, [2]usize{ M, N });
    defer cuda_0.free(C.data);

    try cuda_0.synchronize();

    try A.matmul_t(N, B, C, cuda_0);

    try static_copy_dtoh(dtype, M * N, C.data, cuda_0.stream, cpu_C);

    std.debug.print("{}, {any}\n", .{ &cpu_A, cpu_A });
    std.debug.print("{any}\n", .{cpu_B});
    std.debug.print("{any}\n", .{cpu_C});
}
