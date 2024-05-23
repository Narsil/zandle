const std = @import("std");
const z = @import("lib.zig");

pub fn main() !void {
    const dtype = f16;
    const rank = 2;
    const M = z.Dim{ .name = 'M', .value = 8 };
    const N = z.Dim{ .name = 'N', .value = 8 };
    const K = z.Dim{ .name = 'K', .value = 8 };

    const cuda_0 = try z.Cuda(0).create();
    const gpu_allocator = cuda_0.allocator();
    const cpu = z.Cpu{};
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        //fail test; can't try in defer as defer is executed after we return
        if (deinit_status == .leak) @panic("Leak detected");
    }
    const cpu_allocator = cpu.allocator(gpa.allocator());

    const cpu_A = try cpu_allocator.empty(f16, rank, [rank]z.Dim{ M, K });
    defer cpu_allocator.free(cpu_A);
    for (cpu_A.data.dataptr, 0..) |*ap, i| {
        ap.* = @as(dtype, @floatFromInt(i));
    }

    const A = try gpu_allocator.empty(f16, rank, [rank]z.Dim{ M, K });
    defer gpu_allocator.free(A);
    try A.copy_from_device(@TypeOf(cpu).device_enum, &cpu_A);
    const cpu_B = try cpu_allocator.empty(f16, rank, [rank]z.Dim{ N, K });
    defer cpu_allocator.free(cpu_B);
    for (cpu_B.data.dataptr, 0..) |*bp, i| {
        const j = i % K.value;
        const ii = i / K.value;
        const i_transposed = j * N.value + ii;
        bp.* = @as(dtype, @floatFromInt(i_transposed));
    }

    const B = try gpu_allocator.empty(f16, rank, [rank]z.Dim{ N, K });
    defer gpu_allocator.free(B);
    try B.copy_from_device(@TypeOf(cpu).device_enum, &cpu_B);

    const C = try gpu_allocator.empty(f16, 2, [2]z.Dim{ M, N });
    defer gpu_allocator.free(C);

    try cuda_0.synchronize();

    try A.matmul_t(N, B, C, cuda_0);

    const cpu_C = try cpu_allocator.empty(f16, rank, [rank]z.Dim{ M, N });
    defer cpu_allocator.free(cpu_C);
    try cpu_C.copy_from_device(@TypeOf(cuda_0).device_enum, &C);

    std.debug.print("{any}\n", .{cpu_A.data.dataptr});
    std.debug.print("{any}\n", .{cpu_B.data.dataptr});
    std.debug.print("{any}\n", .{cpu_C.data.dataptr});
}
