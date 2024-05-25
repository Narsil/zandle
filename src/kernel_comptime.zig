const std = @import("std");
const config = @import("config");
pub const IS_COMPILING = if (@hasDecl(config, "kernel_compilation")) config.kernel_compilation else false;
var KERNELS: [20_000]u8 = undefined;
var ptr: usize = 0;

pub fn start() void {
    _emit("#include \"gemm.cu\"");
    _emit("extern \"C\" {");
}

fn _emit(comptime log: []const u8) void {
    std.mem.copyForwards(u8, KERNELS[ptr..], log);
    KERNELS[ptr + log.len] = '\n';
    KERNELS[ptr + log.len + 1] = 0;
    ptr += log.len + 1;
}

pub fn emit(comptime log: []const u8) void {
    if (!std.mem.containsAtLeast(u8, &KERNELS, 1, log)) {
        _emit(log);
    }
}

pub fn finish() !void {
    _emit("}");
    _ = try std.io.getStdOut().write(KERNELS[0..ptr]);
}
