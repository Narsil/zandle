const std = @import("std");
pub const cu = @cImport({
    @cInclude("cuda.h");
});
pub const err = @import("error.zig");

pub fn SizedPtr(comptime T: type) type {
    _ = T;
    return struct {
        ptr: cu.CUdeviceptr,
        n: usize,
    };
}
