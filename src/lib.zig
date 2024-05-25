const dev = @import("device.zig");
const tensor = @import("tensor.zig");

pub const Dim = tensor.Dim;
pub const Cuda = dev.Cuda;
pub const Cpu = dev.Cpu;
