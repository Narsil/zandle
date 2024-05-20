const std = @import("std");
const z = @import("lib.zig");

pub fn main() !void {
    // const dim1 = z.Dim{ .static = 2 };
    // const dim2 = z.Dim{ .static = 3 };
    // const seqlen = z.Dim{ .dyn = 's' };
    // // const a = Tensor([2]Dim{ dim1, dim2 }, Device.cpu, Dtype.f32).zeros();
    // const a = z.Tensor(2, [2]z.Dim{ dim1, dim2 }, z.Device.cpu, z.Dtype.f32).zeros();
    // std.debug.print("{}\n", .{a});
    // const input_ids = z.Tensor(1, [1]z.Dim{seqlen}, z.Device.cpu, z.Dtype.f32).zeros();
    // std.debug.print("{}\n", .{input_ids});

    // var b = z.Tensor(2, [2]z.Dim{ dim1, dim2 }, z.Device.cpu, z.Dtype.f32).zeros();
    // b.add(a);
    // // const dim3 = Dim{ .static = 4 };
    // // var c = Tensor(2, [2]Dim{ dim1, dim3 }, Device.cpu, Dtype.f32).zeros();
    // // c.add(a);
}
