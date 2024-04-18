const std = @import("std");
const testing = std.testing;

const Device = enum { cpu, cuda };
const Dtype = enum {
    f32,
    f16,
    const Self = @This();
    pub fn size(self: Dtype) usize {
        switch (self) {
            Dtype.f32 => {
                return 4;
            },
            Dtype.f16 => {
                return 2;
            },
        }
    }
};

const Rank0 = Rank(0);
const Rank1 = Rank(1);
const Rank2 = Rank(2);
const Rank3 = Rank(3);

fn Rank(comptime n: comptime_int) type {
    struct {
        shape: [n]u8,
    };
}

fn Tensor(comptime shape: anytype, comptime device: Device, comptime dtype: Dtype) type {
    return struct {
        shape: @TypeOf(shape),
        device: Device,
        dtype: Dtype,
        data: [10]u8,

        const Self = @This();
        pub fn zeros() Self {
            return Self{ .shape = shape, .device = device, .dtype = dtype, .data = undefined };
        }

        pub fn split(self: Self, comptime dim: usize, comptime into: usize) [into]Self {
            comptime std.debug.assert(shape[dim] % into == 0);
            var newshape = shape;
            newshape[dim] = newshape[dim] / into;

            var result: [into]Self = undefined;
            for (0..into) |index| {
                result[index] = Self{ .shape = newshape, .device = device, .dtype = dtype, .data = self.data };
            }
            return result;
        }

        pub fn rank(self: Self) usize {
            return self.shape.len;
        }

        pub fn total_size(self: Self) usize {
            var prod: usize = 1;
            for (self.shape) |dim| {
                prod *= dim;
            }
            return prod;
        }

        pub fn byte_size(self: Self) usize {
            var prod: usize = 1;
            for (self.shape) |dim| {
                prod *= dim;
            }
            return prod * dtype.size();
        }
    };
}

test "basic add functionality" {
    const tensor = Tensor([2]usize{ 2, 3 }, Device.cpu, Dtype.f32).zeros();
    try testing.expect(tensor.device == Device.cpu);
    try testing.expect(tensor.shape[0] == 2);
    try testing.expect(tensor.rank() == 2);
    try testing.expect(tensor.total_size() == 6);
    try testing.expect(tensor.byte_size() == 24);

    const tensor2 = Tensor([3]usize{ 2, 3, 4 }, Device.cpu, Dtype.f16).zeros();
    try testing.expect(tensor2.device == Device.cpu);
    try testing.expect(tensor2.shape[0] == 2);
    try testing.expect(tensor2.rank() == 3);
    try testing.expect(tensor2.total_size() == 24);
    try testing.expect(tensor2.byte_size() == 48);

    const splits = tensor2.split(0, 2);
    const left = splits[0];
    const right = splits[1];
    try testing.expect(left.rank() == 3);
    try testing.expect(left.shape[0] == 1);
    try testing.expect(right.shape[0] == 1);

    const tensor3 = Tensor(.{ 2, 3, 4 }, Device.cpu, Dtype.f16).zeros();
    try testing.expect(tensor3.rank() == 3);
    try testing.expect(tensor3.shape[0] == 2);
    try testing.expect(tensor3.shape[1] == 3);

    const newsplits = tensor2.split(1, 3);
    const l = newsplits[0];
    const m = newsplits[1];
    const r = newsplits[2];
    try testing.expect(l.shape[0] == 2);
    try testing.expect(l.shape[1] == 1);
    try testing.expect(m.shape[1] == 1);
    try testing.expect(r.shape[1] == 1);

    // comptime assert
    // const splits2 = tensor2.split(1, 2);
    // try testing.expect(splits2.len == 3);
}
