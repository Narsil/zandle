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

const Dim = union(enum) {
    dyn: u8, // Char
    static: usize,

    pub fn value(self: Dim) usize {
        const val = switch (self) {
            .dyn => 0,
            .static => self.static,
        };
        return val;
    }
};

const SEQ_LEN = Dim("S");
const HIDDEN_SIZE = Dim(4096);

fn Rank(comptime n: comptime_int) type {
    const s = [n]Dim;
    return s;
}

const RealTensor = struct {
    data: []const u8,

    const Self = @This();
    pub fn add(_: Self, _: Self) void {
        // std.debug.assert(self.shape.len == other.shape.len);
        // for (self.shape, other.shape) |d1, d2| {
        //     std.debug.assert(d1 == d2);
        // }
    }
};

fn Tensor(comptime rank: usize, comptime shape: Rank(rank), comptime device: Device, comptime dtype: Dtype) type {
    return struct {
        rank: usize,
        shape: Rank(rank),
        device: Device,
        dtype: Dtype,
        data: []const u8,

        const Self = @This();
        pub fn zeros() Self {
            return Self{ .rank = rank, .shape = shape, .device = device, .dtype = dtype, .data = undefined };
            // return RealTensor{ .data = "x" };
        }

        pub fn add(self: Self, other: Self) void {
            std.debug.assert(self.shape.len == other.shape.len);
            for (self.shape, other.shape) |d1, d2| {
                std.debug.assert(std.meta.eql(d1, d2));
            }
        }

        pub fn split(self: Self, comptime dim: usize, comptime into: usize) [into]Self {
            comptime std.debug.assert(shape[dim].value() % into == 0);
            var newshape = shape;
            newshape[dim] = Dim{ .static = newshape[dim].value() / into };

            var result: [into]Self = undefined;
            for (0..into) |index| {
                result[index] = Self{ .rank = rank, .shape = newshape, .device = device, .dtype = dtype, .data = self.data };
            }
            return result;
        }

        pub fn get_rank(self: Self) usize {
            return self.shape.len;
        }

        pub fn total_size(self: Self) usize {
            var prod: usize = 1;
            for (self.shape) |dim| {
                prod *= dim.value();
            }
            return prod;
        }

        pub fn byte_size(self: Self) usize {
            var prod: usize = 1;
            for (self.shape) |dim| {
                prod *= dim.value();
            }
            return prod * dtype.size();
        }
    };
}

pub fn main() !void {
    const dim1 = Dim{ .static = 2 };
    const dim2 = Dim{ .static = 3 };
    const seqlen = Dim{ .dyn = 's' };
    // const a = Tensor([2]Dim{ dim1, dim2 }, Device.cpu, Dtype.f32).zeros();
    const a = Tensor(2, [2]Dim{ dim1, dim2 }, Device.cpu, Dtype.f32).zeros();
    std.debug.print("{}\n", .{a});
    const input_ids = Tensor(1, [1]Dim{seqlen}, Device.cpu, Dtype.f32).zeros();
    std.debug.print("{:>}", .{input_ids});

    var b = Tensor(2, [2]Dim{ dim1, dim2 }, Device.cpu, Dtype.f32).zeros();
    b.add(a);
    // const dim3 = Dim{ .static = 4 };
    // var c = Tensor(2, [2]Dim{ dim1, dim3 }, Device.cpu, Dtype.f32).zeros();
    // c.add(a);
}

test "basic add functionality" {
    const dim1 = Dim{ .static = 2 };
    const dim2 = Dim{ .static = 3 };
    const dim3 = Dim{ .static = 4 };

    const tensor = Tensor(2, [2]Dim{ dim1, dim2 }, Device.cpu, Dtype.f32).zeros();
    try testing.expect(tensor.device == Device.cpu);
    try testing.expect(tensor.shape[0].value() == 2);
    try testing.expect(tensor.rank == 2);
    try testing.expect(tensor.total_size() == 6);
    try testing.expect(tensor.byte_size() == 24);

    const tensor2 = Tensor(3, [3]Dim{ dim1, dim2, dim3 }, Device.cpu, Dtype.f16).zeros();
    try testing.expect(tensor2.device == Device.cpu);
    try testing.expect(tensor2.shape[0].value() == 2);
    try testing.expect(tensor2.rank == 3);
    try testing.expect(tensor2.total_size() == 24);
    try testing.expect(tensor2.byte_size() == 48);

    const splits = tensor2.split(0, 2);
    const left = splits[0];
    const right = splits[1];
    try testing.expect(left.rank == 3);
    try testing.expect(left.shape[0].value() == 1);
    try testing.expect(right.shape[0].value() == 1);

    const tensor3 = Tensor(3, [3]Dim{ dim1, dim2, dim3 }, Device.cpu, Dtype.f16).zeros();
    try testing.expect(tensor3.rank == 3);
    try testing.expect(tensor3.shape[0].value() == 2);
    try testing.expect(tensor3.shape[1].value() == 3);

    const newsplits = tensor2.split(1, 3);
    const l = newsplits[0];
    const m = newsplits[1];
    const r = newsplits[2];
    try testing.expect(l.shape[0].value() == 2);
    try testing.expect(l.shape[1].value() == 1);
    try testing.expect(m.shape[1].value() == 1);
    try testing.expect(r.shape[1].value() == 1);

    // comptime assert
    // const splits2 = tensor2.split(1, 2);
    // try testing.expect(splits2.len == 3);
}
