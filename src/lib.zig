const std = @import("std");

pub const Device = enum {
    cpu,
    cuda,

    pub fn format(
        self: Device,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        const val = switch (self) {
            .cpu => "cpu",
            .cuda => "cuda",
        };

        try writer.print("{s}", .{val});
    }
};
pub const Dtype = enum {
    f32,
    f16,
    i32,
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

    pub fn format(
        self: Dtype,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        const val = switch (self) {
            .f32 => "f32",
            .f16 => "f16",
        };

        try writer.print("{s}", .{val});
    }
};

const Rank0 = Rank(0);
const Rank1 = Rank(1);
const Rank2 = Rank(2);
const Rank3 = Rank(3);

pub const Dim = union(enum) {
    dyn: u8, // Char
    static: usize,

    pub fn value(self: Dim) usize {
        const val = switch (self) {
            .dyn => @panic("Dynamic shapes have no value"),
            .static => self.static,
        };
        return val;
    }

    pub fn format(
        self: Dim,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("{}", .{self.value()});
    }
};

const SEQ_LEN = Dim("S");
const HIDDEN_SIZE = Dim(4096);

pub fn Rank(comptime n: comptime_int) type {
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

pub fn Tensor(comptime rank: usize, comptime shape: Rank(rank), comptime device: Device, comptime dtype: Dtype) type {
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

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = self;
            _ = fmt;
            _ = options;

            try writer.print("Tensor([", .{});
            for (shape, 0..) |dim, i| {
                if (i == 0) {
                    try writer.print("{}", .{dim});
                } else {
                    try writer.print(", {}", .{dim});
                }
            }
            try writer.print("], {}, {})", .{ device, dtype });
        }

        pub fn add(self: Self, other: Self) void {
            std.debug.assert(self.shape.len == other.shape.len);
            for (self.shape, other.shape) |d1, d2| {
                std.debug.assert(std.meta.eql(d1, d2));
            }
        }

        const Info = struct {
            dim: usize,
            into: usize,
        };
        pub fn split2(self: Self, comptime info: Info) t: {
            const dim = info.dim;
            const into = info.into;
            std.debug.assert(shape[dim].value() % into == 0);
            var newshape = shape;
            newshape[dim] = Dim{ .static = newshape[dim].value() / into };
            const Return = [into]Tensor(rank, newshape, device, dtype);
            break :t Return;
        } {
            _ = self;
            return undefined;
        }

        pub fn split(self: Self, comptime dim: usize, comptime dims: usize, comptime into_dims: [dims]Dim) t: {
            var sum = 0;
            for (into_dims) |into_dim| {
                sum = sum + into_dim.value();
            }
            // std.debug.assert(shape[dim].value() == sum);

            var newshape = shape;
            var types: [dims]type = undefined;
            for (into_dims, 0..) |into_dim, i| {
                newshape[dim] = into_dim;
                types[i] = Tensor(rank, newshape, device, dtype);
            }
            const Return = std.meta.Tuple(&types);
            break :t Return;
        } {
            _ = self;
            return undefined;
        }

        pub fn get_rank(self: Self) usize {
            return self.shape.len;
        }

        pub fn swiglu(self: Self) Self {
            return self;
        }

        pub fn mul(self: Self, other: Self) Self {
            _ = other;
            return self;
        }

        pub fn matmul_t(self: Self, comptime out: Dim, other: Tensor(rank, [2]Dim{ out, shape[shape.len - 1] }, device, dtype)) Tensor(rank, [rank]Dim{ shape[shape.len - 2], out }, device, dtype) {
            _ = other;
            _ = self;
            const Out = Tensor(rank, [2]Dim{ shape[shape.len - 2], out }, device, dtype);
            return Out.zeros();
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

test "basic add functionality" {
    const testing = std.testing;
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
