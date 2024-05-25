const std = @import("std");
const d = @import("device.zig");
const config = @import("config");
const kernel_compilation = if (@hasDecl(config, "kernel_compilation")) config.kernel_compilation else false;
const kernel_comptime = @import("kernel_comptime.zig");
const Device = d.Device;

pub const DimValue = union(enum) {
    static: usize,
    dynamic: *usize,

    pub fn get(self: @This()) usize {
        switch (self) {
            .static => |v| return v,
            .dynamic => |v| return v.*,
        }
    }
};
pub const Dim = struct {
    name: u8,
    value: DimValue,
    const Self = @This();

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        switch (self.value) {
            .static => |v| try writer.print("{c}({})", .{ self.name, v }),
            .dynamic => |_| try writer.print("{c}(?)", .{self.name}),
        }
    }

    pub fn static(name: u8, value: usize) Self {
        return Dim{ .name = name, .value = DimValue{ .static = value } };
    }

    pub fn dynamic(name: u8, value: *usize) Self {
        return Dim{ .name = name, .value = DimValue{ .dynamic = value } };
    }
};

pub fn Tensor(comptime device: Device, comptime T: type, comptime rank: usize, comptime shape: [rank]Dim) type {
    const nelements = d.total_size(&shape);
    const DataType = device.data_type(T, nelements);
    const DeviceType = device.device_type();
    return struct {
        device: DeviceType,
        data: DataType,
        const Self = @This();

        pub fn init(data: device.data_type(T, nelements), realdevice: anytype) !Self {
            return Self{ .data = data, .device = realdevice };
        }

        pub fn ptr(self: Self) DataType.Ptr {
            return self.data.ptr();
        }

        pub fn byte_size(self: Self) usize {
            return self.data.byte_size();
        }

        /// TODO Make this easier to use by doing type reflection on other directly
        pub fn copy_from_device(self: Self, comptime otherdevice: Device, other: *const Tensor(otherdevice, T, rank, shape)) !void {
            return self.device.copy_from(self.data.ptr(), otherdevice, other);
        }

        pub fn matmul_t(self: Self, comptime outdim: Dim, other: Tensor(device, T, 2, [2]Dim{ outdim, shape[1] }), out: Tensor(device, T, 2, [2]Dim{ shape[0], outdim }), realdevice: anytype) !void {
            std.debug.assert(rank == 2);
            const M = shape[shape.len - 2];
            const N = outdim;
            const K = shape[shape.len - 1];
            const m = M.value.get();
            const n = N.value.get();
            const k = K.value.get();
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
