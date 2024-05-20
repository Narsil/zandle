const std = @import("std");
pub const cu = @cImport({
    @cInclude("cuda.h");
});

const Type = std.builtin.Type;

const CudaErrorEnum: type = ErrorsToEnum(u32, cu);
pub const CudaError = EnumToError(CudaErrorEnum);

pub fn EnumToError(comptime some_type: type) type {
    switch (@typeInfo(some_type)) {
        .Enum => |enum_type| {
            var error_names: [enum_type.fields.len]Type.Error = undefined;
            for (enum_type.fields, 0..) |field, index| {
                error_names[index] = .{ .name = field.name };
            }
            const error_type = @Type(std.builtin.Type{ .ErrorSet = &error_names });
            return struct {
                pub const Error: type = error_type;
                pub fn from_error_code(x: enum_type.tag_type) ?error_type {
                    const enum_val: some_type = @enumFromInt(x);
                    inline for (@typeInfo(error_type).ErrorSet orelse unreachable) |error_val| {
                        if (std.mem.eql(u8, error_val.name, @tagName(enum_val))) {
                            return @field(error_type, error_val.name);
                        }
                    }
                    return null;
                }
            };
        },
        else => @compileError("Cannot convert non enum type to Error type"),
    }
}

fn ErrorsToEnum(comptime tag_type: type, comptime cimport: type) type {
    const prefix = switch (cimport) {
        cu => "CUDA_ERROR",
        // cublas => "CUBLAS_STATUS",
        else => @compileError("Can either be cuda / nvrtc type"),
    };
    switch (@typeInfo(cimport)) {
        .Struct => |x| {
            const null_decls: []const Type.Declaration = &.{};
            var errors: [100]Type.EnumField = undefined;
            var counter: usize = 0;
            @setEvalBranchQuota(100000);
            for (x.decls) |declaration| {
                if (std.mem.startsWith(u8, declaration.name, prefix)) {
                    errors[counter] = .{ .name = declaration.name, .value = @field(cimport, declaration.name) };
                    counter += 1;
                }
            }

            return @Type(Type{ .Enum = .{ .tag_type = tag_type, .fields = errors[0..counter], .decls = null_decls, .is_exhaustive = false } });
        },
        else => @compileError("Cannot generate error type"),
    }
}

pub fn fromCudaErrorCode(error_code: u32) CudaError.Error!void {
    if (error_code == 0) return;
    const val = CudaError.from_error_code(error_code).?;
    return val;
}

pub fn Cuda(error_code: u32) CudaError.Error!void {
    return fromCudaErrorCode(error_code);
}

// pub fn Cublas(error_code: u32) CublasError.Error!void {
//     return fromCublasErrorCode(error_code);
// }
//
// pub fn fromCublasErrorCode(error_code: u32) CublasError.Error!void {
//     if (error_code == 0) return;
//     const val = CublasError.from_error_code(error_code).?;
//     return val;
// }
