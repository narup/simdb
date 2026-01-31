const std = @import("std");
const math = std.math;

pub fn SimVector(comptime T: type, len: comptime_int) type {
    return struct {
        vec: @Vector(len, T),
        size: comptime_int,

        const Self = @This();

        pub fn init(items: [len]T) SimVector(T, len) {
            return .{ .vec = items, .size = len };
        }

        //magnitude of vector a -> |a| = √(a₁² + a₂² + a₃² + ... + aₙ²)
        pub fn magnitude(self: Self) f32 {
            const vector_sum = @reduce(.Add, (self.vec * self.vec));

            const sum_f32 = switch (@typeInfo(T)) {
                .int, .comptime_int => @as(f32, @floatFromInt(vector_sum)),
                .float, .comptime_float => @as(f32, @floatCast(vector_sum)),
                else => @compileError("Sim vector requires numeric types"),
            };
            return math.sqrt(sum_f32);
        }

        pub fn dot_product(self: Self, other: Self) T {
            return @reduce(.Add, self.vec * other.vec);
        }

        pub fn cosine_similarity(self: Self, other: Self) f32 {
            const dot_product_val = self.dot_product(other);
            const denominator = self.magnitude() * other.magnitude();
            const float_dot_prod = switch (@typeInfo(T)) {
                .int, .comptime_int => @as(f32, @floatFromInt(dot_product_val)),
                .float, .comptime_float => @as(f32, @floatCast(dot_product_val)),
                else => @compileError("Sim vector requires numeric types"),
            };
            return float_dot_prod / denominator;
        }
    };
}
