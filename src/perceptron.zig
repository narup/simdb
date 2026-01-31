const std = @import("std");

pub fn Perceptron(comptime n: usize) type {
    return struct {
        weights: [n]f32,
        bias: f32,

        const Self = @This();

        pub fn init(weights: [n]f32, bias: f32) Self {
            return .{ .weights = weights, .bias = bias };
        }

        pub fn activate_simd(self: Self, inputs: [n]f32) f32 {
            //convert arrays to Vector for SIMD operations
            const w_vec: @Vector(n, f32) = self.weights;
            const x_vec: @Vector(n, f32) = inputs;

            const product = w_vec * x_vec;
            const sum = @reduce(.Add, product) + self.bias;

            return sum;
        }

        pub fn activate_basic(self: Self, inputs: [n]f32) f32 {
            var sum: f32 = 0.0;
            for (self.weights, inputs) |w, x| {
                sum += w * x;
            }
            sum = sum + self.bias;
            return sum;
        }

        pub fn activate(self: Self, inputs: [n]f32, simd: bool) u1 {
            if (simd) {
                return if (self.activate_simd(inputs) > 0) 1 else 0;
            } else {
                return if (self.activate_basic(inputs) > 0) 1 else 0;
            }
        }

        pub fn sigmoid(self: Self, inputs: [n]f32, simd: bool) f32 {
            var z: f32 = 0.0;
            if (simd) {
                z = self.activate_simd(inputs);
            } else {
                z = self.activate_basic(inputs);
            }
            return 1.0 / (1.0 + @exp(-z));
        }
    };
}
