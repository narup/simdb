const std = @import("std");
const math = std.math;

// dot_product does the vector dot product using SIMD vector of length 8
// which should work on most CPUs
pub fn dot_product(a: []const f32, b: []const f32) f32 {
    const chunk_size: u8 = 8;

    var index: usize = 0;
    var sum: f32 = 0;

    while (index + chunk_size <= a.len) : (index += chunk_size) {
        const va: @Vector(chunk_size, f32) = a[index..][0..chunk_size].*;
        const vb: @Vector(chunk_size, f32) = b[index..][0..chunk_size].*;

        sum += @reduce(.Add, va * vb);
    }

    //handle the remaining tail if vector size is not a multiple of 8 (chunk_size)
    while (index < a.len) : (index += 1) {
        sum += (a[index] * b[index]);
    }

    return sum;
}

test "dotProduct - small array (tail only)" {
    const a = [_]f32{ 1, 2, 3, 4, 5 };
    const b = [_]f32{ 1, 2, 3, 4, 5 };
    const result = dot_product(&a, &b);
    try std.testing.expectEqual(@as(f32, 55.0), result);
}

test "dotProduct - exactly 8 elements (one SIMD pass)" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    const result = dot_product(&a, &b);
    try std.testing.expectEqual(@as(f32, 36.0), result);
}

test "dotProduct - 10 elements (SIMD + tail)" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const b = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    const result = dot_product(&a, &b);
    try std.testing.expectEqual(@as(f32, 55.0), result);
}

test "dotProduct - 16 elements (two SIMD passes)" {
    const a = [_]f32{ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    const b = [_]f32{ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };
    const result = dot_product(&a, &b);
    try std.testing.expectEqual(@as(f32, 96.0), result);
}

test "dotProduct - real floats" {
    const a = [_]f32{ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 };
    const b = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    const result = dot_product(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 26.25), result, 0.0001);
}

test "dotProduct - orthogonal vectors" {
    const a = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0, 0, 0, 0, 0 };
    const result = dot_product(&a, &b);
    try std.testing.expectEqual(@as(f32, 0.0), result);
}
