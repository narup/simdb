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

// cosine_similarity calculates the cosine(a, b) = dot(a, b) / (magnitude(a) * magnitude(b))
pub fn cosine_similarity(a: []const f32, b: []const f32) f32 {
    return dot_product(a, b) / (magnitude(a) * magnitude(b));
}

// euclidean_distance calculates the mentioned between 2 vectors
// euclidean(a, b) = √((a₁-b₁)² + (a₂-b₂)² + ...)
pub fn euclidean_distance(a: []const f32, b: []const f32) f32 {
    const chunk_size: u8 = 8;

    var index: usize = 0;
    var sum: f32 = 0;
    while (index + chunk_size <= a.len) : (index += chunk_size) {
        const va: @Vector(chunk_size, f32) = a[index..][0..chunk_size].*;
        const vb: @Vector(chunk_size, f32) = b[index..][0..chunk_size].*;

        const diff = (va - vb);
        sum += @reduce(.Add, diff * diff);
    }

    //handle the tail
    while (index < a.len) : (index += 1) {
        const diff = a[index] - b[index];
        sum += diff * diff;
    }

    return math.sqrt(sum);
}

fn magnitude(a: []const f32) f32 {
    return math.sqrt(dot_product(a, a));
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

// ============ cosine_similarity tests ============

test "cosineSimilarity - identical vectors" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 1, 2, 3 };
    const result = cosine_similarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.0001);
}

test "cosineSimilarity - orthogonal vectors" {
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 0, 1, 0 };
    const result = cosine_similarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 0.0001);
}

test "cosineSimilarity - opposite vectors" {
    const a = [_]f32{ 1, 0 };
    const b = [_]f32{ -1, 0 };
    const result = cosine_similarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), result, 0.0001);
}

test "cosineSimilarity - SIMD path (8 elements)" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const result = cosine_similarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.0001);
}

// ============ euclidean_distance tests ============

test "euclideanDistance - 3-4-5 triangle" {
    const a = [_]f32{ 0, 0 };
    const b = [_]f32{ 3, 4 };
    const result = euclidean_distance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result, 0.0001);
}

test "euclideanDistance - same point" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 1, 2, 3 };
    const result = euclidean_distance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 0.0001);
}

test "euclideanDistance - SIMD path (8 elements)" {
    const a = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const b = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    const result = euclidean_distance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 2.828427), result, 0.0001); // sqrt(8)
}

test "euclideanDistance - SIMD + tail (10 elements)" {
    const a = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const b = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    const result = euclidean_distance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 3.162278), result, 0.0001); // sqrt(10)
}
