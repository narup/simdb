const std = @import("std");
const simdb = @import("simdb");
const vector = @import("vector.zig");
const distance = @import("distance.zig");

const Stack = @import("stack.zig").Stack;
const Perceptron = @import("perceptron.zig").Perceptron;

const expect = @import("std").testing.expect;

pub fn main() !void {
    // Prints to stderr, ignoring potential errors.
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    try simdb.bufferedPrint();

    const a1 = IntArray(3){ 1, 2, 3 };
    const a2 = IntArray(5){ 1, 3, 3, 4, 5 };

    std.debug.print("Array a1: {any}\n", .{a1});
    std.debug.print("Array a2: {any}\n", .{a2});

    try callStack();
    try vector_operations();
    try perceptron_operations();
}

fn vector_operations() !void {
    std.debug.print("===== sim vector operations =====\n", .{});

    const sv1 = vector.SimVector(u32, 5).init([_]u32{ 0, 3, 4, 5, 6 });

    const sv1_magnitude = sv1.magnitude();
    std.debug.print("Vector1 magnitude: {d}\n", .{sv1_magnitude});

    const sv2 = vector.SimVector(u32, 5).init([_]u32{ 4, 5, 6, 7, 8 });
    const sv2_magnitude = sv2.magnitude();
    std.debug.print("Vector2 magnitude: {d}\n", .{sv2_magnitude});

    const dot_product = sv1.dot_product(sv2);
    std.debug.print("Vector1 and Vector 2 dot product: {d}\n", .{dot_product});

    const cosine_val = sv1.cosine_similarity(sv2);
    std.debug.print("Vector1 and Vector 2 cosine val: {d}\n", .{cosine_val});

    const fv1 = vector.SimVector(f32, 5).init([_]f32{ 34.29, 22.12, 1.0, 4, 577.30 });
    const fv2 = vector.SimVector(f32, 5).init([_]f32{ 4.30, 2.52, 8.30, 1.34, 7.60 });

    const fdot_product = fv1.dot_product(fv2);
    std.debug.print("Floating Vector1 and Vector 2 dot product: {d}\n", .{fdot_product});

    const fv_magnitude = fv1.magnitude();
    std.debug.print("Float Vector2 magnitude: {d}\n", .{fv_magnitude});

    std.debug.print("=== end sim vector operations ====\n", .{});
}

fn perceptron_operations() !void {
    std.debug.print("====== Perceptron Operations ===== \n", .{});

    const inputs = [_]f32{ 1.0, 0.5, 0.3 };
    // Initialize with specific weights and bias
    const perceptron = Perceptron(3).init([_]f32{ 0.5, -0.3, 0.8 }, // weights
        -0.2 // bias
    );
    const output_simd = perceptron.activate(inputs, true);
    const output_regular = perceptron.activate(inputs, false);

    std.debug.print("Input: [{d:.2}, {d:.2}, {d:.2}]\n", .{ inputs[0], inputs[1], inputs[2] });
    std.debug.print("Weights: [{d:.2}, {d:.2}, {d:.2}]\n", .{ perceptron.weights[0], perceptron.weights[1], perceptron.weights[2] });
    std.debug.print("Bias: {d:.2}\n", .{perceptron.bias});
    std.debug.print("Output (SIMD): {}\n", .{output_simd});
    std.debug.print("Output (Regular): {}\n", .{output_regular});

    const sigmoid_value = perceptron.sigmoid(inputs, true);
    std.debug.print("Sigmoid Value: {}\n", .{sigmoid_value});
}

fn callStack() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var stack = try Stack(u8).init(allocator, 10);
    defer stack.deinit();

    try stack.push(1);
    try stack.push(2);
    try stack.push(3);
    try stack.push(4);
    try stack.push(5);
    try stack.push(6);

    std.debug.print("Stack len: {d}\n", .{stack.length});
    std.debug.print("Stack capacity: {d}\n", .{stack.capacity});

    stack.pop();
    std.debug.print("Stack len: {d}\n", .{stack.length});
    stack.pop();
    std.debug.print("Stack len: {d}\n", .{stack.length});
    std.debug.print("Stack state: {any}\n", .{stack.items[0..stack.length]});
}

fn IntArray(comptime length: usize) type {
    return [length]i64;
}

fn fibonacci(index: u32) u32 {
    if (index < 2) return index;
    return fibonacci(index - 1) + fibonacci(index - 2);
}

test "fibonacci" {
    // test fibonacci at run-time
    try expect(fibonacci(7) == 13);
    // test fibonacci at compile-time
    try comptime expect(fibonacci(7) == 13);
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
