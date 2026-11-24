// Append to Bridge.swift for this standalone native test, so the actual private
// materializer is graded without exporting a test-only production entry point.
// Oracle: physical arrays below, interpreted by Apple's documented NDArray
// InterleaveLayout contract. Kill: ignore the interleaved block stride, as the
// initial bridge did (RGB becomes [1,4,0,0,0,0]).
@available(macOS 27, *)
func checkLayout<T: BitwiseCopyable & Equatable>(
    _ label: String, type: NDArray.ScalarType, shape: [Int], strides: [Int],
    layout: NDArray.InterleaveLayout?, physical: [T], expected: [T]
) throws {
    var array: NDArray
    if let layout {
        array = NDArray(shape: shape, scalarType: type, strides: strides, interleaveLayout: layout)
    } else {
        array = NDArray(shape: shape, scalarType: type, strides: strides)
    }
    array.mutableView(as: T.self).withUnsafeMutablePointer { pointer, _, _ in
        for (index, value) in physical.enumerated() { pointer[index] = value }
    }
    let actual = try copy(array, as: T.self)
    guard actual == expected else {
        throw BridgeError("\(label): \(actual) != \(expected)")
    }
    print("PASS \(label)")
}

if #available(macOS 27, *) {
    let rgb = NDArray.InterleaveLayout(dimension: 0, factor: 3)
    try checkLayout("RGB float32", type: .float32, shape: [3, 1, 2], strides: [6, 6, 3],
                    layout: rgb, physical: [Float(1),2,3,4,5,6], expected: [1,4,2,5,3,6])
    try checkLayout("RGB float16", type: .float16, shape: [3, 1, 2], strides: [6, 6, 3],
                    layout: rgb, physical: [Float16(1),2,3,4,5,6], expected: [1,4,2,5,3,6])
    try checkLayout("RGB int32", type: .int32, shape: [3, 1, 2], strides: [6, 6, 3],
                    layout: rgb, physical: [Int32(1),2,3,4,5,6], expected: [1,4,2,5,3,6])
    try checkLayout("RGB int64", type: .int64, shape: [3, 1, 2], strides: [6, 6, 3],
                    layout: rgb, physical: [Int64(1),2,3,4,5,6], expected: [1,4,2,5,3,6])
    // This SDK traps when a dimension is not divisible by its factor, although
    // the current documentation permits it. Exercise two complete blocks here.
    try checkLayout("multiple interleaved blocks", type: .float32, shape: [6, 1, 2], strides: [6, 6, 3],
                    layout: rgb, physical: [Float(1),2,3,4,5,6,7,8,9,10,11,12],
                    expected: [1,4,2,5,3,6,7,10,8,11,9,12])
    try checkLayout("padded rows", type: .float32, shape: [2, 3], strides: [5, 1],
                    layout: nil, physical: [Float(1),2,3,-1,-1,4,5,6], expected: [1,2,3,4,5,6])
    try checkLayout("scalar", type: .float32, shape: [], strides: [],
                    layout: nil, physical: [Float(7)], expected: [7])
    print("7 native NDArray layout checks passed")
} else {
    fputs("SKIP: native NDArray tests require macOS 27\n", stderr)
    exit(77)
}
