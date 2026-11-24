import Foundation
#if canImport(CoreAI)
import CoreAI
import Synchronization
#endif

private struct BridgeError: Error, CustomStringConvertible {
    let description: String
    init(_ text: String) { description = text }
}

@c(ferment_coreai_free_string)
public func freeString(_ value: UnsafeMutablePointer<CChar>?) { free(value) }

@c(ferment_coreai_architecture)
public func architecture() -> UnsafeMutablePointer<CChar>? {
    #if canImport(CoreAI)
    if #available(macOS 27, *) { return strdup(AIModel.deviceArchitectureName) }
    #endif
    return nil
}

// The semaphore publishes the result before the caller reads it. This wait is
// used only on Rust's dedicated native load/prediction workers, never the UI or
// a Swift cooperative executor. The detached task owns its inputs until done.
private final class Reply<T: Sendable>: @unchecked Sendable {
    let ready = DispatchSemaphore(value: 0)
    var value: Result<T, Error>?
}
private func blocking<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
    let reply = Reply<T>()
    Task.detached {
        do { reply.value = .success(try await operation()) }
        catch { reply.value = .failure(error) }
        reply.ready.signal()
    }
    reply.ready.wait()
    guard let result = reply.value else { throw BridgeError("CoreAI lost its result") }
    return try result.get()
}

#if canImport(CoreAI)
@available(macOS 27, *)
private final class ModelBox: Sendable {
    let model: AIModel
    private let functions = Mutex<[String: InferenceFunction]>([:])
    init(_ model: AIModel) { self.model = model }

    func function(named name: String) throws -> InferenceFunction {
        try functions.withLock { loaded in
            if let function = loaded[name] { return function }
            guard let function = try model.loadFunction(named: name) else {
                throw BridgeError("CoreAI function is absent")
            }
            loaded[name] = function
            return function
        }
    }
}
@available(macOS 27, *)
private struct Output: Sendable {
    let name: String
    let shape: [Int64]
    let dtype: Int32
    let data: Data
}

private func elementCount(_ shape: [Int]) throws -> Int {
    var count = 1
    guard shape.count <= 16 else { throw BridgeError("Tensor rank exceeds 16") }
    for dim in shape {
        let next = count.multipliedReportingOverflow(by: dim)
        guard dim > 0, !next.overflow else { throw BridgeError("Invalid tensor shape") }
        count = next.partialValue
    }
    return count
}
private func read<T: BitwiseCopyable>(_ ptr: UnsafeRawPointer, _ count: Int, as: T.Type) -> [T] {
    (0..<count).map { ptr.loadUnaligned(fromByteOffset: $0 * MemoryLayout<T>.stride, as: T.self) }
}

@available(macOS 27, *)
private func input(_ ptr: UnsafeRawPointer, count: Int, dtype: Int32, shape: [Int], expected: NDArray.ScalarType) throws -> NDArray {
    switch (dtype, expected) {
    case (1, .float32): return NDArray(scalars: read(ptr, count, as: Float.self), shape: shape)
    case (2, .float16): return NDArray(scalars: read(ptr, count, as: Float16.self), shape: shape)
    case (3, .int32): return NDArray(scalars: read(ptr, count, as: Int32.self), shape: shape)
    case (4, .int64): return NDArray(scalars: read(ptr, count, as: Int64.self), shape: shape)
    case (1, .float16): return NDArray(scalars: read(ptr, count, as: Float.self).map(Float16.init), shape: shape)
    case (2, .float32): return NDArray(scalars: read(ptr, count, as: Float16.self).map(Float.init), shape: shape)
    case (3, .int64): return NDArray(scalars: read(ptr, count, as: Int32.self).map(Int64.init), shape: shape)
    case (4, .int32):
        let values = try read(ptr, count, as: Int64.self).map { value in
            guard let narrowed = Int32(exactly: value) else { throw BridgeError("Token exceeds int32") }
            return narrowed
        }
        return NDArray(scalars: values, shape: shape)
    default: throw BridgeError("Unsupported CoreAI input dtype \(dtype) → \(expected)")
    }
}

@available(macOS 27, *)
private func copy<T: BitwiseCopyable>(_ array: NDArray, as: T.Type) throws -> [T] {
    let shape = array.shape
    let count = try elementCount(shape)
    let layout = array.interleaveLayout
    if let layout {
        guard shape.indices.contains(layout.dimension), layout.factor > 0 else {
            throw BridgeError("Invalid CoreAI interleave layout")
        }
    }
    return array.view(as: T.self).withUnsafePointer { pointer, _, strides in
        // An interleaved axis reports a BLOCK stride, not an element stride.
        // Keep the owning output alive throughout this logical-coordinate copy.
        (0..<count).map { linear in
            var remainder = linear, offset = 0
            for axis in shape.indices.reversed() {
                let index = remainder % shape[axis]
                if let layout, axis == layout.dimension {
                    offset += (index / layout.factor) * strides[axis] + index % layout.factor
                } else {
                    offset += index * strides[axis]
                }
                remainder /= shape[axis]
            }
            return pointer[offset]
        }
    }
}

@available(macOS 27, *)
private func materialize(_ array: NDArray, name: String) throws -> Output {
    let data: Data, dtype: Int32
    switch array.scalarType {
    case .float16:
        data = try copy(array, as: Float16.self).map(Float.init).withUnsafeBytes { Data($0) }; dtype = 1
    case .float32:
        data = try copy(array, as: Float.self).withUnsafeBytes { Data($0) }; dtype = 1
    case .int32:
        data = try copy(array, as: Int32.self).withUnsafeBytes { Data($0) }; dtype = 3
    case .int64:
        data = try copy(array, as: Int64.self).withUnsafeBytes { Data($0) }; dtype = 4
    default: throw BridgeError("Unsupported CoreAI output dtype \(array.scalarType)")
    }
    return Output(name: name, shape: array.shape.map(Int64.init), dtype: dtype, data: data)
}
#endif

@c(ferment_coreai_load)
public func load(_ path: UnsafePointer<CChar>, _ frequentReshapes: Int32, _ error: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> UnsafeMutableRawPointer? {
    do {
        #if canImport(CoreAI)
        if #available(macOS 27, *) {
            let url = URL(fileURLWithPath: String(cString: path))
            guard frequentReshapes == 0 || frequentReshapes == 1 else { throw BridgeError("Invalid reshape option") }
            let box = try blocking {
                var options = SpecializationOptions.default
                options.expectFrequentReshapes = frequentReshapes == 1
                return ModelBox(try await AIModel(contentsOf: url, options: options))
            }
            return Unmanaged.passRetained(box).toOpaque()
        }
        #endif
        throw BridgeError("CoreAI requires macOS 27 and a supported SDK")
    } catch let failure { error.pointee = strdup(String(describing: failure)); return nil }
}

@c(ferment_coreai_release)
public func release(_ handle: UnsafeMutableRawPointer) {
    #if canImport(CoreAI)
    if #available(macOS 27, *) { Unmanaged<ModelBox>.fromOpaque(handle).release() }
    #endif
}

public typealias OutputCallback = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>, UnsafePointer<Int64>?, Int, Int32, UnsafeRawPointer?, Int) -> Int32

@c(ferment_coreai_predict)
public func predict(
    _ handle: UnsafeMutableRawPointer, _ entrypoint: UnsafePointer<CChar>, _ count: Int,
    _ names: UnsafePointer<UnsafePointer<CChar>>, _ shapes: UnsafePointer<UnsafePointer<Int64>>,
    _ ranks: UnsafePointer<Int>, _ types: UnsafePointer<Int32>,
    _ buffers: UnsafePointer<UnsafeRawPointer>, _ lengths: UnsafePointer<Int>,
    _ context: UnsafeMutableRawPointer?, _ callback: OutputCallback
) -> UnsafeMutablePointer<CChar>? {
    do {
        #if canImport(CoreAI)
        if #available(macOS 27, *) {
            let box = Unmanaged<ModelBox>.fromOpaque(handle).takeUnretainedValue()
            let function = try box.function(named: String(cString: entrypoint))
            guard count == function.descriptor.inputCount else { throw BridgeError("Wrong CoreAI input count") }
            var inputs: [String: NDArray] = [:]
            for i in 0..<count {
                guard ranks[i] >= 0 && ranks[i] <= 16 else { throw BridgeError("Invalid input rank") }
                let name = String(cString: names[i])
                let shape = Array(UnsafeBufferPointer(start: shapes[i], count: ranks[i])).map(Int.init)
                let elements = try elementCount(shape)
                let width = types[i] == 2 ? 2 : (types[i] == 4 ? 8 : 4)
                let bytes = elements.multipliedReportingOverflow(by: width)
                guard !bytes.overflow, lengths[i] == bytes.partialValue,
                      inputs[name] == nil,
                      case .ndArray(let descriptor)? = function.descriptor.inputDescriptor(of: name) else {
                    throw BridgeError("Invalid input \(name)")
                }
                inputs[name] = try input(buffers[i], count: elements, dtype: types[i], shape: shape, expected: descriptor.scalarType)
            }
            let ownedInputs = inputs
            let outputs = try blocking {
                var raw = try await function.run(inputs: ownedInputs)
                var outputs: [Output] = []
                for name in Array(raw.names) {
                    guard let value = raw.remove(name), let array = value.ndArray else { throw BridgeError("Non-tensor output \(name)") }
                    outputs.append(try materialize(array, name: name))
                }
                return outputs
            }
            for output in outputs {
                let status = output.name.withCString { name in
                    output.shape.withUnsafeBufferPointer { shape in
                        output.data.withUnsafeBytes { data in
                            callback(context, name, shape.baseAddress, shape.count, output.dtype, data.baseAddress, data.count)
                        }
                    }
                }
                guard status == 0 else { throw BridgeError("CoreAI output consumer refused result") }
            }
            return nil
        }
        #endif
        throw BridgeError("CoreAI is unavailable")
    } catch { return strdup(String(describing: error)) }
}
