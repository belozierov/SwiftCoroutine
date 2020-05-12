//
//  CoScope.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

/// The holder of uncompleted `CoCancellable` and coroutines.
///
/// `CoScope` helps to manage lifecycle of coroutines and `CoCancellable`, like `CoFuture` and `CoChannel`.
/// It keeps weak references on inner objects and cancels them on `cancel()` or deinit.
/// All completed objects are automaticaly removed from scope.
///
/// ```
/// let scope = CoScope()
///
/// let future = makeSomeFuture().added(to: scope)
///
/// queue.startCoroutine(in: scope) {
///     . . . some code . . .
///     let result = try future.await()
///     . . . some code . . .
/// }
///
/// let future2 = queue.coroutineFuture {
///     try Coroutine.delay(.seconds(5)) // imitate some work
///     return 5
/// }.added(to: scope)
///
/// //cancel all added futures and coroutines
/// scope.cancel()
/// ```
///
public final class CoScope {
    
    private typealias Block = () -> Void
    
    private var items = [Int: Block]()
    private var callbacks = [Block]()
    private var state = Int.free
    private var keyCounter = 0
    
    /// Initializes a scope.
    public init() {}
    
    /// Adds `CoCancellable` to be canceled when the scope is being canceled or deinited.
    /// - Parameter item: `CoCancellable` to add.
    @inlinable public func add(_ item: CoCancellable) {
        item.whenComplete(add { [weak item] in item?.cancel() })
    }
    
    @usableFromInline internal func add(_ cancel: @escaping () -> Void) -> () -> Void {
        let key = atomicAdd(&keyCounter, value: 1)
        addItem(for: key, block: cancel)
        return { [weak self] in self?.removeItem(for: key) }
    }
    
    /// Returns `true` if the scope is empty (contains no `CoCancellable`).
    public var isEmpty: Bool {
        if isCanceled { return true }
        var isEmpty = true
        locked { isEmpty = items.isEmpty }
        return isEmpty
    }
    
    // MARK: - items
    
    private func addItem(for key: Int, block: @escaping Block) {
        if !locked({ items[key] = block }) { block() }
    }
    
    private func removeItem(for key: Int) {
        locked { items[key] = nil }
    }
    
    // MARK: - lock
    
    @discardableResult private func locked(_ block: Block) -> Bool {
        while true {
            switch state {
            case .free:
                if atomicCAS(&state, expected: .free, desired: .busy) {
                    block()
                    unlock()
                    return true
                }
            case .busy:
                continue
            default:
                return false
            }
        }
    }
    
    private func unlock() {
        while true {
            switch state {
            case .busy:
                if atomicCAS(&state, expected: .busy, desired: .free) { return }
            default:
                return completeAll()
            }
        }
    }
    
    // MARK: - cancel
    
    /// Returns `true` if the scope is canceled.
    public var isCanceled: Bool {
        state == .canceled
    }
    
    /// Cancels the scope and all `CoCancellable` that it contains.
    public func cancel() {
        while true {
            switch state {
            case .free:
                if !atomicCAS(&state, expected: .free, desired: .canceled) { continue }
                return completeAll()
            case .busy:
                if atomicCAS(&state, expected: .busy, desired: .canceled) { return }
            default:
                return
            }
        }
    }
    
    /// Adds an observer callback that is called when the `CoScope` is canceled or deinited.
    /// - Parameter callback: The callback that is called when the scope is canceled or deinited.
    public func whenComplete(_ callback: @escaping () -> Void) {
        if !locked({ callbacks.append(callback) }) { callback() }
    }
    
    private func completeAll() {
        items.values.forEach { $0() }
        items.removeAll()
        callbacks.forEach { $0() }
        callbacks.removeAll()
    }
    
    deinit {
        if !isCanceled { completeAll() }
    }
    
}

fileprivate extension Int {
    
    static let canceled = -1
    static let free = 0
    static let busy = 1
    
}
