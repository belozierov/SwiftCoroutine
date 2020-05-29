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
/// - Note: `CoScope` keeps weak references.
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
    
    internal typealias Completion = () -> Void
    private var storage = Storage<Completion>()
    private var callbacks = CallbackStack<Void>()
    
    /// Initializes a scope.
    public init() {}
    
    /// Adds weak referance of `CoCancellable` to be canceled when the scope is being canceled or deinited.
    /// - Parameter item: `CoCancellable` to add.
    public func add(_ item: CoCancellable) {
        add { [weak item] in item?.cancel() }.map(item.whenComplete)
    }
    
    internal func add(_ cancel: @escaping () -> Void) -> Completion? {
        if isCanceled { cancel(); return nil }
        let key = storage.append(cancel)
        if isCanceled { storage.remove(key)?(); return nil }
        return { [weak self] in self?.remove(key: key) }
    }
    
    private func remove(key: Storage<Completion>.Index) {
        storage.remove(key)
    }
    
    /// Returns `true` if the scope is empty (contains no `CoCancellable`).
    public var isEmpty: Bool {
        isCanceled || storage.isEmpty
    }
    
    // MARK: - cancel
    
    /// Returns `true` if the scope is canceled.
    public var isCanceled: Bool {
        callbacks.isClosed
    }
    
    /// Cancels the scope and all `CoCancellable` that it contains.
    public func cancel() {
        if isCanceled { return }
        completeAll()
        storage.removeAll()
    }
    
    private func completeAll() {
        callbacks.close()?.finish(with: ())
        storage.forEach { $0() }
    }
    
    /// Adds an observer callback that is called when the `CoScope` is canceled or deinited.
    /// - Parameter callback: The callback that is called when the scope is canceled or deinited.
    public func whenComplete(_ callback: @escaping () -> Void) {
        if !callbacks.append(callback) { callback() }
    }
    
    deinit {
        if !isCanceled { completeAll() }
        storage.free()
    }
    
}
