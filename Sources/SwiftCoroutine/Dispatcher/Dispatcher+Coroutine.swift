//
//  Dispatcher+Async.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

// MARK: - Dispatcher

extension Coroutine {
    
    @inlinable public static func setDispatcher(_ dispatcher: Dispatcher) throws {
        assert(isInsideCoroutine, "setDispatcher must be called inside coroutine")
        try Coroutine.current().restart(with: dispatcher)
    }
    
}

// MARK: - Async

@inlinable public func async(on dispatcher: Coroutine.Dispatcher = .global, execute work: @escaping () -> Void) {
    dispatcher.perform(work: work)
}

@inlinable public func async<T>(on dispatcher: Coroutine.Dispatcher = .global, execute work: @escaping () throws -> T) -> CoFuture<T> {
    let item = CoPromise<T>()
    dispatcher.perform { item.perform(work) }
    return item
}

// MARK: - Coroutine

@inlinable public func coroutine(on dispatcher: Coroutine.Dispatcher = .current,
                                 stackSize: Coroutine.StackSize = .recommended,
                                 execute work: @escaping () throws -> Void) {
    Coroutine(stackSize: stackSize, dispatcher: dispatcher).start { try? work() }
}

@inlinable public func coroutine<T>(on dispatcher: Coroutine.Dispatcher = .current,
                                    stackSize: Coroutine.StackSize = .recommended,
                                    execute work: @escaping () throws -> T) -> CoFuture<T> {
    let item = CoPromise<T>()
    coroutine(on: dispatcher, stackSize: stackSize) { item.perform(work) }
    return item
}

// MARK: - Compose

@inlinable public func compose<T>(@CoFututeComposite<T> builder: @escaping () -> [CoFuture<T>]) -> CoFuture<[T]> {
    coroutine(on: .sync, stackSize: .minimal) { try builder().map { try $0.await() } }
}
