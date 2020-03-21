//
//  Coroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 01.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public struct Coroutine {
    
    public enum State: Int {
        case prepared, running, suspending, suspended, finished, restarting
    }
    
    @usableFromInline let coroutine: CoroutineProtocol
    
    @inlinable init(coroutine: CoroutineProtocol) {
        self.coroutine = coroutine
    }
    
    public init(stackSize: StackSize = .recommended, scheduler: TaskScheduler = .immediate, task: @escaping () -> Void) {
        let context = CoroutineContext(stackSize: stackSize.size)
        coroutine = StackfullCoroutine(context: context, scheduler: scheduler)
        context.block = task
    }
    
    @inlinable public var state: State {
        coroutine.state
    }
    
    // MARK: - resume
    
    @inlinable public func resume() throws {
        try coroutine.resume()
    }
    
    // MARK: - suspend
    
    @inlinable public static func suspend() throws {
        try current().coroutine.suspend()
    }
    
    // MARK: - await
    
    @inlinable public static func await<T>(_ callback: (@escaping (T) -> Void) -> Void) throws -> T {
        try current().coroutine.await(callback)
    }
    
}

extension Coroutine: Hashable {
    
    @inlinable public static func == (lhs: Coroutine, rhs: Coroutine) -> Bool {
        lhs.coroutine === rhs.coroutine
    }
    
    @inlinable public func hash(into hasher: inout Hasher) {
        ObjectIdentifier(coroutine).hash(into: &hasher)
    }
    
}
