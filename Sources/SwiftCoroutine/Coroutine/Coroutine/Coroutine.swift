//
//  Coroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 01.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public struct Coroutine {
    
    public enum State: Int {
        case prepared, running, suspending, suspended, finished
    }
    
    @usableFromInline let coroutine: CoroutineProtocol
    
    @usableFromInline init(coroutine: CoroutineProtocol) {
        self.coroutine = coroutine
    }
    
    public init(stackSize: StackSize = .recommended, scheduler: TaskScheduler = .immediate, task: @escaping () -> Void) {
        let context = CoroutineContext(stackSize: stackSize.size)
        coroutine = StackfullCoroutine(context: context, scheduler: scheduler)
    }
    
    @inlinable public var state: State {
        coroutine.state
    }
    
    // MARK: - resume
    
    @inlinable public func resume() throws {
        switch coroutine.state {
        case .prepared: coroutine.start()
        case .suspended: coroutine.resume()
        default: throw CoroutineError.wrongState
        }
    }
    
    // MARK: - suspend
    
    @inlinable public func suspend() throws {
        try validateSuspend()
        coroutine.suspend()
    }
    
    @inlinable public func suspend(with completion: @escaping () -> Void) throws {
        try validateSuspend()
        coroutine.suspend(with: completion)
    }
    
    @inlinable func validateSuspend() throws {
        guard isCurrent else { throw CoroutineError.mustBeCalledInsideCoroutine }
        guard state == .running else { throw CoroutineError.wrongState }
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
