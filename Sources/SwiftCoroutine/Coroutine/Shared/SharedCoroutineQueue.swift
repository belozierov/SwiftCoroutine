//
//  SharedCoroutineQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal final class SharedCoroutineQueue {
    
    internal let context: CoroutineContext
    private var prepared = FifoQueue<SharedCoroutine>()
    private var suspendedCoroutine: SharedCoroutine?
    private(set) var started = 0
    var isFree = true
    
    internal init(context: CoroutineContext) {
        self.context = context
    }
    
    internal func push(_ coroutine: SharedCoroutine) {
        prepared.push(coroutine)
    }
    
    internal func pop() -> SharedCoroutine? {
        prepared.pop()
    }
    
    internal func start(_ task: @escaping () -> Void) -> Bool {
        started += 1
        suspendedCoroutine?.saveStack()
        suspendedCoroutine = nil
        context.block = task
        let isFinished = context.start()
        if isFinished { started -= 1 }
        return isFinished
    }
    
    @inlinable internal func suspend(_ coroutine: SharedCoroutine) {
        suspendedCoroutine = coroutine
        context.suspend(to: coroutine.environment)
    }
    
    internal func resume(_ coroutine: SharedCoroutine) -> Bool {
        if suspendedCoroutine !== coroutine {
            suspendedCoroutine?.saveStack()
            coroutine.restoreStack()
        }
        suspendedCoroutine = nil
        let isFinished = context.resume(from: coroutine.environment.pointee.env)
        if isFinished { started -= 1 }
        return isFinished
    }
    
}

extension SharedCoroutineQueue: Hashable {
    
    @inlinable internal static func == (lhs: SharedCoroutineQueue, rhs: SharedCoroutineQueue) -> Bool {
        lhs === rhs
    }
    
    @inlinable internal func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}
