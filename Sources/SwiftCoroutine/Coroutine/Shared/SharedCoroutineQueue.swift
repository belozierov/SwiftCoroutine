//
//  SharedCoroutineQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

final class SharedCoroutineQueue {
    
    let context: CoroutineContext
    private var prepared = FifoQueue<SharedCoroutine>()
    private var suspendedCoroutine: SharedCoroutine?
    private(set) var started = 0
    
    init(context: CoroutineContext) {
        self.context = context
    }
    
    func push(_ coroutine: SharedCoroutine) {
        prepared.push(coroutine)
    }
    
    func pop() -> SharedCoroutine? {
        prepared.pop()
    }
    
    func start(_ task: @escaping () -> Void) -> Bool {
        started += 1
        suspendedCoroutine?.saveStack()
        suspendedCoroutine = nil
        context.block = task
        let isFinished = context.start()
        if isFinished { started -= 1 }
        return isFinished
    }
    
    @inlinable func suspend(_ coroutine: SharedCoroutine) {
        suspendedCoroutine = coroutine
        context.suspend(to: coroutine.environment)
    }
    
    func resume(_ coroutine: SharedCoroutine) -> Bool {
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
    
    static func == (lhs: SharedCoroutineQueue, rhs: SharedCoroutineQueue) -> Bool {
        lhs === rhs
    }
    
    func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}
