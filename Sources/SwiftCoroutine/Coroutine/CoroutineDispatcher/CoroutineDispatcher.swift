//
//  CoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@usableFromInline internal protocol CoroutineTaskExecutor: class {
    
    func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void)
    
}

@usableFromInline internal struct CoroutineDispatcher {
    
    @usableFromInline
    internal static let `default` = newShared(capacity: .processorsNumber * 2)
    
    internal static func newShared(capacity: Int, stackSize: Coroutine.StackSize = .recommended) -> CoroutineDispatcher {
        let executor = SharedCoroutineDispatcher(capacity: capacity, stackSize: stackSize)
        return CoroutineDispatcher(executor: executor)
    }
    
    @usableFromInline let executor: CoroutineTaskExecutor
    
    @inlinable internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        executor.execute(on: scheduler, task: task)
    }
    
}
