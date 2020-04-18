//
//  CoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@usableFromInline protocol CoroutineTaskExecutor: class {
    
    func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void)
    
}

@usableFromInline internal struct CoroutineDispatcher {
    
    @usableFromInline
    internal static let `default` = SharedCoroutineDispatcher(capacity: .processorsNumber * 2,
                                                  stackSize: .recommended)
    
//    internal static func newShared(coroutinePoolSize poolSize: Int, stackSize: Coroutine.StackSize = .recommended) -> CoroutineDispatcher {
//        let executor = SharedCoroutineDispatcher(
//        return CoroutineDispatcher(executor: executor)
//    }
    
    @usableFromInline let executor: CoroutineTaskExecutor
    
    @inlinable internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        executor.execute(on: scheduler, task: task)
    }
    
}
