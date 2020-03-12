//
//  CoroutineTaskExecutor.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@usableFromInline protocol _CoroutineTaskExecutor: class {
    
    func execute(on scheduler: TaskScheduler, task: @escaping () -> Void)
    
}

public struct CoroutineTaskExecutor {
    
    public static let defaultShared = newShared(coroutinePoolSize: .processorsNumber)
    
    public static func newShared(coroutinePoolSize poolSize: Int, stackSize: Coroutine.StackSize = .recommended) -> CoroutineTaskExecutor {
        let executor = SharedCoroutineDispatcher(contextsCount: poolSize, stackSize: stackSize.size)
        return CoroutineTaskExecutor(executor: executor)
    }
    
    @usableFromInline let executor: _CoroutineTaskExecutor
    
    @inlinable public func execute(on scheduler: TaskScheduler, task: @escaping () -> Void) {
        executor.execute(on: scheduler, task: task)
    }
    
}
