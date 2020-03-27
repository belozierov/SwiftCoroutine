//
//  CoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 30.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public struct CoroutineDispatcher {
    
    public static let main = CoroutineDispatcher(scheduler: .main)
    public static let global = CoroutineDispatcher(scheduler: .global)
    
    @usableFromInline let scheduler: TaskScheduler
    @usableFromInline let executor: _CoroutineTaskExecutor
    
    @inlinable public init(scheduler: TaskScheduler,
                           executor: CoroutineTaskExecutor = .defaultShared) {
        self.scheduler = scheduler
        self.executor = executor.executor
    }
    
    @inlinable public func execute(_ task: @escaping () -> Void) {
        executor.execute(on: scheduler, task: task)
    }
    
}


