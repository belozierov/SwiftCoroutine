//
//  CoFuture+scheduler.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - scheduler
    
    @inlinable public func onCoroutineDispatcher(_ dispatcher: CoroutineDispatcher) -> CoFuture {
        flatMapResult { dispatcher.submit($0.get) }
    }
    
    @inlinable public func onTaskScheduler(_ scheduler: TaskScheduler) -> CoFuture {
        flatMapResult { scheduler.submit($0.get) }
    }
    
}
