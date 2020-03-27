//
//  TaskScheduler.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Foundation

public struct TaskScheduler {
    
    public static let main = TaskScheduler(scheduler: RunLoop.main.perform,
                                           isCurrent: { pthread_main_np() != 0 })
    
    public static let global: TaskScheduler = {
        let queue = OperationQueue()
        queue.underlyingQueue = .global()
        queue.maxConcurrentOperationCount = .processorsNumber
        return .operationQueue(queue)
    }()
    
    public static let immediate = TaskScheduler(scheduler: { $0() }, isCurrent: { true })
    
    @usableFromInline internal let scheduler: (@escaping () -> Void) -> Void
    @usableFromInline internal let isCurrent: () -> Bool
    
    @inlinable public init(scheduler: @escaping (@escaping () -> Void) -> Void,
                           isCurrent: @escaping () -> Bool = { false }) {
        self.scheduler = scheduler
        self.isCurrent = isCurrent
    }
    
    @inlinable public func execute(_ block: @escaping () -> Void) {
        scheduler(block)
    }
    
    @inlinable internal func executeWithCheckIfCurrent(_ block: @escaping () -> Void) {
        isCurrent() ? block() : scheduler(block)
    }
    
    // MARK: - RunLoop
    
    @inlinable public static func runLoop(_ runLoop: RunLoop) -> TaskScheduler {
        TaskScheduler(scheduler: runLoop.perform) { RunLoop.current == runLoop }
    }
    
    // MARK: - DispatchQueue
    
    @inlinable public static func dispatchQueue(_ queue: DispatchQueue, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], group: DispatchGroup? = nil) -> TaskScheduler {
        TaskScheduler(scheduler: {
            queue.async(group: group, qos: qos, flags: flags, execute: $0)
        })
    }
    
    // MARK: - OperationQueue
    
    @inlinable public static func operationQueue(_ queue: OperationQueue) -> TaskScheduler {
        TaskScheduler(scheduler: { task in
            autoreleasepool { queue.addOperation(task) }
        }, isCurrent: { OperationQueue.current == queue })
    }
    
}
