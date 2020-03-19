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
    
    @usableFromInline typealias Scheduler = (@escaping () -> Void) -> Void
    
    @usableFromInline let scheduler: Scheduler
    @usableFromInline let getIsCurrent: () -> Bool
    
    @inlinable public init(scheduler: @escaping (@escaping () -> Void) -> Void,
                           isCurrent: @escaping () -> Bool = { false }) {
        self.scheduler = scheduler
        getIsCurrent = isCurrent
    }
    
    @inlinable public var isCurrent: Bool {
        getIsCurrent()
    }
    
    @inlinable public func execute(_ task: @escaping () -> Void) {
        scheduler(task)
    }
    
    @inlinable public func submit<T>(_ task: @escaping () throws -> T) -> CoFuture<T> {
        let promise = CoPromise<T>()
        scheduler { promise.complete(with: Result { try task() }) }
        return promise
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
        TaskScheduler(scheduler: queue.addOperation) { OperationQueue.current == queue }
    }
    
}
