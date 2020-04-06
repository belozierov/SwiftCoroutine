//
//  SharedCoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal final class SharedCoroutineDispatcher: CoroutineTaskExecutor {
    
    internal struct Task {
        let scheduler: CoroutineScheduler, task: () -> Void
    }
    
    private let queuesCount: Int
    private var tasks = ThreadSafeFifoQueues<Task>()
    private let queues: UnsafeMutablePointer<SharedCoroutineQueue>
    private var freeQueuesMask = AtomicBitMask()
    private var suspendedQueuesMask = AtomicBitMask()
    
    internal init(contextsCount: Int, stackSize: Int) {
        queuesCount = min(contextsCount, 64)
        queues = .allocate(capacity: queuesCount)
        (0..<queuesCount).forEach {
            freeQueuesMask.insert($0)
            (queues + $0).initialize(to: .init(tag: $0, stackSize: stackSize))
        }
    }
    
    // MARK: - Free
    
    private var hasFree: Bool {
        !freeQueuesMask.isEmpty || !suspendedQueuesMask.isEmpty
    }
    
    private var freeQueue: SharedCoroutineQueue? {
        if !freeQueuesMask.isEmpty, let index = freeQueuesMask.pop() { return queues[index] }
        if !suspendedQueuesMask.isEmpty, let index = suspendedQueuesMask
            .pop(offset: suspendedQueuesMask.rawValue % queuesCount) {
            return queues[index]
        }
        return nil
    }
    
    // MARK: - Start
    
    internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        if hasFree {
            start(task: .init(scheduler: scheduler, task: task))
        } else {
            tasks.push(.init(scheduler: scheduler, task: task))
            if hasFree { tasks.pop().map(start) }
        }
    }
    
    private func start(task: Task) {
        task.scheduler.scheduleTask {
            if let queue = self.freeQueue {
                queue.start(dispatcher: self, task: task)
            } else {
                self.tasks.push(task)
            }
        }
    }
    
    // MARK: - Resume
    
    internal func resume(_ coroutine: SharedCoroutine) {
        coroutine.queue.mutex.lock()
        if suspendedQueuesMask.remove(coroutine.queue.tag) {
            coroutine.queue.mutex.unlock()
            coroutine.resumeOnQueue()
        } else {
            coroutine.queue.prepared.push(coroutine)
            coroutine.queue.mutex.unlock()
        }
    }
    
    // MARK: - Next
    
    internal func performNext(for queue: SharedCoroutineQueue) {
        queue.mutex.lock()
        if let coroutine = queue.prepared.pop() {
            queue.mutex.unlock()
            coroutine.resumeOnQueue()
        } else {
            queue.started == 0
                ? freeQueuesMask.insert(queue.tag)
                : suspendedQueuesMask.insert(queue.tag)
            queue.mutex.unlock()
            if hasFree { tasks.pop().map(start) }
        }
    }
    
    deinit {
        queues.deinitialize(count: queuesCount)
        queues.deallocate()
    }
    
}

extension SharedCoroutine {
    
    fileprivate func resumeOnQueue() {
        scheduler.scheduleTask { self.queue.resume(coroutine: self) }
    }
    
}
