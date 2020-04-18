//
//  SharedCoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@usableFromInline
internal final class SharedCoroutineDispatcher: CoroutineTaskExecutor {
    
    private let stackSize, capacity: Int
    private var queues: ThreadSafeFifoQueues<SharedCoroutineQueue>
    
    internal init(capacity: Int, stackSize: Coroutine.StackSize) {
        self.stackSize = stackSize.size
        self.capacity = capacity
        queues = .init(number: capacity)
    }
    
    @usableFromInline
    internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        scheduler.scheduleTask {
            self.getFreeQueue().start(dispatcher: self, scheduler: scheduler, task: task)
        }
    }
    
    private func getFreeQueue() -> SharedCoroutineQueue {
        while let queue = queues.pop() {
            queue.inQueue = false
            if queue.occupy() { return queue }
        }
        return SharedCoroutineQueue(stackSize: stackSize)
    }
    
    internal func push(_ queue: SharedCoroutineQueue) {
        if queue.started != 0 {
            if queue.inQueue { return }
            queue.inQueue = true
            queues.push(queue)
        } else if queues.count < capacity {
            queues.insertAtStart(queue)
        }
    }
    
    deinit {
        queues.free()
    }
    
}
