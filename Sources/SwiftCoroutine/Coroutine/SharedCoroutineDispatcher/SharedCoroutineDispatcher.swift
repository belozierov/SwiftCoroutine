//
//  SharedCoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@usableFromInline internal final class SharedCoroutineDispatcher {
    
    @usableFromInline internal
    static let `default` = SharedCoroutineDispatcher(capacity: .processorsNumber * 2,
                                                     stackSize: .recommended)
    
    private let stackSize, capacity: Int
    private var queues = FifoQueue<SharedCoroutineQueue>()
    private var queuesCount = 0
    
    internal init(capacity: Int, stackSize: Coroutine.StackSize) {
        self.stackSize = stackSize.size
        self.capacity = capacity
    }
    
    @usableFromInline
    internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        scheduler.scheduleTask {
            self.getFreeQueue().start(dispatcher: self, scheduler: scheduler, task: task)
        }
    }
    
    private func getFreeQueue() -> SharedCoroutineQueue {
        while let queue = queues.pop() {
            atomicAdd(&queuesCount, value: -1)
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
            atomicAdd(&queuesCount, value: 1)
        } else if queuesCount < capacity {
            queues.insertAtStart(queue)
            atomicAdd(&queuesCount, value: 1)
        }
    }
    
    deinit {
        queues.free()
    }
    
}
