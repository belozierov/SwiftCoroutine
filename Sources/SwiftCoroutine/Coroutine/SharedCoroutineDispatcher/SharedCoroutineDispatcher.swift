//
//  SharedCoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@usableFromInline
internal final class SharedCoroutineDispatcher: CoroutineTaskExecutor {
    
    // MARK: - Push
    
    private let stackSize, capacity: Int
    private var lifo = LifoQueue<SharedCoroutineQueue>()
    private var fifo = FifoQueue<SharedCoroutineQueue>()
    private var counter = 0
    
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
        while let queue = popQueue() {
            queue.inQueue = false
            if queue.occupy() { return queue }
        }
        return SharedCoroutineQueue(stackSize: stackSize)
    }
    
    internal func push(_ queue: SharedCoroutineQueue) {
        if queue.started != 0 {
            if queue.inQueue || !increaseCounter() { return }
            queue.inQueue = true
            fifo.push(queue)
        } else if increaseCounter() {
            lifo.push(queue)
        }
    }
    
    // MARK: - Queue
    
    private func popQueue() -> SharedCoroutineQueue? {
        let old = atomicUpdate(&counter) { max(0, $0 - 1) }.old
        if old == 0 { return nil }
        return lifo.pop() ?? fifo.pop()
    }
    
    private func increaseCounter() -> Bool {
        atomicUpdate(&counter) { min(capacity, $0 + 1) }.old < capacity
    }
    
    deinit {
        lifo.free()
        fifo.free()
    }
    
}
