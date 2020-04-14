//
//  SharedCoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

struct AtomicList {
    
    private var atomic = AtomicInt()
    
    mutating func decrease(at index: Int) -> Bool {
        update(at: index) { max(0, $0 - 1) }
    }
    
    mutating func increase(at index: Int) {
        update(at: index) { $0 + 1 }
    }
    
    mutating func increaseAll() {
        atomic.update {
            var value = $0
            withUnsafeMutableBytes(of: &value) {
                let pointer = $0.bindMemory(to: Int32.self)
                (0..<2).forEach { pointer[$0] += 1 }
            }
            return value
        }
        
    }
    
    mutating func decreaseAll() -> Bool {
        let (old, new) = atomic.update {
            var value = $0
            let success: Bool = withUnsafeMutableBytes(of: &value) {
                let pointer = $0.bindMemory(to: Int32.self)
                for i in 0..<2 {
                    guard pointer[i] > 0 else { return false }
                    pointer[i] -= 1
                }
                return true
            }
            return success ? value : $0
        }
        return old != new
    }
    
    @discardableResult
    mutating func update(at index: Int, updater: (Int32) -> Int32) -> Bool {
        let (old, new) = atomic.update {
            var value = $0
            withUnsafeMutableBytes(of: &value) {
                let pointer = $0.bindMemory(to: Int32.self).baseAddress! + index
                pointer.pointee = updater(pointer.pointee)
            }
            return value
        }
        return old != new
    }
    
}

internal final class SharedCoroutineStorage {
    
    typealias Task = (SharedCoroutineQueue) -> Void
    
    internal static let shared = SharedCoroutineStorage(capacity: .processorsNumber * 2,
                                                        stackSize: .recommended)
    
    private let capacity: Int
    private var tasks = ThreadSafeFifoQueues<Task>()
    private let queues: UnsafeMutablePointer<SharedCoroutineQueue>
    private var free = AtomicBitMask()
    private var suspended = AtomicBitMask()
    private var counters = AtomicList()
    
    internal init(capacity: Int, stackSize: Coroutine.StackSize) {
        self.capacity = min(63, capacity)
        queues = .allocate(capacity: self.capacity)
        (0..<self.capacity).forEach {
            counters.increase(at: 0)
            free.insert($0)
            (queues + $0).initialize(to: .init(storage: self, tag: $0,
                                               stackSize: stackSize.size))
        }
    }
    
    func addTask(_ task: @escaping Task) {
        if counters.decrease(at: 0) {
            if let queue = freeQueue { return task(queue) }
            counters.increase(at: 0)
        } else {
            tasks.push(task)
            counters.increase(at: 1)
        }
        tryToResume()
    }
    
    private var freeQueue: SharedCoroutineQueue? {
        if let index = free.pop() { return queues[index] }
        if let index = suspended.pop(offset: suspended.rawValue % capacity) {
            return queues[index]
        }
        return nil
    }
    
    internal func push(_ queue: SharedCoroutineQueue) {
        if counters.decrease(at: 1) {
            tasks.pop()(queue)
        } else if queue.started == 0 {
            free.insert(queue.tag)
            counters.increase(at: 0)
            tryToResume()
        } else {
            suspended.insert(queue.tag)
            counters.increase(at: 0)
            if !queue.hasPrepared {
                tryToResume()
            } else if removeSuspended(with: queue.tag) {
                queue.startPrepared()
            }
        }
    }
    
    private func tryToResume() {
        while counters.decreaseAll() {
            if let queue = freeQueue { return tasks.pop()(queue) }
            counters.increaseAll()
        }
    }
    
    internal func removeSuspended(with tag: Int) -> Bool {
        guard suspended.remove(tag) else { return false }
        counters.update(at: 0) { $0 - 1 }
        return true
    }
    
    deinit {
        queues.deinitialize(count: capacity)
        queues.deallocate()
    }
    
}

internal protocol SharedCoroutineDispatcher: class {
    
    func receiveQueue(_ queue: SharedCoroutineQueue)
    
}

@usableFromInline
internal final class UniversalSharedCoroutineDispatcher: SharedCoroutineDispatcher {
    
    private let storage: SharedCoroutineStorage
    
    internal init(storage: SharedCoroutineStorage) {
        self.storage = storage
    }
    
    @usableFromInline
    internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        storage.addTask { queue in
            scheduler.scheduleTask {
                queue.start(dispatcher: self, scheduler: scheduler, task: task)
            }
        }
    }
    
    internal func receiveQueue(_ queue: SharedCoroutineQueue) {
        storage.push(queue)
    }
    
}

//internal final class SharedSerialCoroutineDispatcher: _SharedCoroutineDispatcher {
//
//    private let storage: SharedCoroutineStorage
//    private var tasks = ThreadSafeFifoQueues<Task>()
//
//    internal init(storage: SharedCoroutineStorage) {
//        self.storage = storage
//    }
//
//}



//internal final class SharedCoroutineDispatcher: CoroutineTaskExecutor {
//
//    internal struct Task {
//        let scheduler: CoroutineScheduler, task: () -> Void
//    }
//
//    private let queuesCount: Int
//    private let queues: UnsafeMutablePointer<SharedCoroutineQueue>
//    private var freeQueuesMask = AtomicBitMask()
//    private var suspendedQueuesMask = AtomicBitMask()
//    private var tasks = ThreadSafeFifoQueues<Task>()
//
//    internal init(contextsCount: Int, stackSize: Int) {
//        queuesCount = min(contextsCount, 63)
//        queues = .allocate(capacity: queuesCount)
//        (0..<queuesCount).forEach {
//            freeQueuesMask.insert($0)
//            (queues + $0).initialize(to: .init(tag: $0, stackSize: stackSize))
//        }
//    }
//
//    // MARK: - Free
//
//    private var hasFree: Bool {
//        !freeQueuesMask.isEmpty || !suspendedQueuesMask.isEmpty
//    }
//
//    private var freeQueue: SharedCoroutineQueue? {
//        if !freeQueuesMask.isEmpty, let index = freeQueuesMask.pop() { return queues[index] }
//        if !suspendedQueuesMask.isEmpty, let index = suspendedQueuesMask
//            .pop(offset: suspendedQueuesMask.rawValue % queuesCount) {
//            return queues[index]
//        }
//        return nil
//    }
//
//    private func pushTask(_ task: Task) {
//        tasks.push(task)
//        if hasFree { tasks.pop().map(startTask) }
//    }
//
//    // MARK: - Start
//
//    internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
//        hasFree
//            ? startTask(.init(scheduler: scheduler, task: task))
//            : pushTask(.init(scheduler: scheduler, task: task))
//    }
//
//    private func startTask(_ task: Task) {
//        task.scheduler.scheduleTask {
//            if let queue = self.freeQueue {
//                queue.start(dispatcher: self, task: task)
//            } else {
//                self.pushTask(task)
//            }
//        }
//    }
//
//    // MARK: - Resume
//
//    internal func resume(_ coroutine: SharedCoroutine) {
//        coroutine.queue.mutex.lock()
//        if suspendedQueuesMask.remove(coroutine.queue.tag) {
//            coroutine.queue.mutex.unlock()
//            coroutine.resumeOnQueue()
//        } else {
//            coroutine.queue.prepared.push(coroutine)
//            coroutine.queue.mutex.unlock()
//        }
//    }
//
//    // MARK: - Next
//
//    internal func performNext(for queue: SharedCoroutineQueue) {
//        queue.mutex.lock()
//        if let coroutine = queue.prepared.pop() {
//            queue.mutex.unlock()
//            coroutine.resumeOnQueue()
//        } else {
//            queue.started == 0
//                ? freeQueuesMask.insert(queue.tag)
//                : suspendedQueuesMask.insert(queue.tag)
//            queue.mutex.unlock()
//            if hasFree { tasks.pop().map(startTask) }
//        }
//    }
//
//    deinit {
//        queues.deinitialize(count: queuesCount)
//        queues.deallocate()
//    }
//
//}
//
//extension SharedCoroutine {
//
//    fileprivate func resumeOnQueue() {
//        scheduler.scheduleTask { self.queue.resume(coroutine: self) }
//    }
//
//}
