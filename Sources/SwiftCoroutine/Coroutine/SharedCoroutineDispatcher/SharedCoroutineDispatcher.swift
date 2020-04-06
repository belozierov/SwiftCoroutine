//
//  SharedCoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Dispatch

internal final class SharedCoroutineDispatcher: CoroutineTaskExecutor {
    
    internal struct Task {
        let scheduler: CoroutineScheduler, task: () -> Void
    }
    
    private var tasks = ThreadSafeFifoQueues<Task>()
    
    private let queuesCount: Int
    private let queues: UnsafeMutablePointer<SharedCoroutineQueue>
    private var freeQueuesMask = AtomicBitMask()
    private var suspendedQueuesMask = AtomicBitMask()
    
    private var hasFree: Bool {
        !freeQueuesMask.isEmpty || !suspendedQueuesMask.isEmpty
    }
    
    internal init(contextsCount: Int, stackSize: Int) {
        queuesCount = contextsCount
        queues = .allocate(capacity: contextsCount)
        (0..<contextsCount).forEach {
            freeQueuesMask.insert($0)
            (queues + $0).initialize(to: .init(tag: $0, stackSize: stackSize))
        }
        startDispatchSource()
    }
    
    // MARK: - Start
    
    internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        func perform() {
            if let queue = freeQueue {
                queue.start(dispatcher: self, task: .init(scheduler: scheduler, task: task))
            } else {
                tasks.push(.init(scheduler: scheduler, task: task))
            }
        }
        guard hasFree else {
            tasks.push(.init(scheduler: scheduler, task: task))
//            precondition(!hasFree)
            return
        }
        scheduler.scheduleTask(perform)
    }
    
    private var suspendedIterator = AtomicInt(value: 0)
    
    private var freeQueue: SharedCoroutineQueue? {
        if !freeQueuesMask.isEmpty, let index = freeQueuesMask.pop() { return queues[index] }
        if !suspendedQueuesMask.isEmpty,
            let index = suspendedQueuesMask.pop(offset: suspendedIterator.value % queuesCount) {
            suspendedIterator.increase()
            return queues[index]
        }
        return nil
    }
    
    // MARK: - Resume
    
    internal func resume(_ coroutine: SharedCoroutine) {
        coroutine.queue.mutex.lock()
        if suspendedQueuesMask.remove(coroutine.queue.tag) {
            coroutine.queue.mutex.unlock()
            coroutine.scheduler.scheduleTask {
                coroutine.queue.resume(coroutine: coroutine)
            }
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
            coroutine.scheduler.scheduleTask {
                queue.resume(coroutine: coroutine)
            }
        } else if let task = tasks.pop() {
            queue.mutex.unlock()
            task.scheduler.scheduleTask {
                queue.start(dispatcher: self, task: task)
            }
        } else {
            queue.started == 0
                ? freeQueuesMask.insert(queue.tag)
                : suspendedQueuesMask.insert(queue.tag)
            queue.mutex.unlock()
        }
    }
    
    // MARK: - DispatchSourceMemoryPressure
    
    #if os(Linux)
    
    private func startDispatchSource() {}
    
    #else
    
    private lazy var memoryPressureSource: DispatchSourceMemoryPressure = {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])
        source.setEventHandler { [unowned self] in self.reset() }
        return source
    }()
    
    private func startDispatchSource() {
        if #available(OSX 10.12, iOS 10.0, *) {
            memoryPressureSource.activate()
        } else {
            memoryPressureSource.resume()
        }
    }
    
    #endif
    
    internal func reset() {
//        mutex.lock()
//        contextsCount += freeQueues.count
//        freeCount.add(freeQueues.count)
//        freeQueues.removeAll(keepingCapacity: true)
//        mutex.unlock()
    }
    
}

