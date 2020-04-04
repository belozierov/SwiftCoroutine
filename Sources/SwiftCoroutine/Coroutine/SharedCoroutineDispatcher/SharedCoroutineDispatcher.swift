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
    
    private let mutex = PsxLock()
    private let stackSize: Int
    private var tasks = FifoQueue<Task>()
    
    private var contextsCount: Int
    private var freeQueues = [SharedCoroutineQueue]()
    private var suspendedQueues = Set<SharedCoroutineQueue>()
    private var freeCount: AtomicInt
    
    internal init(contextsCount: Int, stackSize: Int) {
        self.stackSize = stackSize
        self.contextsCount = contextsCount
        freeCount = AtomicInt(value: contextsCount)
        freeQueues.reserveCapacity(contextsCount)
        suspendedQueues.reserveCapacity(contextsCount)
        startDispatchSource()
    }
    
    // MARK: - Start
    
    internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        func perform() {
            freeCount.update { max(0, $0 - 1) }
            mutex.lock()
            if let queue = freeQueue {
                mutex.unlock()
                queue.start(dispatcher: self, task: .init(scheduler: scheduler, task: task))
            } else {
                tasks.push(.init(scheduler: scheduler, task: task))
                mutex.unlock()
            }
        }
        if freeCount.value == 0 {
            mutex.lock()
            defer { mutex.unlock() }
            if freeCount.value == 0 {
                return tasks.push(.init(scheduler: scheduler, task: task))
            }
        }
        scheduler.scheduleTask(perform)
    }
    
    private var freeQueue: SharedCoroutineQueue? {
        if let queue = freeQueues.popLast() { return queue }
        if contextsCount > 0 {
            contextsCount -= 1
            return SharedCoroutineQueue(stackSize: stackSize)
        } else if suspendedQueues.count < 2 {
            return suspendedQueues.popFirst()
        }
        var min = suspendedQueues.first!
        for queue in suspendedQueues {
            if queue.started == 1 {
                return suspendedQueues.remove(queue)
            } else if queue.started < min.started {
                min = queue
            }
        }
        return suspendedQueues.remove(min)
    }
    
    // MARK: - Resume
    
    internal func resume(_ coroutine: SharedCoroutine) {
        mutex.lock()
        if suspendedQueues.remove(coroutine.queue) == nil {
            coroutine.queue.push(coroutine)
            mutex.unlock()
        } else {
            mutex.unlock()
            freeCount.decrease()
            coroutine.scheduler.scheduleTask {
                coroutine.queue.resume(coroutine: coroutine)
            }
        }
    }
    
    // MARK: - Next
    
    internal func performNext(for queue: SharedCoroutineQueue) {
        mutex.lock()
        if let coroutine = queue.pop() {
            mutex.unlock()
            coroutine.scheduler.scheduleTask {
                queue.resume(coroutine: coroutine)
            }
        } else if let task = tasks.pop() {
            mutex.unlock()
            task.scheduler.scheduleTask {
                queue.start(dispatcher: self, task: task)
            }
        } else {
            if queue.started == 0 {
                freeQueues.append(queue)
            } else {
                suspendedQueues.insert(queue)
            }
            freeCount.increase()
            mutex.unlock()
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
        mutex.lock()
        contextsCount += freeQueues.count
        freeCount.add(freeQueues.count)
        freeQueues.removeAll(keepingCapacity: true)
        mutex.unlock()
    }
    
    deinit {
        mutex.free()
    }
    
}

