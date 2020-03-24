//
//  SharedCoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Dispatch

final class SharedCoroutineDispatcher: _CoroutineTaskExecutor {
    
    private struct Task {
        let scheduler: TaskScheduler, task: () -> Void
    }
    
    private let mutex = PsxLock()
    private let stackSize: Int
    private var contextsCount: Int
    private var freeQueues = [SharedCoroutineQueue]()
    private var suspendedQueues = Set<SharedCoroutineQueue>()
    private var tasks = FifoQueue<Task>()
    
    init(contextsCount: Int, stackSize: Int) {
        self.stackSize = stackSize
        self.contextsCount = contextsCount
        freeQueues.reserveCapacity(contextsCount)
        suspendedQueues.reserveCapacity(contextsCount)
        startDispatchSource()
    }
    
    func execute(on scheduler: TaskScheduler, task: @escaping () -> Void) {
        mutex.lock()
        if let queue = freeQueue {
            mutex.unlock()
            func perform() {
                start(task: .init(scheduler: scheduler, task: task), on: queue)
                performNext(for: queue)
            }
            scheduler.isCurrent() ? perform() : scheduler.execute(perform)
        } else {
            tasks.push(.init(scheduler: scheduler, task: task))
            mutex.unlock()
        }
    }
    
    private var freeQueue: SharedCoroutineQueue? {
        if let queue = freeQueues.popLast() { return queue }
        if contextsCount > 0 {
            contextsCount -= 1
            let context = CoroutineContext(stackSize: stackSize)
            return SharedCoroutineQueue(context: context)
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
    
    func resume(_ coroutine: SharedCoroutine) {
        mutex.lock()
        if suspendedQueues.remove(coroutine.queue) == nil {
            coroutine.queue.push(coroutine)
            mutex.unlock()
        } else {
            mutex.unlock()
            func perform() {
                coroutine.resume()
                performNext(for: coroutine.queue)
            }
            coroutine.scheduler.isCurrent() ? perform() : coroutine.scheduler.execute(perform)
        }
    }
    
    private func performNext(for queue: SharedCoroutineQueue) {
        while true {
            mutex.lock()
            if let coroutine = queue.pop() {
                mutex.unlock()
                if coroutine.scheduler.isCurrent() {
                    coroutine.resume()
                } else {
                    return coroutine.scheduler.execute {
                        coroutine.resume()
                        self.performNext(for: queue)
                    }
                }
            } else if let task = tasks.pop() {
                mutex.unlock()
                if task.scheduler.isCurrent() {
                    start(task: task, on: queue)
                } else {
                    return task.scheduler.execute {
                        self.start(task: task, on: queue)
                        self.performNext(for: queue)
                    }
                }
            } else if queue.started == 0 {
                freeQueues.append(queue)
                return mutex.unlock()
            } else {
                suspendedQueues.insert(queue)
                return mutex.unlock()
            }
        }
    }
    
    private func start(task: Task, on queue: SharedCoroutineQueue) {
        SharedCoroutine(dispatcher: self, queue: queue, scheduler: task.scheduler).start(task.task)
    }
    
    // MARK: - DispatchSourceMemoryPressure
    
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
    
    func reset() {
        mutex.lock()
        contextsCount += freeQueues.count
        freeQueues.removeAll(keepingCapacity: true)
        mutex.unlock()
    }
    
    deinit {
        mutex.free()
    }
    
}
