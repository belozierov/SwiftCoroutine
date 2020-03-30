//
//  SharedCoroutineDispatcher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Dispatch

internal final class SharedCoroutineDispatcher: _CoroutineTaskExecutor {
    
    private struct Task {
        let scheduler: CoroutineScheduler, task: () -> Void
    }
    
    private let mutex = PsxLock()
    private let stackSize: Int
    private var contextsCount: Int
    private var freeQueues = [SharedCoroutineQueue]()
    private var suspendedQueues = Set<SharedCoroutineQueue>()
    private var tasks = FifoQueue<Task>()
    
    internal init(contextsCount: Int, stackSize: Int) {
        self.stackSize = stackSize
        self.contextsCount = contextsCount
        freeQueues.reserveCapacity(contextsCount)
        suspendedQueues.reserveCapacity(contextsCount)
        startDispatchSource()
    }
    
    internal func execute(on scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        mutex.lock()
        if let queue = freeQueue {
            mutex.unlock()
            scheduler.scheduleTask {
                self.start(task: .init(scheduler: scheduler, task: task), on: queue)
                self.performNext(for: queue)
            }
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
    
    internal func resume(_ coroutine: SharedCoroutine) {
        mutex.lock()
        if suspendedQueues.remove(coroutine.queue) == nil {
            coroutine.queue.push(coroutine)
            mutex.unlock()
        } else {
            mutex.unlock()
            coroutine.scheduler.scheduleTask {
                coroutine.resume()
                self.performNext(for: coroutine.queue)
            }
        }
    }
    
    private enum NextState: Int {
        case running, none
    }
    
    private func performNext(for queue: SharedCoroutineQueue) {
        var state = AtomicEnum(value: NextState.none)
        while true {
            mutex.lock()
            if let coroutine = queue.pop() {
                mutex.unlock()
                state.value = .running
                coroutine.scheduler.scheduleTask {
                    coroutine.resume()
                    if state.update(.none) == .running { return }
                    self.performNext(for: queue)
                }
            } else if let task = tasks.pop() {
                mutex.unlock()
                state.value = .running
                task.scheduler.scheduleTask {
                    self.start(task: task, on: queue)
                    if state.update(.none) == .running { return }
                    self.performNext(for: queue)
                }
            } else if queue.started == 0 {
                freeQueues.append(queue)
                return mutex.unlock()
            } else {
                suspendedQueues.insert(queue)
                return mutex.unlock()
            }
            if state.update(.none) == .running { return }
        }
    }
    
    private func start(task: Task, on queue: SharedCoroutineQueue) {
        SharedCoroutine(dispatcher: self, queue: queue, scheduler: task.scheduler).start(task.task)
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
        freeQueues.removeAll(keepingCapacity: true)
        mutex.unlock()
    }
    
    deinit {
        mutex.free()
    }
    
}
