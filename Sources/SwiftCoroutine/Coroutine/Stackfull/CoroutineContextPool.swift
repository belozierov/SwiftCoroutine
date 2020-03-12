//
//  CoroutineContextPool.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Dispatch

final class StackfullCoroutineDispatcher {
    
    let stackSize, capacity: Int
    private let mutex = PsxLock()
    private var pool = [CoroutineContext]()
    
    public init(stackSize: Coroutine.StackSize, capacity: Int) {
        self.stackSize = stackSize.size
        self.capacity = capacity
    }
    
    func execute(on scheduler: TaskScheduler, task: @escaping () -> Void) {
        let context = getContext()
        context.block = task
        StackfullCoroutine(context: context, scheduler: scheduler) {
            self.push(context)
        }.start()
    }
    
    private func getContext() -> CoroutineContext {
        mutex.lock()
        let context = pool.popLast()
        mutex.unlock()
        return context ?? CoroutineContext(stackSize: stackSize)
    }
    
    private func push(_ context: CoroutineContext) {
        mutex.lock()
        if pool.count < capacity {
            pool.append(context)
        }
        mutex.unlock()
    }
    
    func reset() {
        mutex.perform { pool.removeAll() }
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
    
}
