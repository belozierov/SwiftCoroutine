//
//  CoroutineContextPool.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 29.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation
import Dispatch

class CoroutineContextPool {
    
    typealias StackSize = Coroutine.StackSize
    
    private let mutex = NSLock()
    private let pool: CostIdentifierPool<Int, CoroutineContext>
    
    init(stackSizeLimit: StackSize) {
        pool = CostIdentifierPool(costLimit: stackSizeLimit.size)
        startDispatchSource()
    }
    
    // MARK: - Pool
    
    @inlinable func pop(stackSize: StackSize) -> CoroutineContext {
        mutex.lock()
        let context = pool.pop(stackSize.size)
        mutex.unlock()
        return context ?? CoroutineContext(stackSize: stackSize.size)
    }
    
    @inlinable func push(_ context: CoroutineContext) {
        mutex.lock()
        pool.push(context, for: context.stackSize)
        mutex.unlock()
    }
    
    @inlinable func reset() {
        mutex.lock()
        pool.removeAll()
        mutex.unlock()
    }
    
    // MARK: - DispatchSourceMemoryPressure

    // We can ignore this on Linux because we never actually get a memory warning. Instead the OS just kills the process.
    // https://forums.swift.org/t/what-should-i-use-in-place-of-dispatchsource-makememorypressuresource-on-linux/32936/2
    #if os(macOS)
    private lazy var memoryPressureSource: DispatchSourceMemoryPressure = {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])
        source.setEventHandler { [unowned self] in self.reset() }
        return source
    }()
    #endif
    
    private func startDispatchSource() {
        #if os(macOS)
        if #available(OSX 10.12, iOS 10.0, *) {
            memoryPressureSource.activate()
        } else {
            memoryPressureSource.resume()
        }
        #endif
    }
    
}
