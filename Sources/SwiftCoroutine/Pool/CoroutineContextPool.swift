//
//  CoroutineContextPool.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 29.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation
import Dispatch

//#if os(macOS)
//fileprivate typealias PressureSource = DispatchSourceMemoryPressure
//#else
//fileprivate typealias PressureSource = DispatchSourceMachReceive
//#endif

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
    
    private lazy var memoryPressureSource: String = {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])
        source.setEventHandler { [unowned self] in self.reset() }
        return source
    }()
    
    private func startDispatchSource() {
//        if #available(OSX 10.12, iOS 10.0, *) {
//            memoryPressureSource.activate()
//        } else {
//            memoryPressureSource.resume()
//        }
    }
    
}
