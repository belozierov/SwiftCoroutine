//
//  Pool.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class Pool<T> {
    
    private let mutex = NSLock()
    private let creator: () -> T
    private var pool = [T]()
    
    public init(creator: @escaping () -> T) {
        self.creator = creator
        if #available(OSX 10.12, iOS 10.0, *) {
            memoryPressureSource.activate()
        }
    }
    
    open func pop() -> T {
        mutex.lock()
        let coroutine = pool.popLast()
        mutex.unlock()
        return coroutine ?? creator()
    }
    
    open func push(_ element: T) {
        mutex.lock()
        pool.append(element)
        mutex.unlock()
    }
    
    open func reset() {
        mutex.lock()
        pool.removeAll()
        mutex.unlock()
    }
    
    // MARK: - DispatchSourceMemoryPressure
    
    private lazy var memoryPressureSource: DispatchSourceMemoryPressure = {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])
        source.setEventHandler { [unowned self] in self.reset() }
        return source
    }()
    
}
