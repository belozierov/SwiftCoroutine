//
//  Pool.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

class Pool<T> {
    
    private let mutex = NSLock()
    private let creator: () -> T
    private var _maxElements: Int?
    private var pool = [T]()
    
    init(maxElements: Int?, creator: @escaping () -> T) {
        self.creator = creator
        _maxElements = maxElements
        if #available(OSX 10.12, iOS 10.0, *) {
            memoryPressureSource.activate()
        }
    }
    
    func pop() -> T {
        mutex.lock()
        let coroutine = pool.popLast()
        mutex.unlock()
        return coroutine ?? creator()
    }
    
    func push(_ element: T) {
        mutex.lock()
        defer { mutex.unlock() }
        if let max = _maxElements,
            pool.count >= max { return }
        pool.append(element)
    }
    
    func reset() {
        mutex.lock()
        pool.removeAll()
        mutex.unlock()
    }
    
    // MARK: - MaxElements
    
    var maxElements: Int? {
        get {
            mutex.lock()
            defer { mutex.unlock() }
            return _maxElements
        }
        set {
            mutex.lock()
            _maxElements = newValue
            newValue
                .map { pool.prefix($0) }
                .map { pool = Array($0) }
            mutex.unlock()
        }
    }
    
    // MARK: - DispatchSourceMemoryPressure
    
    private lazy var memoryPressureSource: DispatchSourceMemoryPressure = {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])
        source.setEventHandler { [unowned self] in self.reset() }
        return source
    }()
    
}
