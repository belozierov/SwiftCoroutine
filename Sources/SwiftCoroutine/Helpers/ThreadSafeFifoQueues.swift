//
//  ThreadSafeFifoQueues.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 06.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct ThreadSafeFifoQueues<T> {
    
    private struct Queue {
        
        let condition = PsxCondition()
        var queue = FifoQueue<T>()
        var waiting = 0
        
        mutating func push(_ item: T) {
            condition.lock()
            queue.push(item)
            if waiting != 0 { condition.signal() }
            condition.unlock()
        }
        
        mutating func pop() -> T? {
            condition.lock()
            repeat {
                if let item = queue.pop() {
                    condition.unlock()
                    return item
                }
                waiting += 1
                condition.wait()
                waiting -= 1
            } while true
        }
        
    }
    
    private let count: Int
    private let queues: UnsafeMutablePointer<Queue>
    private var pushIndex = AtomicInt(value: -1)
    private var popIndex = AtomicInt(value: -1)
    private var counter = AtomicInt(value: 0)
    
    @inlinable internal var isEmpty: Bool {
        counter.value == 0
    }
    
    internal init(count: Int = .processorsNumber) {
        self.count = count
        queues = .allocate(capacity: count)
        (0..<count).forEach { (queues + $0).initialize(to: .init()) }
    }
    
    internal mutating func push(_ item: T) {
        counter.increase()
        let index = pushIndex.update { $0 + 1 < count ? $0 + 1 : 0 }.new
        (queues + index).pointee.push(item)
    }
    
    internal mutating func pop() -> T? {
        if count == 0 { return nil }
        if counter.update({ max(0, $0 - 1) }).old == 0 { return nil }
        let index = popIndex.update { $0 + 1 < count ? $0 + 1 : 0 }.new
        return (queues + index).pointee.pop()
    }
    
    internal func free() {
        (0..<count).forEach { queues[$0].condition.free() }
        queues.deinitialize(count: count)
        queues.deallocate()
    }
    
}
