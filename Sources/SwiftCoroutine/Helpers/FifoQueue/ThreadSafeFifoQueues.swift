//
//  ThreadSafeFifoQueues.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 06.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct ThreadSafeFifoQueues<T> {
    
    private let number: Int32
    private let queues: UnsafeMutablePointer<BlockingFifoQueue<T>>
    private var atomic = AtomicTuple()
    
    internal var count: Int {
        Int(atomic.value.0)
    }
    
    internal init(number: Int = .processorsNumber) {
        self.number = Int32(number)
        queues = .allocate(capacity: number)
        (0..<number).forEach { (queues + $0).initialize(to: .init()) }
    }
    
    internal mutating func insertAtStart(_ item: T) {
        let (_, index) = atomic.update { count, index in
            if index - 1 < 0 {
                return (count + 1, number - 1)
            } else {
                return (count + 1, index - 1)
            }
        }.new
        (queues + Int(index)).pointee.insertAtStart(item)
    }
    
    internal mutating func push(_ item: T) {
        let (count, index) = atomic.update { count, index in
            (count + 1, index)
        }.old
        (queues + Int((count + index) % number)).pointee.push(item)
    }
    
    internal mutating func pop() -> T? {
        let (count, index) = atomic.update { count, index in
            if count > 0 {
                return (count - 1, index + 1 < number ? index + 1 : 0)
            } else {
                return (0, index)
            }
        }.old
        return count == 0 ? nil : (queues + Int(index)).pointee.pop()
    }
    
    internal func free() {
        (0..<Int(number)).forEach { queues[$0].free() }
        queues.deinitialize(count: Int(number))
        queues.deallocate()
    }
    
}
