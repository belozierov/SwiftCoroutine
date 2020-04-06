//
//  TestFifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

struct AbcQueue<T> {
    
    private let mutex = PsxLock()
    private var queue = FifoQueue<T>()
    
    mutating func push(_ item: T) {
        mutex.lock()
        queue.push(item)
        mutex.unlock()
    }
    
    mutating func pop() -> T? {
        mutex.lock()
        defer { mutex.unlock() }
        return queue.pop()
    }
    
    func free() {
        mutex.free()
    }
    
}

struct AbcQueue2<T> {
    
    private let condition = PsxCondition()
    private var queue = FifoQueue<T>()
    private var counter = AtomicInt(value: 0)
    private var waiting = 0
    
    mutating func push(_ item: T) {
        counter.increase()
        condition.lock()
        queue.push(item)
        if waiting != 0 { condition.signal() }
        condition.unlock()
    }
    
    mutating func pop() -> T? {
        if counter.update({ max(0, $0 - 1) }).old == 0 { return nil }
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
    
    func free() {
        condition.free()
    }
    
}

class TestFifoQueue: XCTestCase {
    
    func test() {
        var mask = AtomicBitMask()
        mask.insert(3)
        mask.insert(60)
        mask.insert(5)
        XCTAssertEqual(mask.pop(offset: 4), 60)
        XCTAssertEqual(mask.pop(offset: 4), 3)
        XCTAssertEqual(mask.pop(offset: 4), 5)
    }
    
    func testThreadSafeFifoQueues3() {
        var queue = AbcQueue2<Int>()
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) { index in
                queue.push(index)
//                XCTAssertNotNil(queue.pop())
                _ = queue.pop()
            }
        }
        queue.free()
    }
    
    func testThreadSafeFifoQueues2() {
        var queue = AbcQueue<Int>()
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) { index in
                queue.push(index)
                _ = queue.pop()
            }
        }
        queue.free()
    }
    
    func testThreadSafeFifoQueues() {
        var queue = ThreadSafeFifoQueues<Int>()
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) { index in
                queue.push(index)
                _ = queue.pop()
//                XCTAssertNotNil(queue.pop())
            }
        }
        queue.free()
    }
    
    func testQueue() {
        var queue = FifoQueue<Int>()
        for i in 0..<100 { queue.push(i) }
        for i in 0..<50 { XCTAssertEqual(queue.pop(), i) }
        for i in 100..<200 { queue.push(i) }
        for i in 50..<200 { XCTAssertEqual(queue.pop(), i) }
        XCTAssertNil(queue.pop())
    }
    
}
