//
//  TestFifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class Aa<T> {
    
    private let queue = DispatchQueue(label: "asdasd", attributes: .concurrent)
    private var value = [T]()
    
    func push(_ item: T) {
        queue.async(flags: .barrier) {
            self.value.append(item)
        }
    }
    
    func pop() -> T? {
        queue.sync {
            if self.value.isEmpty { return nil }
            return self.value.removeFirst()
        }
    }
    
}

class TestFifoQueue: XCTestCase {
    
    func testThreadSafeFifoQueues() {
        let exp = expectation(description: "testThreadSafeFifoQueues")
        exp.expectedFulfillmentCount = 10
//        var queue = BlockingFifoQueue<Int>()
        var queue = ThreadSafeFifoQueues<Int>()
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) { index in
                queue.push(index)
                _ = queue.pop()
            }
            exp.fulfill()
        }
        queue.free()
        wait(for: [exp], timeout: 10)
    }
    
    func testThreadSafeFifoQueues2() {
        var queue = ThreadSafeFifoQueues<Int>()
        queue.push(0)
        queue.push(3)
        XCTAssertEqual(queue.pop(), 0)
        queue.insertAtStart(2)
        queue.insertAtStart(1)
        queue.push(4)
        queue.push(5)
        queue.insertAtStart(0)
        for i in 0..<6 {
            XCTAssertEqual(queue.pop(), i)
        }
    }
    
    func testQueue() {
        let exp = expectation(description: "testQueue")
        DispatchQueue.global().async {
            var queue = BlockingFifoQueue<Int>()
            for i in 0..<100 { queue.push(i) }
            for i in 0..<50 { XCTAssertEqual(queue.pop(), i) }
            for i in 100..<200 { queue.push(i) }
            for i in 50..<200 { XCTAssertEqual(queue.pop(), i) }
            queue.free()
            exp.fulfill()
        }
        wait(for: [exp], timeout: 10)
    }
    
}
