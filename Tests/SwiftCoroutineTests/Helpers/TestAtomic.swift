//
//  TestAtomic.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
import Dispatch
@testable import SwiftCoroutine

struct AtomicList<T> {
    
    private struct Node {
        let value: T
        var next = 0
    }
    
    private var head = AtomicInt()
    
    mutating func push(_ value: T) {
        let pointer = UnsafeMutablePointer<Node>.allocate(capacity: 1)
        pointer.initialize(to: Node(value: value))
        head.update {
            pointer.pointee.next = $0
            return Int(bitPattern: pointer)
        }
    }
    
    var count: Int {
        var count = 0
        for _ in self { count += 1 }
        return count
    }
    
}

extension AtomicList: Sequence {
    
    func makeIterator() -> AnyIterator<T> {
        var address = head.value
        return AnyIterator {
            guard address > 0, let pointer = UnsafePointer<Node>(bitPattern: address)
                else { return nil }
            address = pointer.pointee.next
            return pointer.pointee.value
        }
    }
    
}

struct ThreadSafeList<T> {
    
    private let mutex = PsxLock()
    private var array = [T]()
    
    mutating func push(_ value: T) {
        mutex.lock()
        array.append(value)
        mutex.unlock()
    }
    
    var count: Int {
        array.count
    }
    
}

class TestAtomic: XCTestCase {
    
    func testAbc() {
        var list = AtomicList<Int>()
        measure {
//            for i in 0..<100_000 {
//                list.push(i)
//            }
            DispatchQueue.concurrentPerform(iterations: 100_000) {
                list.push($0)
            }
        }
        print(list.count)
    }
    
    func testTuple() {
        var tuple = AtomicTuple()
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            tuple.update { ($0.0 + 1, $0.1) }
            tuple.update { ($0.0, $0.1 + 1) }
        }
        XCTAssertEqual(tuple.value.0, 100_000)
        XCTAssertEqual(tuple.value.1, 100_000)
    }
    
    func testTuple2() {
        var atomic = AtomicTuple(), counter = 0
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            while true {
                if atomic.update(keyPath: \.0, with: 1) == 0 {
                    counter += 1
                    atomic.value.0 = 0
                    break
                }
            }
        }
        XCTAssertEqual(counter, 100_000)
    }
    
    func testIntUpdate() {
        var atomic = AtomicInt(value: 0)
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            atomic.update { $0 + 1 }
        }
        XCTAssertEqual(atomic.value, 100_000)
    }
    
    func testIntIncrease() {
        var atomic = AtomicInt(value: 0)
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            atomic.add(1)
        }
        XCTAssertEqual(atomic.value, 100_000)
    }
    
}
