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

//struct AtomicButter<T> {
//    
//    private struct Box {
//        
//        private enum State: Int {
//            case hasValue, isFree, busy
//        }
//        
//        private var value: T?
//        private var hasValue = AtomicEnum(value: State.isFree)
//        
//        mutating func pop() -> T {
//            while hasValue.update(.isFree) != .hasValue {
////                Thread
//            }
//            return value!
//        }
//        
//        mutating func push(_ item: T) {
//            while hasValue.update(.hasValue) != .isFree {
//            //                Thread
//            }
//            
//        }
//        
//    }
//    
//    private let capacity: Int32
//    private let buffer: UnsafeMutablePointer<T>
//    private var atomic = AtomicInt()
//    
//    init(capacity: Int) {
//        self.capacity = Int32(capacity)
//        buffer = .allocate(capacity: capacity)
//    }
//    
//    mutating func push(_ item: T) {
//        var index: Int32!
//        atomic.update {
//            var value = $0
//            withUnsafeMutableBytes(of: &value) {
//                let pointer = $0.bindMemory(to: Int32.self)
//                if pointer[0] == capacity { index = nil; return }
//                index = pointer[0] + pointer[1]
//                if index >= capacity { index -= capacity }
//                pointer[0] += 1
//            }
//            return value
//        }
//        
//    }
//    
//    mutating func pop() -> T? {
//        
//    }
//    
//}

class TestAtomic: XCTestCase {
    
//    func testList() {
//        var list = List<Int>()
//        DispatchQueue.concurrentPerform(iterations: 100) { index in
//            list.push(index)
//            XCTAssertNotNil(list.pop())
//        }
//        XCTAssertNil(list.pop())
//    }
    
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
    
    enum State: Int {
        case free, blocked
    }
    
    func testEnumUpdate() {
        var atomic = AtomicEnum(value: State.free), counter = 0
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            while true {
                if atomic.update(.blocked) == .free {
                    counter += 1
                    atomic.value = .free
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
