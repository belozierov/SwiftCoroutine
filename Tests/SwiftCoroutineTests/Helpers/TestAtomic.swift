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

class TestAtomic: XCTestCase {
    
    func testAaa3() {
        var atomic = 0
        measure {
            for _ in 0..<1_000_000 {
                atomicStore(&atomic, value: atomic + 1)
            }
        }
    }
    
    func testAaa() {
        var atomic = 0
        measure {
            for _ in 0..<1_000_000 {
                _ = atomicExchange(&atomic, with: atomic + 1)
            }
        }
    }
    
    func testAaa2() {
        var atomic = 0
        measure {
            for _ in 0..<1_000_000 {
                while true {
                    let a = atomic
                    if atomicCAS(&atomic, expected: a, desired: a + 1) {
                        break
                    }
                }
            }
        }
    }
    
    func testAbc3() {
        var atomic = 0
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
                atomicAdd(&atomic, value: 1)
            }
        }
        XCTAssertEqual(atomic, 1_000_000)
    }
    
    func testAbc4() {
        var atomic = 0
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
                while true {
                    var value = atomic
                    let result = withUnsafeMutablePointer(to: &atomic) {
                        __atomicCompareExchange(OpaquePointer($0), &value, value + 1)
                    }
                    if result != 0 { return }
                }
            }
        }
        XCTAssertEqual(atomic, 1_000_000)
    }
    
    func testAbc2() {
        var atomic = 0
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
                while true {
                    let value = atomic
                    if atomicCAS(&atomic, expected: value, desired: value + 1) {
                        return
                    }
                }
            }
        }
        XCTAssertEqual(atomic, 1_000_000)
    }
    
    func testAbc1() {
        var atomic = 0
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
                atomicUpdate(&atomic) { $0 + 1 }
            }
        }
        XCTAssertEqual(atomic, 1_000_000)
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
        var atomic = 0
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            atomicUpdate(&atomic) { $0 + 1 }
        }
        XCTAssertEqual(atomic, 100_000)
    }
    
    func testIntIncrease() {
        var atomic = 0
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            atomicAdd(&atomic, value: 1)
        }
        XCTAssertEqual(atomic, 100_000)
    }
    
}
